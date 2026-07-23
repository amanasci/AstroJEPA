"""
Dataset and streaming data loading for AstroJEPA.

Provides:
- GalaxyStreamDataset: wraps HuggingFace streaming dataset for galaxies
- StreamingDataLoader: PyTorch DataLoader with streaming-aware iteration
"""

from __future__ import annotations

import functools

import einops
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from astrojepa.masks import BlockMaskCollator, make_positions


def normalise(x: torch.Tensor) -> torch.Tensor:
    """Per-patch normalisation: zero-mean, unit-variance."""
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    return (x - mean) / (std + 1e-8)


def process_galaxy_patches(raw_galaxy: torch.Tensor, patch_size: int, img_size: int | None = None) -> torch.Tensor:
    """Convert (C, H, W) image to (N, P) patch tokens.

    If img_size is provided, the image is resized to (img_size, img_size) first.
    """
    if img_size is not None and (raw_galaxy.shape[1] != img_size or raw_galaxy.shape[2] != img_size):
        raw_galaxy = torch.nn.functional.interpolate(
            raw_galaxy.unsqueeze(0),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    galaxy = einops.rearrange(
        raw_galaxy,
        "c (h p1) (w p2) -> (h w) (p1 p2 c)",
        p1=patch_size,
        p2=patch_size,
    )
    return normalise(galaxy)


class GalaxyStreamDataset(IterableDataset):
    """Streaming dataset for the Smith42/galaxies HuggingFace dataset.

    Each item yields a dict with:
        - ``images``: (N, P) float32 patch tokens
        - ``images_positions``: (N,) long position indices

    Args:
        split: 'train', 'validation', or 'test'
        patch_size: size of square patches (default 16)
        in_chans: number of channels (default 3)
        img_size: if set, resize images to this spatial size before patching
        revision: dataset revision (default 'v2.0')
        streaming: if True, stream from HF hub instead of downloading
        max_samples: if set, stop after this many samples (for debugging)
    """

    def __init__(
        self,
        split: str = "train",
        patch_size: int = 16,
        in_chans: int = 3,
        img_size: int | None = None,
        revision: str = "v2.0",
        streaming: bool = True,
        max_samples: int | None = None,
    ):
        self.split = split
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.img_size = img_size
        self.revision = revision
        self.streaming = streaming
        self.max_samples = max_samples

        from datasets import load_dataset
        self._dataset = load_dataset(
            "Smith42/galaxies",
            revision=revision,
            split=split,
            streaming=streaming,
        )

    def __iter__(self):
        """Yield processed galaxy patches."""
        count = 0
        for sample in self._dataset:
            if self.max_samples is not None and count >= self.max_samples:
                break
            try:
                img = np.array(sample["image"]).swapaxes(0, 2)  # (H, W, C) -> (C, H, W)
                img = torch.from_numpy(img).float()
                patches = process_galaxy_patches(img, self.patch_size, img_size=self.img_size)
                yield {
                    "images": patches,
                    "images_positions": make_positions(patches.size(0), device=patches.device),
                }
                count += 1
            except Exception:
                continue


def _collate_galaxy(batch: list[dict], patch_size: int) -> dict[str, torch.Tensor]:
    """Collate function: pad variable-length patch sequences."""
    images = [item["images"] for item in batch]
    positions = [item["images_positions"] for item in batch]

    max_len = max(i.size(0) for i in images)
    B = len(images)

    images_padded = torch.zeros(B, max_len, images[0].size(1), dtype=images[0].dtype)
    positions_padded = torch.zeros(B, max_len, dtype=positions[0].dtype)

    for b, (img, pos) in enumerate(zip(images, positions)):
        images_padded[b, :img.size(0)] = img
        positions_padded[b, :pos.size(0)] = pos

    return {
        "images": images_padded,
        "images_positions": positions_padded,
    }


class StreamingDataLoader:
    """Wrapper around PyTorch DataLoader with HF streaming support.

    Handles DDP sharding and buffered shuffle.
    """

    def __init__(
        self,
        dataset: GalaxyStreamDataset,
        mask_collator: BlockMaskCollator,
        batch_size: int = 16,
        num_workers: int = 4,
        shuffle_buffer_size: int = 1000,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        device: torch.device | None = None,
    ):
        self.dataset = dataset
        self.mask_collator = mask_collator
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # DDP sharding
        if ddp_world_size > 1:
            self.dataset._dataset = self._shard_dataset(
                self.dataset._dataset, ddp_rank, ddp_world_size
            )

        # Buffered shuffle
        if shuffle_buffer_size > 0 and dataset.split == "train":
            self.dataset._dataset = self.dataset._dataset.shuffle(
                seed=1337, buffer_size=shuffle_buffer_size
            )

        self._loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=functools.partial(_collate_galaxy, patch_size=dataset.patch_size),
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True,
        )
        self._iterator = iter(self._loader)

    @staticmethod
    def _shard_dataset(dataset, rank: int, world_size: int):
        from datasets.distributed import split_dataset_by_node
        return split_dataset_by_node(dataset, rank=rank, world_size=world_size)

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        """Fetch and process next batch.

        Returns dict with 'patches', 'masks', 'context_mask', 'target_mask'.
        """
        batch = next(self._iterator)
        B = batch["images"].size(0)

        # Generate masks on CPU, move to device
        masks = self.mask_collator(B, device="cpu")
        masks = {k: v.to(self.device) for k, v in masks.items()}

        # Move patches to device
        patches = batch["images"].to(self.device, non_blocking=True)

        return {
            "patches": patches,
            "masks": masks,
        }

    def reset(self) -> None:
        """Reset the underlying DataLoader iterator."""
        self._iterator = iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)
