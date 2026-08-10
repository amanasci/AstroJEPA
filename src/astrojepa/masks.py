"""
Masking utilities for AstroJEPA.

Implements I-JEPA-style block masking with aspect-ratio limits and
per-rank RNG for DDP determinism.
"""

from __future__ import annotations

import random

import torch


def make_positions(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.arange(seq_len, dtype=torch.long, device=device)


class BlockMaskCollator:
    """Generate context/target block masks for a batch of images.

    For each sample we sample ``num_target_blocks`` non-overlapping
    rectangles whose area is in ``[min_target_block_size,
    max_target_block_size)`` and whose aspect ratio is clipped to
    ``[min_aspect, max_aspect]``. The remaining patches form the context.
    """

    def __init__(
        self,
        num_target_blocks: int = 4,
        min_target_block_size: int = 16,
        max_target_block_size: int = 64,
        context_scale: float = 2.0,
        grid_size: int = 32,
        min_aspect: float = 0.75,
        max_aspect: float = 1.5,
        seed: int | None = None,
    ):
        self.num_target_blocks = num_target_blocks
        self.min_target_block_size = min_target_block_size
        self.max_target_block_size = max_target_block_size
        self.context_scale = context_scale
        self.grid_size = grid_size
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.seed = seed
        self.total_patches = grid_size * grid_size
        self._rng = random.Random(seed) if seed is not None else random
        self._valid_dims = self._build_valid_dims()

    def _build_valid_dims(self) -> list[tuple[int, int]]:
        gs = self.grid_size
        dims: list[tuple[int, int]] = []
        for bh in range(1, gs + 1):
            for bw in range(1, gs + 1):
                area = bh * bw
                if not (self.min_target_block_size <= area < self.max_target_block_size):
                    continue
                aspect = bw / bh
                if self.min_aspect <= aspect <= self.max_aspect:
                    dims.append((bh, bw))
        if not dims:
            for bh in range(1, gs + 1):
                for bw in range(1, gs + 1):
                    area = bh * bw
                    if self.min_target_block_size <= area < self.max_target_block_size:
                        dims.append((bh, bw))
        return dims

    def _sample_block(self, occupied: torch.Tensor, rng: random.Random) -> tuple[int, int, int, int]:
        gs = self.grid_size
        valid_dims = self._valid_dims
        if not valid_dims:
            return 0, 0, gs, gs

        for _ in range(200):
            bh, bw = rng.choice(valid_dims)
            y0 = rng.randint(0, gs - bh)
            x0 = rng.randint(0, gs - bw)
            y1, x1 = y0 + bh, x0 + bw
            if not occupied[y0:y1, x0:x1].any():
                occupied[y0:y1, x0:x1] = True
                return y0, x0, y1, x1

        for bh, bw in valid_dims:
            for y0 in range(gs - bh + 1):
                for x0 in range(gs - bw + 1):
                    y1, x1 = y0 + bh, x0 + bw
                    if not occupied[y0:y1, x0:x1].any():
                        occupied[y0:y1, x0:x1] = True
                        return y0, x0, y1, x1

        raise RuntimeError("Could not place a non-overlapping block; grid may be too small")

    def _block_to_mask(self, y0: int, x0: int, y1: int, x1: int) -> torch.Tensor:
        mask = torch.zeros(self.total_patches, dtype=torch.bool)
        idx = torch.arange(self.total_patches).reshape(self.grid_size, self.grid_size)
        mask[idx[y0:y1, x0:x1].flatten()] = True
        return mask

    def __call__(
        self,
        batch_size: int,
        device: torch.device | None = None,
        generator: torch.Generator | random.Random | None = None,
        seed: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate masks for ``batch_size`` images.

        Args:
            generator: optional ``random.Random`` or ``torch.Generator``
                for per-rank determinism. If ``seed`` is given it takes
                precedence.

        Returns:
            dict with ``context`` (B,N), ``target`` (B,N),
            ``target_blocks`` (B,K,N).
        """
        if seed is not None:
            rng: random.Random = random.Random(seed)
        elif isinstance(generator, random.Random):
            rng = generator
        elif isinstance(generator, torch.Generator):
            rng = random.Random(int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item()))
        else:
            rng = self._rng

        context_masks: list[torch.Tensor] = []
        target_masks: list[torch.Tensor] = []
        target_block_masks: list[torch.Tensor] = []

        for _ in range(batch_size):
            occupied = torch.zeros(self.grid_size, self.grid_size, dtype=torch.bool)
            target_mask = torch.zeros(self.total_patches, dtype=torch.bool)
            block_masks: list[torch.Tensor] = []

            for _ in range(self.num_target_blocks):
                y0, x0, y1, x1 = self._sample_block(occupied, rng)
                block_mask = self._block_to_mask(y0, x0, y1, x1)
                block_masks.append(block_mask)
                target_mask |= block_mask

            context_mask = ~target_mask

            context_masks.append(context_mask)
            target_masks.append(target_mask)
            target_block_masks.append(torch.stack(block_masks))

        return {
            "context": torch.stack(context_masks).to(device),
            "target": torch.stack(target_masks).to(device),
            "target_blocks": torch.stack(target_block_masks).to(device),
        }
