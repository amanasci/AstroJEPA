"""
Masking utilities for AstroJEPA.

Implements I-JEPA-style block masking:
- Context blocks: a small number of large, spatially distributed blocks
- Target blocks: multiple blocks of varying scales

Reference: Assran et al., "Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture" (I-JEPA, 2023).
"""

import random

import torch


def make_positions(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Return a 1-D position tensor ``[0, 1, ..., seq_len - 1]``."""
    return torch.arange(seq_len, dtype=torch.long, device=device)


class BlockMaskCollator:
    """Generate context/target block masks for a batch of images.

    For each sample we:
    1. Split the 2-D patch grid into *num_blocks* target blocks whose sizes
       are sampled uniformly in ``[min_target_block_size, max_target_block_size)``
       (measured as number of patches, not pixels).
    2. The remaining patches form the *context*.
    3. For every target block we also produce a *context around target* view
       (a larger block that contains the target block plus some surrounding
       context).  This is used to build the predictor queries.

    Args:
        num_target_blocks: number of target blocks per image.
        min_target_block_size: minimum target block size (in patches).
        max_target_block_size: maximum target block size (in patches, exclusive).
        context_scale: multiplier applied to target block size to get the
            surrounding context block size.
        grid_size: number of patches per side (e.g. 32 for 512px / 16px patches).
    """

    def __init__(
        self,
        num_target_blocks: int = 4,
        min_target_block_size: int = 16,
        max_target_block_size: int = 64,
        context_scale: float = 2.0,
        grid_size: int = 32,
    ):
        self.num_target_blocks = num_target_blocks
        self.min_target_block_size = min_target_block_size
        self.max_target_block_size = max_target_block_size
        self.context_scale = context_scale
        self.grid_size = grid_size
        self.total_patches = grid_size * grid_size

    def _sample_block(self, occupied: torch.Tensor) -> tuple[int, int, int, int]:
        """Sample a non-overlapping rectangular block.

        Returns ``(y0, x0, y1, x1)`` in patch coordinates.
        """
        gs = self.grid_size
        min_area = self.min_target_block_size
        max_area = self.max_target_block_size

        # Precompute all valid (bh, bw) pairs for this grid size
        valid_dims = []
        for bh in range(1, gs + 1):
            for bw in range(1, gs + 1):
                area = bh * bw
                if min_area <= area < max_area:
                    valid_dims.append((bh, bw))

        if not valid_dims:
            # Fallback: use the whole grid as one block
            return 0, 0, gs, gs

        for _ in range(200):
            bh, bw = random.choice(valid_dims)
            y0 = random.randint(0, gs - bh)
            x0 = random.randint(0, gs - bw)
            y1, x1 = y0 + bh, x0 + bw
            if not occupied[y0:y1, x0:x1].any():
                occupied[y0:y1, x0:x1] = True
                return y0, x0, y1, x1

        # Fallback: find any free spot
        for bh, bw in valid_dims:
            for y0 in range(gs - bh + 1):
                for x0 in range(gs - bw + 1):
                    y1, x1 = y0 + bh, x0 + bw
                    if not occupied[y0:y1, x0:x1].any():
                        occupied[y0:y1, x0:x1] = True
                        return y0, x0, y1, x1

        raise RuntimeError("Could not place a non-overlapping block; grid may be too small")

    def _block_to_mask(self, y0: int, x0: int, y1: int, x1: int) -> torch.Tensor:
        """Convert a rectangular block to a 1-D boolean mask of length ``grid_size**2``."""
        mask = torch.zeros(self.total_patches, dtype=torch.bool)
        idx = torch.arange(self.total_patches).reshape(self.grid_size, self.grid_size)
        mask[idx[y0:y1, x0:x1].flatten()] = True
        return mask

    def __call__(self, batch_size: int, device: torch.device | None = None) -> dict[str, torch.Tensor]:
        """Generate masks for a batch of ``batch_size`` images.

        Returns a dict with keys:
            - ``context``:      (B, N) bool – True where patch is context
            - ``target``:       (B, N) bool – True where patch is target
            - ``target_ctx``:   (B, N) bool – True where patch is context *around* target
        """
        context_masks = []
        target_masks = []
        target_ctx_masks = []

        for _ in range(batch_size):
            occupied = torch.zeros(self.grid_size, self.grid_size, dtype=torch.bool)
            target_mask = torch.zeros(self.total_patches, dtype=torch.bool)
            target_ctx_mask = torch.zeros(self.total_patches, dtype=torch.bool)

            for _ in range(self.num_target_blocks):
                y0, x0, y1, x1 = self._sample_block(occupied)
                block_mask = self._block_to_mask(y0, x0, y1, x1)
                target_mask |= block_mask

            # Context = everything not in target
            context_mask = ~target_mask

            # For each target block, build a surrounding context block
            target_indices = target_mask.nonzero(as_tuple=False).squeeze(1)
            if target_indices.numel() > 0:
                target_grid_y = target_indices // self.grid_size
                target_grid_x = target_indices % self.grid_size
                t_y0 = target_grid_y.min().item()
                t_x0 = target_grid_x.min().item()
                t_y1 = target_grid_y.max().item() + 1
                t_x1 = target_grid_x.max().item() + 1

                th = t_y1 - t_y0
                tw = t_x1 - t_x0
                ch = max(self.grid_size, int(th * self.context_scale))
                cw = max(self.grid_size, int(tw * self.context_scale))

                cy0 = max(0, t_y0 - (ch - th) // 2)
                cx0 = max(0, t_x0 - (cw - tw) // 2)
                cy1 = min(self.grid_size, cy0 + ch)
                cx1 = min(self.grid_size, cx0 + cw)
                cy0 = max(0, cy1 - ch)
                cx0 = max(0, cx1 - cw)

                ctx_idx = torch.arange(self.total_patches).reshape(self.grid_size, self.grid_size)
                target_ctx_mask[ctx_idx[cy0:cy1, cx0:cx1].flatten()] = True
                # Context-around-target should not include target itself
                target_ctx_mask &= ~target_mask

            context_masks.append(context_mask)
            target_masks.append(target_mask)
            target_ctx_masks.append(target_ctx_mask)

        return {
            "context": torch.stack(context_masks).to(device),
            "target": torch.stack(target_masks).to(device),
            "target_ctx": torch.stack(target_ctx_masks).to(device),
        }
