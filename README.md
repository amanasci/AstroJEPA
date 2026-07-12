# AstroJEPA

Joint-Embedding Predictive Architecture for astronomy galaxy images.

AstroJEPA adapts Meta AI's [I-JEPA](https://arxiv.org/abs/2301.08243) / [V-JEPA](https://arxiv.org/abs/2404.08471) to the [Smith42/galaxies](https://huggingface.co/datasets/Smith42/galaxies) dataset, a collection of ~8.5M galaxy cutouts from the DESI Legacy Imaging Surveys.

Instead of autoregressively predicting the next patch (like AstroPT) or reconstructing pixels (like MAE), AstroJEPA learns by predicting the *latent representations* of masked image regions from visible context — a non-generative, semantic self-supervised objective.

## How it works

```
Galaxy image (512×512)
       │
       ▼
  Split into 32×32 = 1024 patches (16×16 px each)
       │
       ▼
  Block masking (I-JEPA style)
  • Context: ~50% of patches in large blocks
  • Target:  several blocks of varying scale
       │
       ▼
  Context Encoder (ViT) ──► Context embeddings
                               │
                               ▼
                         Predictor (small ViT)
                               │
                               ▼
                  Predicted target embeddings
                               │
                               ▼
  Target Encoder (ViT, frozen EMA) ──► Target embeddings
                               │
                               ▼
                    Loss = ||predicted − target||²
```

### Key components

| Component | Role |
|-----------|------|
| **Context Encoder** | ViT that processes only the visible (non-masked) patches |
| **Target Encoder** | EMA copy of Context Encoder; processes all patches, gradients stopped |
| **Predictor** | Small ViT with learned query tokens; predicts target embeddings from context |

## Installation

```bash
# Clone the repo
git clone https://github.com/<yourname>/AstroJEPA.git
cd AstroJEPA

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick start

### Pretrain on Smith42/galaxies

```bash
# Single GPU
python scripts/train.py --config configs/astrojepa070M.py

# 4 GPUs with DDP
torchrun --standalone --nproc_per_node=4 scripts/train.py --config configs/astrojepa070M.py
```

Overrides:
```bash
python scripts/train.py --config configs/astrojepa070M.py --batch_size=64 --max_iters=50000
```

### Linear probe

```bash
python scripts/linear_probe.py \
    --checkpoint logs/astrojepa070M/ckpt_05000.pt \
    --config configs/astrojepa070M.py \
    --target_property galaxy_size \
    --task regression
```

## Config files

| Config | Params | Use case |
|--------|--------|----------|
| `configs/astrojepa070M.py` | ~70M | Small-scale debugging / single-GPU |
| `configs/astrojepa300M.py` | ~300M | Full-scale training |

Edit the config file or pass CLI overrides to change batch size, learning rate, etc.

## Dataset

We stream the [Smith42/galaxies](https://huggingface.co/datasets/Smith42/galaxies) dataset directly from HuggingFace. The dataset contains ~8.5M 512×512 JPG galaxy cutouts from DESI DR8.

- **Train:** ~8.47M images
- **Validation:** ~86.5k images
- **Test:** ~86.5k images

No local download required — the training script streams data with a configurable shuffle buffer.

## Model sizes

| Model | ContextEncoder | Predictor | Total |
|-------|---------------|-----------|-------|
| 070M | ViT-B/16 (12L, 768C) | 4L, 384C | ~70M |
| 300M | ViT-L/16 (24L, 1024C) | 6L, 512C | ~300M |

## Project structure

```
AstroJEPA/
├── src/astrojepa/
│   ├── __init__.py
│   ├── model.py        # JEPA, ViTEncoder, Predictor
│   ├── dataset.py      # GalaxyStreamDataset, StreamingDataLoader
│   └── masks.py        # BlockMaskCollator (I-JEPA masking)
├── configs/
│   ├── astrojepa070M.py
│   └── astrojepa300M.py
├── scripts/
│   ├── train.py
│   └── linear_probe.py
├── logs/
└── pyproject.toml
```

## License

MIT
