<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/self--supervised-JEPA-9cf.svg" alt="JEPA">
</p>

<h1 align="center">AstroJEPA</h1>

<p align="center">
  <b>Joint-Embedding Predictive Architecture for galaxy imaging.</b><br>
  Self-supervised representation learning on millions of galaxy cutouts from the DESI Legacy Surveys.
</p>

---

## Overview

**AstroJEPA** adapts Meta AI's [I-JEPA](https://arxiv.org/abs/2301.08243) and
[V-JEPA](https://arxiv.org/abs/2404.08471) to astronomical imagery, pretraining on
the [Smith42/galaxies](https://huggingface.co/datasets/Smith42/galaxies) dataset — a
collection of ~8.5M galaxy cutouts from the DESI Legacy Imaging Surveys.

Unlike autoregressive methods (e.g. *AstroPT*) that predict the next patch, or
reconstruction methods (e.g. *MAE*) that recover pixels, AstroJEPA learns by
**predicting the latent representations** of masked image regions from visible
context. This yields a non-generative, semantic self-supervised objective that is
robust to pixel-level noise and produces transferable features for downstream
astronomy tasks.

### Key features

- **Semantic objectives** — predicts embeddings, not pixels, avoiding shortcut
  low-level reconstruction.
- **Streaming dataset** — trains directly from HuggingFace with a configurable
  shuffle buffer; no local download of ~8.5M images required.
- **Scalable configs** — five ready-to-run presets from ~1M to ~300M parameters.
- **Multi-GPU ready** — single-GPU and `torchrun` DDP entry points.
- **Linear-probe evaluation** — assess learned representations on galaxy properties.

## How it works

```
              Galaxy image (512×512)
                       │
                       ▼
        Split into 32×32 = 1024 patches (16×16 px each)
                       │
                       ▼
            Block masking (I-JEPA style)
   • Target:  K non-overlapping blocks of varying scale
   • Context: all remaining patches
                       │
        ┌──────────────┴───────────────┐
        ▼                              ▼
 Context Encoder (ViT)          Target Encoder (ViT)
  · visible patches only         · ALL patches (frozen EMA)
        │                              │
        ▼                              ▼
  Context embeddings            Target embeddings
        │
        ▼
   Predictor (small ViT, cross-attention)
    · one query token per target block
        │
        ▼
 Predicted target embeddings
        │
        ▼
  Loss = mean_k ‖ predicted_k − target_k ‖²
```

### Components

| Component | Role |
|-----------|------|
| **Context Encoder** | ViT that processes only the visible (non-masked) patches. |
| **Target Encoder** | EMA copy of the Context Encoder; processes all patches with gradients stopped. |
| **Predictor** | Small ViT with learned query tokens; predicts each target block's embedding from context via cross-attention. |

## Installation

```bash
# Clone the repository
git clone https://github.com/amanasci/AstroJEPA.git
cd AstroJEPA

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

> Requires Python ≥ 3.10 and a CUDA-capable GPU for training.

## Quick start

### 1. Smoke test (verify the pipeline)

```bash
python scripts/train.py --config configs/smoke_test.py
```

### 2. Pretrain on Smith42/galaxies

```bash
# Single GPU
python scripts/train.py --config configs/astrojepa070M.py

# 4 GPUs with DDP
torchrun --standalone --nproc_per_node=4 scripts/train.py --config configs/astrojepa070M.py
```

Override any config field from the CLI:

```bash
python scripts/train.py --config configs/astrojepa070M.py --batch_size=64 --max_iters=50000
```

### 3. Linear probe

```bash
python scripts/linear_probe.py \
    --checkpoint logs/astrojepa070M/ckpt_05000.pt \
    --config configs/astrojepa070M.py \
    --target_property galaxy_size \
    --task regression
```

## Configuration

Edit a config file or pass CLI overrides to change batch size, learning rate,
masking, and more.

| Config | Params | Use case |
|--------|--------|----------|
| `configs/astrojepa001M.py` | ~1M | Fast debugging / smoke tests (single GPU) |
| `configs/astrojepa005M.py` | ~5M | Moderate-scale pretraining |
| `configs/astrojepa021M.py` | ~21M | Deeper pretraining |
| `configs/astrojepa070M.py` | ~70M | Large-scale pretraining (single / 4 GPUs) |
| `configs/astrojepa300M.py` | ~300M | Full-scale training (multi-GPU) |
| `configs/smoke_test.py` | ~1M | Tiny config for CI / pipeline checks |

## Dataset

We stream the [Smith42/galaxies](https://huggingface.co/datasets/Smith42/galaxies)
dataset directly from HuggingFace. The dataset contains ~8.5M 512×512 JPG galaxy
cutouts from DESI DR8.

| Split | Images |
|-------|--------|
| **Train** | ~8.47M |
| **Validation** | ~86.5k |
| **Test** | ~86.5k |

No local download is required — the training script streams data with a
configurable shuffle buffer.

## Model sizes

All models share the same architecture (a custom ViT encoder plus a
cross-attention predictor); only width and depth differ. The Target Encoder is an
EMA copy of the Context Encoder and is not trained directly, so the listed
*Total* includes it, while the trainable parameter count equals that of the
Context Encoder + Predictor.

| Model | Context / Target Encoder | Predictor | Trainable | Total (incl. EMA) |
|-------|--------------------------|-----------|-----------|-------------------|
| 001M | 3L, 128C, 2H  | 2L, 64C, 2H   | ~1.0M  | ~1.9M  |
| 005M | 5L, 256C, 4H  | 2L, 128C, 4H  | ~5.1M  | ~9.5M  |
| 021M | 5L, 512C, 8H  | 3L, 256C, 4H  | ~20M   | ~37M   |
| 070M | 8L, 768C, 12H | 4L, 384C, 6H  | ~68M   | ~126M  |
| 300M | 13L, 1280C, 16H | 5L, 640C, 8H | ~293M | ~551M |

*L = layers, C = embedding dimension, H = attention heads.*

## Project structure

```
AstroJEPA/
├── src/astrojepa/
│   ├── __init__.py
│   ├── model.py        # JEPA, ViTEncoder, Predictor
│   ├── dataset.py      # GalaxyStreamDataset, StreamingDataLoader
│   └── masks.py        # BlockMaskCollator (I-JEPA masking)
├── configs/
│   ├── astrojepa001M.py
│   ├── astrojepa005M.py
│   ├── astrojepa021M.py
│   ├── astrojepa070M.py
│   ├── astrojepa300M.py
│   └── smoke_test.py
├── scripts/
│   ├── train.py
│   └── linear_probe.py
├── logs/
└── pyproject.toml
```
# Contributions
You are welcome to contribute! Please submit issues or pull requests for bug fixes, new features, or improvements.
