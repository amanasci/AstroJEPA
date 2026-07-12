"""
Linear probe evaluation for AstroJEPA.

Extracts frozen embeddings from a pre-trained JEPA model and trains a
linear classifier/regressor on top for galaxy property prediction.

Usage:
    python scripts/linear_probe.py \
        --checkpoint logs/astrojepa070M/ckpt_05000.pt \
        --config configs/astrojepa070M.py \
        --target_property galaxy_size \
        --task regression
"""

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from astrojepa.dataset import GalaxyStreamDataset, _collate_galaxy, process_galaxy_patches
from astrojepa.model import JEPA, JEPAConfig


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    jepa_config = JEPAConfig(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        in_chans=config["in_chans"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        predictor_n_embd=config["predictor_n_embd"],
        predictor_n_head=config["predictor_n_head"],
        predictor_n_layer=config["predictor_n_layer"],
        predictor_num_queries=config["predictor_num_queries"],
        bias=config.get("bias", False),
        dropout=0.0,
        use_cls_token=config.get("use_cls_token", True),
    )
    model = JEPA(jepa_config, master_process=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, config


def extract_embeddings_and_labels(
    model: JEPA,
    split: str,
    config: dict,
    batch_size: int = 32,
    device: torch.device | None = None,
    max_samples: int = 50000,
    target_property: str = "galaxy_size",
) -> tuple[np.ndarray, np.ndarray]:
    """Stream data, extract embeddings, and collect labels in one pass."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from datasets import load_dataset
    hf_ds = load_dataset(
        "Smith42/galaxies",
        revision=config.get("dataset_revision", "v2.0"),
        split=split,
        streaming=True,
    )

    all_embeds = []
    all_labels = []
    count = 0
    batch_patches = []
    batch_labels = []

    with torch.no_grad():
        for sample in hf_ds:
            if count >= max_samples:
                break
            try:
                img = np.array(sample["image"]).swapaxes(0, 2)
                img = torch.from_numpy(img).float()
                patches = process_galaxy_patches(
                    img, config["patch_size"], img_size=config.get("img_size")
                )
                batch_patches.append(patches)
                val = sample.get(target_property)
                if val is None:
                    val = 0
                batch_labels.append(val)
                count += 1

                if len(batch_patches) == batch_size:
                    batch_tensor = torch.stack(batch_patches).to(device)
                    embeds = model.get_embeddings(batch_tensor, reduction="mean")
                    all_embeds.append(embeds.cpu().numpy())
                    all_labels.extend(batch_labels)
                    batch_patches = []
                    batch_labels = []
            except Exception:
                continue

    if batch_patches:
        batch_tensor = torch.stack(batch_patches).to(device)
        embeds = model.get_embeddings(batch_tensor, reduction="mean")
        all_embeds.append(embeds.cpu().numpy())
        all_labels.extend(batch_labels)

    if not all_embeds:
        raise RuntimeError("No samples extracted. Check dataset and streaming config.")

    return np.concatenate(all_embeds, axis=0), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--target_property", type=str, default="galaxy_size")
    parser.add_argument("--task", type=str, default="regression", choices=["regression", "classification"])
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_checkpoint(args.checkpoint, device)

    print(f"Extracting train embeddings + labels ({args.target_property})...")
    X_train, y_train = extract_embeddings_and_labels(
        model, "train", config, batch_size=args.batch_size,
        device=device, max_samples=args.max_samples, target_property=args.target_property,
    )
    print(f"Extracting val embeddings + labels...")
    X_val, y_val = extract_embeddings_and_labels(
        model, "validation", config, batch_size=args.batch_size,
        device=device, max_samples=10000, target_property=args.target_property,
    )

    print(f"Train: {X_train.shape}, labels: {y_train.shape}")
    print(f"Val:   {X_val.shape}, labels: {y_val.shape}")

    if args.task == "regression":
        clf = Ridge(alpha=1.0)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        print(f"Linear Probe Regression ({args.target_property}):")
        print(f"  MSE: {mse:.4f}")
        print(f"  R²:  {r2:.4f}")
    else:
        train_bins = np.digitize(y_train, np.quantile(y_train, np.linspace(0, 1, args.num_classes + 1))) - 1
        val_bins = np.digitize(y_val, np.quantile(y_val, np.linspace(0, 1, args.num_classes + 1))) - 1
        train_bins = np.clip(train_bins, 0, args.num_classes - 1)
        val_bins = np.clip(val_bins, 0, args.num_classes - 1)

        clf = LogisticRegression(max_iter=1000, n_jobs=-1)
        clf.fit(X_train, train_bins)
        preds = clf.predict(X_val)
        acc = accuracy_score(val_bins, preds)
        print(f"Linear Probe Classification ({args.target_property}, {args.num_classes} classes):")
        print(f"  Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
