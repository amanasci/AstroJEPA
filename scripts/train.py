"""
Unified pretraining script for AstroJEPA.

Streams the Smith42/galaxies dataset from HuggingFace, applies I-JEPA-style
block masking, and trains the JEPA model with cosine LR decay and EMA.

Usage:
    # Single GPU
    python scripts/train.py --config configs/astrojepa070M.py

    # Multi-GPU (DDP)
    torchrun --standalone --nproc_per_node=4 scripts/train.py --config configs/astrojepa070M.py

    # With overrides
    python scripts/train.py --config configs/astrojepa070M.py --batch_size=64 --max_iters=50000
"""

import math
import os
import sys
import time
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from astrojepa.dataset import GalaxyStreamDataset, StreamingDataLoader
from astrojepa.masks import BlockMaskCollator
from astrojepa.model import JEPA, JEPAConfig


def get_lr(it: int, config: dict) -> float:
    """Cosine LR schedule with linear warmup."""
    if it < config["warmup_iters"]:
        return config["learning_rate"] * it / config["warmup_iters"]
    if it > config["lr_decay_iters"]:
        return config["min_lr"]
    decay_ratio = (it - config["warmup_iters"]) / (config["lr_decay_iters"] - config["warmup_iters"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config["min_lr"] + coeff * (config["learning_rate"] - config["min_lr"])


def ema_momentum_schedule(it: int, config: dict) -> float:
    """Linearly ramp EMA momentum from base toward ema_max over warmup, then hold.

    Never reaches 1.0 (which would freeze the target encoder).
    """
    base = config.get("ema_momentum", 0.996)
    ema_max = config.get("ema_max", 0.999)
    warmup = config.get("ema_warmup_iters", 2000)
    if it >= warmup:
        return ema_max
    return base + (ema_max - base) * it / warmup


def estimate_loss(
    model: JEPA,
    val_loader: StreamingDataLoader,
    eval_iters: int,
    device: str,
    ctx,
) -> float:
    """Average validation loss over ``eval_iters`` batches."""
    model.eval()
    val_loader.reset()
    losses = []
    for _ in range(eval_iters):
        try:
            batch = next(val_loader)
        except StopIteration:
            break
        patches = batch["patches"]
        masks = batch["masks"]
        with ctx:
            out = model(patches, masks=masks)
        losses.append(out["loss"].item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def validate(
    model: JEPA,
    val_loader: StreamingDataLoader,
    out_dir: str,
    iter_num: int,
    device: str,
    ctx,
    log_via_wandb: bool = False,
):
    """Save validation preview images (reconstructions of target patches)."""
    model.eval()
    val_loader.reset()
    try:
        batch = next(val_loader)
    except StopIteration:
        model.train()
        return

    patches = batch["patches"]
    masks = batch["masks"]
    with ctx:
        out = model(patches, masks=masks)

    loss = out["loss"].item()

    # Log
    with open(os.path.join(out_dir, "loss_val.txt"), "a") as f:
        f.write(f"{iter_num},{loss}\n")

    if log_via_wandb:
        import wandb
        wandb.log({"val_loss": loss}, step=iter_num)

    model.train()


def save_checkpoint(
    model: JEPA,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
    best_val_loss: float,
    out_dir: str,
    config: dict,
):
    """Save model checkpoint."""
    raw_model = model.module if isinstance(model, DDP) else model
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    ckpt_path = os.path.join(out_dir, f"ckpt_{iter_num:08d}.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"  saved checkpoint to {ckpt_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AstroJEPA pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to config .py file")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--max_iters", type=int, default=None, help="Override max_iters")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override num_workers")
    parser.add_argument("--eval_interval", type=int, default=None, help="Override eval_interval")
    parser.add_argument("--log_interval", type=int, default=None, help="Override log_interval")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning_rate")
    args = parser.parse_args()

    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config: dict = {}
    with open(config_path) as f:
        exec(compile(f.read(), config_path, "exec"), config)
    config = {k: v for k, v in config.items() if not k.startswith("_")}

    # Apply CLI overrides
    override_map = {
        "max_iters": args.max_iters,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "eval_interval": args.eval_interval,
        "log_interval": args.log_interval,
        "learning_rate": args.learning_rate,
    }
    for key, value in override_map.items():
        if value is not None:
            config[key] = value

    # DDP setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        gradient_accumulation_steps = max(1, config.get("gradient_accumulation_steps", 1) // ddp_world_size)
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_rank = 0
        device = config.get("device", "cuda")
        gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

    if master_process:
        os.makedirs(config["out_dir"], exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.get("dtype", "bfloat16")]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # --- Datasets ---
    grid_size = config["img_size"] // config["patch_size"]
    mask_collator = BlockMaskCollator(
        num_target_blocks=config["num_target_blocks"],
        min_target_block_size=config["min_target_block_size"],
        max_target_block_size=config["max_target_block_size"],
        context_scale=config["context_scale"],
        grid_size=grid_size,
    )

    train_dataset = GalaxyStreamDataset(
        split="train",
        patch_size=config["patch_size"],
        in_chans=config["in_chans"],
        img_size=config["img_size"],
        revision=config.get("dataset_revision", "v2.0"),
        streaming=config.get("stream_hf", True),
        max_samples=None,
    )
    val_dataset = GalaxyStreamDataset(
        split="validation",
        patch_size=config["patch_size"],
        in_chans=config["in_chans"],
        img_size=config["img_size"],
        revision=config.get("dataset_revision", "v2.0"),
        streaming=config.get("stream_hf", True),
        max_samples=10000,
    )

    train_loader = StreamingDataLoader(
        dataset=train_dataset,
        mask_collator=mask_collator,
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 8),
        shuffle_buffer_size=config.get("shuffle_buffer_size", 10000),
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        device=torch.device(device),
    )
    val_loader = StreamingDataLoader(
        dataset=val_dataset,
        mask_collator=mask_collator,
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 8),
        shuffle_buffer_size=0,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        device=torch.device(device),
    )

    # --- Model ---
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
        dropout=config.get("dropout", 0.0),
        use_cls_token=config.get("use_cls_token", True),
        num_target_blocks=config["num_target_blocks"],
        min_target_block_size=config["min_target_block_size"],
        max_target_block_size=config["max_target_block_size"],
        context_scale=config["context_scale"],
        use_vicreg=config.get("use_vicreg", False),
    )
    model = JEPA(jepa_config, master_process=master_process).to(device)

    optimizer = model.configure_optimizers(
        weight_decay=config["weight_decay"],
        learning_rate=config["learning_rate"],
        betas=(config["beta1"], config["beta2"]),
        device_type=device_type,
    )

    scaler = torch.amp.GradScaler(enabled=(config.get("dtype") == "float16"))
    iter_num = 0
    best_val_loss = float("inf")

    # Compile
    if config.get("compile", True):
        if master_process:
            print("compiling model...")
        model = torch.compile(model)

    # DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Resume
    if args.resume:
        ckpt_files = sorted([f for f in os.listdir(config["out_dir"]) if f.endswith(".pt")])
        if ckpt_files:
            latest = os.path.join(config["out_dir"], ckpt_files[-1])
            if master_process:
                print(f"resuming from {latest}")
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            iter_num = ckpt["iter_num"]
            best_val_loss = ckpt["best_val_loss"]

    raw_model = model.module if ddp else model

    if master_process:
        print(f"starting training from iter {iter_num}")
        print(f"config: {config_path}")

    t0 = time.time()
    dts = []
    stream_exhausted = False

    # --- Training loop ---
    while True:
        lr = get_lr(iter_num, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Eval
        if iter_num % config["eval_interval"] == 0 and master_process and iter_num > 0:
            val_loss = estimate_loss(model, val_loader, config["eval_iters"], device, ctx)
            print(f"iter {iter_num}: val loss {val_loss:.6f}")
            with open(os.path.join(config["out_dir"], "loss.txt"), "a") as f:
                if os.path.getsize(os.path.join(config["out_dir"], "loss.txt")) == 0:
                    f.write("iter_num,train_loss,val_loss,lr\n")
            with open(os.path.join(config["out_dir"], "loss.txt"), "a") as f:
                f.write(f"{iter_num},,{val_loss},{lr}\n")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, iter_num, best_val_loss, config["out_dir"], config)

        # Checkpoint schedule
        if config.get("num_checkpoints", 0) > 1:
            geo = np.geomspace(1, config["max_iters"], config["num_checkpoints"] - 1)
            ckpt_iters = {int(round(x)) for x in geo} | {0, config["max_iters"]}
            if master_process and iter_num in ckpt_iters and iter_num > 0:
                save_checkpoint(model, optimizer, iter_num, best_val_loss, config["out_dir"], config)

        # Forward / backward
        stream_exhausted = False
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
            try:
                batch = next(train_loader)
            except StopIteration:
                stream_exhausted = True
                break

            patches = batch["patches"]
            masks = batch["masks"]

            with ctx:
                out = model(patches, masks=masks)
                loss = out["loss"] / gradient_accumulation_steps

            scaler.scale(loss).backward()

        if stream_exhausted:
            if any(p.grad is not None for p in model.parameters()):
                if config.get("grad_clip", 1.0) != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema_m = ema_momentum_schedule(iter_num, config)
                model.update_target_encoder(m=ema_m)
            if master_process:
                print(f"stream exhausted at iter {iter_num}; stopping")
                save_checkpoint(model, optimizer, iter_num, best_val_loss, config["out_dir"], config)
            break

        if config.get("grad_clip", 1.0) != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Update target encoder EMA
        ema_m = ema_momentum_schedule(iter_num, config)
        model.update_target_encoder(m=ema_m)

        # Logging
        t1 = time.time()
        dt = t1 - t0
        dts.append(dt)
        t0 = t1

        if iter_num % config.get("log_interval", 100) == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.6f}, lr {lr:.2e}, ema_m {ema_m:.6f}, time {np.mean(dts)*1000:.2f}ms")
            dts = []
            with open(os.path.join(config["out_dir"], "loss.txt"), "a") as f:
                f.write(f"{iter_num},{lossf},,{lr}\n")

        iter_num += 1

        if iter_num >= config["max_iters"]:
            if master_process:
                save_checkpoint(model, optimizer, iter_num, best_val_loss, config["out_dir"], config)
            break

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
