#!/usr/bin/env python3
"""Self-supervised PointNet++ pretraining on an RF3DPose processed subset.

Example:
    python my_models/train_feature_extract.py --max_samples 2000 --epochs 10

Evaluate the trained extractor with:
    python my_models/test_feature_extract.py --checkpoint \
        checkpoints/feature_extract/best.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split

from dataset.m4human_dataset import RF3DPoseDataset
from my_models.feature_extract_vol_3 import PointNetPPFeatureExtractor


REQUIRED_CACHE_FILES = (
    "calib.lmdb",
    "indicator.lmdb",
    "params.lmdb",
    "radar_comp.lmdb",
    "radar_pc.lmdb",
    "indeces.pkl.gz",
)


class RadarReconstructionDataset(Dataset):
    """Keep valid radar points and provide a clean reconstruction target."""

    def __init__(self, dataset, num_points, in_channels, seed):
        self.dataset = dataset
        self.num_points = num_points
        self.in_channels = in_channels
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        points = torch.as_tensor(sample["radar_points"]).float()
        if points.ndim != 3 or points.shape[-1] < self.in_channels:
            raise ValueError(
                "Expected radar_points shaped (T, P, C) with at least "
                f"{self.in_channels} channels, got {tuple(points.shape)}."
            )

        points = points[..., :self.in_channels]
        # Padding is added after coordinate normalization and remains XYZ zero.
        valid = points[..., :3].abs().sum(dim=-1).gt(1e-6).any(dim=0)
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            raise ValueError(f"RF3DPose sample {index} has no valid radar points.")

        generator = torch.Generator().manual_seed(self.seed + index)
        order = torch.randperm(valid_indices.numel(), generator=generator)
        selected = valid_indices[order[:self.num_points]]
        if selected.numel() < self.num_points:
            repeats = selected[
                torch.randint(
                    selected.numel(),
                    (self.num_points - selected.numel(),),
                    generator=generator,
                )
            ]
            selected = torch.cat((selected, repeats))

        clean = points.index_select(1, selected)
        return {"radar_points": clean, "dataset_index": index}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(parser, args):
    for name in ("num_points", "temporal_window", "batch_size", "epochs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be greater than zero.")
    if args.in_channels < 3:
        parser.error("--in_channels must be at least 3 for XYZ.")
    if not 3 <= args.reconstruct_channels <= args.in_channels:
        parser.error(
            "--reconstruct_channels must be between 3 and --in_channels."
        )
    if args.lr <= 0:
        parser.error("--lr must be greater than zero.")
    if args.weight_decay < 0:
        parser.error("--weight_decay cannot be negative.")
    if args.transform_weight < 0:
        parser.error("--transform_weight cannot be negative.")
    if args.val_fraction < 0 or args.val_fraction >= 1:
        parser.error("--val_fraction must be in [0, 1).")
    if args.jitter_std < 0:
        parser.error("--jitter_std cannot be negative.")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative.")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max_samples must be greater than zero.")
    if args.save_every <= 0:
        parser.error("--save_every must be greater than zero.")


def make_loaders(args):
    cache_dir = args.cache_dir.expanduser().resolve()
    processed_cache = cache_dir / "rf3dpose_all"
    missing = [
        name for name in REQUIRED_CACHE_FILES
        if not (processed_cache / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{cache_dir} is not a complete RF3DPose processed-cache subset; "
            f"missing from {processed_cache}: {', '.join(missing)}."
        )

    source = RF3DPoseDataset(
        file_pairs=[],
        split=args.split,
        cache_dir=str(cache_dir),
        load_save=True,
        smplx_model_path=str(args.smplx_model_path.expanduser().resolve()),
        use_image=False,
        temporal_window=args.temporal_window,
        scale_id=args.scale_id,
        split_id=args.split_id,
    )
    dataset = RadarReconstructionDataset(
        source,
        num_points=args.num_points,
        in_channels=args.in_channels,
        seed=args.seed,
    )
    if args.max_samples is not None and args.max_samples < len(dataset):
        selected = torch.randperm(
            len(dataset),
            generator=torch.Generator().manual_seed(args.seed),
        )[:args.max_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, selected)

    if len(dataset) < 2 and args.val_fraction > 0:
        raise ValueError(
            "At least two samples are required for a train/validation split."
        )

    if args.val_fraction > 0:
        val_count = max(1, int(len(dataset) * args.val_fraction))
        train_count = len(dataset) - val_count
        train_set, val_set = random_split(
            dataset,
            (train_count, val_count),
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_set, val_set = dataset, None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if val_set is not None
        else None
    )
    return train_loader, val_loader


def run_epoch(
    model,
    decoders,
    loader,
    optimizer,
    device,
    reconstruct_channels,
    transform_weight,
    jitter_std,
    training,
):
    model.train(training)
    decoders.train(training)
    total_loss = 0.0
    total_reconstruction = 0.0
    total_regularization = 0.0
    batch_count = 0

    progress = tqdm(
        loader,
        desc="Training" if training else "Validation",
        leave=False,
    )
    for batch in progress:
        clean_points = batch["radar_points"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        model_points = clean_points
        if training and jitter_std:
            model_points = clean_points.clone()
            model_points[..., :3] += torch.randn_like(
                model_points[..., :3]
            ) * jitter_std

        with torch.set_grad_enabled(training):
            _, aux = model(model_points, return_aux=True)
            target = clean_points[..., :reconstruct_channels]
            branch_losses = {
                "stage1": torch.nn.functional.smooth_l1_loss(
                    decoders["stage1"](aux["stage1_point_features"]),
                    target,
                ),
                "stage2": torch.nn.functional.smooth_l1_loss(
                    decoders["stage2"](aux["stage2_point_features"]),
                    target,
                ),
                "mrg_lower": torch.nn.functional.smooth_l1_loss(
                    decoders["mrg_lower"](aux["mrg_lower_point_features"]),
                    target,
                ),
                "mrg_raw": torch.nn.functional.smooth_l1_loss(
                    decoders["mrg_raw"](aux["mrg_raw_point_features"]),
                    target,
                ),
            }
            reconstruction_loss = (
                0.4 * branch_losses["stage1"]
                + 0.3 * branch_losses["stage2"]
                + 0.15 * branch_losses["mrg_lower"]
                + 0.15 * branch_losses["mrg_raw"]
            )
            regularization_loss = aux["transform_regularization"]
            loss = reconstruction_loss + transform_weight * regularization_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss.")

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(decoders.parameters()),
                    max_norm=10.0,
                )
                optimizer.step()

        total_loss += loss.detach().item()
        total_reconstruction += reconstruction_loss.detach().item()
        total_regularization += regularization_loss.detach().item()
        batch_count += 1
        progress.set_postfix(loss=f"{total_loss / batch_count:.6f}")

    if batch_count == 0:
        raise ValueError("DataLoader produced no batches.")

    return {
        "loss": total_loss / batch_count,
        "reconstruction": total_reconstruction / batch_count,
        "transform_regularization": total_regularization / batch_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain PointNet++ by reconstructing radar XYZ from "
            "interpolated stage-1 features; no segmentation labels are used."
        )
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=ROOT_DIR / "rf3dpose_10gb",
        help="Subset destination produced by create_rf3dpose_10gb_subset.py.",
    )
    parser.add_argument(
        "--smplx_model_path",
        type=Path,
        default=ROOT_DIR / "models",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--scale_id", choices=("p1", "p2", "p3"), default="p3")
    parser.add_argument("--split_id", choices=("s1", "s2", "s3"), default="s2")
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--reconstruct_channels", type=int, default=3)
    parser.add_argument("--num_points", type=int, default=256)
    parser.add_argument("--temporal_window", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--jitter_std", type=float, default=0.01)
    parser.add_argument("--transform_weight", type=float, default=0.001)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=ROOT_DIR / "checkpoints" / "feature_extract",
    )
    args = parser.parse_args()
    validate_args(parser, args)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders(args)

    model = PointNetPPFeatureExtractor(
        in_channels=args.in_channels,
        feature_dim=256,
        first_sample_count=256,
        second_sample_count=64,
        use_input_transform=True,
        use_feature_transform=True,
        use_mrg_second_stage=True,
    ).to(device)
    def make_decoder(in_channels):
        return nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, args.reconstruct_channels),
        )

    decoders = nn.ModuleDict({
        "stage1": make_decoder(128),
        "stage2": make_decoder(256),
        "mrg_lower": make_decoder(128),
        "mrg_raw": make_decoder(128),
    }).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(decoders.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            decoders,
            train_loader,
            optimizer,
            device,
            args.reconstruct_channels,
            args.transform_weight,
            args.jitter_std,
            training=True,
        )
        val_metrics = (
            run_epoch(
                model,
                decoders,
                val_loader,
                optimizer,
                device,
                args.reconstruct_channels,
                args.transform_weight,
                0.0,
                training=False,
            )
            if val_loader is not None
            else None
        )
        val_loss = val_metrics["loss"] if val_metrics else train_metrics["loss"]
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train={train_metrics['loss']:.6f} "
            f"(recon={train_metrics['reconstruction']:.6f}, "
            f"transform={train_metrics['transform_regularization']:.6f})"
            + (
                f" val={val_metrics['loss']:.6f}"
                if val_metrics is not None
                else ""
            )
        )

        if epoch % args.save_every == 0 or val_loss < best_val_loss:
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "decoder_state": decoders.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "model_config": {
                    "in_channels": args.in_channels,
                    "feature_dim": 256,
                    "first_sample_count": 256,
                    "second_sample_count": 64,
                    "use_input_transform": True,
                    "use_feature_transform": True,
                    "use_mrg_second_stage": True,
                },
            }
            path = args.out_dir / f"epoch_{epoch:03d}.pt"
            torch.save(checkpoint, path)
            print(f"Saved checkpoint: {path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "decoder_state": decoders.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "model_config": {
                        "in_channels": args.in_channels,
                        "feature_dim": 256,
                        "first_sample_count": 256,
                        "second_sample_count": 64,
                        "use_input_transform": True,
                        "use_feature_transform": True,
                        "use_mrg_second_stage": True,
                    },
                },
                args.out_dir / "best.pt",
            )

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    print(f"Best checkpoint: {args.out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
