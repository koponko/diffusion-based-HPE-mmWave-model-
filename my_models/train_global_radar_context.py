"""Phase-1 training loop for Global Radar Context only.

This follows the paper's phase-one idea:
    PointNet++ encoder + GRC + coarse pose head
are trained together with an L2 pose regression loss.

No local context, limb loss, TMC, or diffusion is used here.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset.m4human_dataset import RF3DPoseDataset
from my_models.feature_extract_vol_3 import PointNetPPFeatureExtractor
from my_models.global_mm_diff import GlobalRadarConditionEncoder



def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unpack_batch(batch, num_joints: int = 22):
    """Support the RF3DPose dict returned by your dataset."""
    if not isinstance(batch, dict):
        raise TypeError("Expected RF3DPoseDataset batches to be dictionaries.")
    if "radar_points" not in batch:
        raise KeyError("Batch needs 'radar_points'.")
    if "joints_root" not in batch:
        raise KeyError(
            "Batch needs 'joints_root'. Your RF3DPoseDataset currently returns "
            f"this field; make sure it is collated as (B,{num_joints},3)."
        )

    radar = torch.as_tensor(batch["radar_points"]).float()
    target = torch.as_tensor(batch["joints_root"]).float()

    # Some datasets store an extra temporal dimension for joints. For the
    # current global-context implementation we supervise the current frame.
    if target.ndim == 4:
        target = target[:, -1]
    if target.ndim != 3 or target.shape[-2:] != (num_joints, 3):
        raise ValueError(f"Expected joints_root as (B,{num_joints},3), got {tuple(target.shape)}")

    # Make the pose root-relative, matching the reference implementation's
    # pelvis/root normalization used before its phase-one loss.
    target = target - target[:, :1]
    return radar, target


def run_epoch(context_encoder, loader, optimizer, device, training=True, num_joints=22):
    context_encoder.train(training)
    total = 0.0
    count = 0

    for batch in loader:
        radar, target = unpack_batch(batch, num_joints=num_joints)
        radar = radar.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            out = context_encoder(radar)
            coarse_pose = out["coarse_pose"]
            loss = F.mse_loss(coarse_pose, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), 1.0)
                optimizer.step()

        total += loss.item() * radar.shape[0]
        count += radar.shape[0]

    return total / max(count, 1)


def build_dataset(args):
    dataset = RF3DPoseDataset(
        file_pairs=[],
        split=args.split,
        cache_dir=str(Path(args.cache_dir).expanduser()),
        load_save=True,
        smplx_model_path=str(Path(args.smplx_model_path).expanduser()),
        use_image=False,
        temporal_window=args.temporal_window,
        scale_id=args.scale_id,
        split_id=args.split_id,
    )
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--smplx_model_path", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temporal_window", type=int, default=4)
    parser.add_argument("--num_joints", type=int, default=22)
    parser.add_argument("--scale_id", choices=["p1", "p2", "p3"], default="p3")
    parser.add_argument("--split_id", choices=["s1", "s2", "s3"], default="s2")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="checkpoints/global_phase1_best.pt")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = build_dataset(args)
    if args.max_samples is not None and args.max_samples < len(dataset):
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(dataset), generator=g)[:args.max_samples]
        dataset = torch.utils.data.Subset(dataset, idx.tolist())

    val_n = max(1, int(len(dataset) * args.val_fraction))
    train_n = len(dataset) - val_n
    train_set, val_set = random_split(
        dataset,
        [train_n, val_n],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Keep the PointNet++ design you already selected for the project.
    pointnet = PointNetPPFeatureExtractor(
        in_channels=3,
        feature_dim=256,
        first_sample_count=128,
        second_sample_count=32,
        use_input_transform=True,
        use_feature_transform=True,
        robust_grouping=True,
        radius_scale=2.0,
        first_sampling_mode="hybrid",
        second_sampling_mode="fps",
        use_mrg_second_stage=True,
    )

    context_encoder = GlobalRadarConditionEncoder(
        pointnet_encoder=pointnet,
        pointnet_feature_dim=256,
        condition_dim=64,
        temporal_window=args.temporal_window,
        grc_dim=256,
        grc_layers=4,
        grc_heads=8,
        dropout=0.1,
        num_joints=args.num_joints,
    ).to(device)

    optimizer = torch.optim.AdamW(
        context_encoder.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best = float("inf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            context_encoder, train_loader, optimizer, device, True, num_joints=args.num_joints
        )
        val_loss = run_epoch(
            context_encoder, val_loader, optimizer, device, False, num_joints=args.num_joints
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train={train_loss:.6f} val={val_loss:.6f}"
        )

        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": context_encoder.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val": best,
                    "config": {
                        "pointnet_feature_dim": 256,
                        "condition_dim": 64,
                        "temporal_window": args.temporal_window,
                        "grc_dim": 256,
                        "grc_layers": 4,
                        "grc_heads": 8,
                        "num_joints": args.num_joints,
                    },
                },
                args.out,
            )
            print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
