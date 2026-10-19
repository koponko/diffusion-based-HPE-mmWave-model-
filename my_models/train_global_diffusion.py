"""Phase-2 DDPM training loop for GRC-only mmWave pose diffusion.

DDIM sampling is intentionally NOT implemented here; implement it yourself
using the schedule fields and the formula note in global_mm_diff.py.

Phase 2 assumption:
    phase-1 global-context parameters are frozen.

Inputs per batch:
    radar_points: (B,T,P,C)
    joints_root:  (B,num_joints,3) current GT pose

Training objective:
    E || eps - eps_theta(H_t, t, C_global) ||^2

This file intentionally omits LRC/TMC/SLC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.m4human_dataset import RF3DPoseDataset
from my_models.feature_extract import PointNetPPFeatureExtractor
from my_models.global_mm_diff import (
    GlobalRadarConditionEncoder,
    GlobalConditionedPoseDiffusion,
    ConstantBetaSchedule,
)


def build_dataset(args):
    return RF3DPoseDataset(
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


def unpack(batch, device, num_joints: int = 22):
    radar = torch.as_tensor(batch["radar_points"]).float().to(device, non_blocking=True)
    target = torch.as_tensor(batch["joints_root"]).float().to(device, non_blocking=True)
    if target.ndim == 4:
        target = target[:, -1]
    if target.ndim != 3 or target.shape[-2:] != (num_joints, 3):
        raise ValueError(f"Expected joints_root as (B,{num_joints},3), got {tuple(target.shape)}")
    target = target - target[:, :1]
    return radar, target


def train_epoch(context_encoder, denoiser, loader, optimizer, schedule, device, num_joints: int = 22):
    denoiser.train()
    total = 0.0
    count = 0

    for batch in loader:
        radar, x0 = unpack(batch, device, num_joints=num_joints)
        b = x0.shape[0]
        t = torch.randint(0, schedule.num_steps, (b,), device=device)

        with torch.no_grad():
            context = context_encoder(radar)
            c_global = context["c_global"]

        x_t, noise = schedule.q_sample(x0, t)
        pred = denoiser(x_t, t, c_global)
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        optimizer.step()

        total += loss.item() * b
        count += b

    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--smplx_model_path", required=True)
    parser.add_argument("--global_checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--temporal_window", type=int, default=4)
    parser.add_argument("--num_joints", type=int, default=22)
    parser.add_argument("--scale_id", choices=["p1", "p2", "p3"], default="p3")
    parser.add_argument("--split_id", choices=["s1", "s2", "s3"], default="s2")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--out", default="checkpoints/global_diffusion_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(args.global_checkpoint, map_location=device)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    num_joints = config.get("num_joints", args.num_joints)

    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

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
        num_joints=num_joints,
    ).to(device)

    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    context_encoder.load_state_dict(state, strict=True)
    context_encoder.eval()
    for p in context_encoder.parameters():
        p.requires_grad = False

    denoiser = GlobalConditionedPoseDiffusion(
        condition_dim=64,
        hidden_dim=96,
        time_dim=96,
        num_blocks=6,
        dropout=0.1,
        num_joints=num_joints,
    ).to(device)

    optimizer = torch.optim.AdamW(
        denoiser.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    schedule = ConstantBetaSchedule(args.num_steps, args.beta).to(device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            context_encoder,
            denoiser,
            loader,
            optimizer,
            schedule,
            device,
            num_joints=num_joints,
        )
        print(f"Epoch {epoch:03d}/{args.epochs} diffusion_loss={loss:.6f}")
        torch.save(
            {
                "epoch": epoch,
                "denoiser_state": denoiser.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "num_steps": schedule.num_steps,
                "beta": args.beta,
                "num_joints": num_joints,
            },
            args.out,
        )

    print("Training complete.")
    print("DDIM sampling is intentionally left for you to implement.")


if __name__ == "__main__":
    main()
