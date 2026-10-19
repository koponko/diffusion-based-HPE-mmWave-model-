#!/usr/bin/env python3
"""
PointNet++ hyperparameter evaluation on a fixed subsample of M4Human.

This script evaluates multiple PointNet++ feature-extractor configurations
on the same examples from an RF3DPose processed-cache subset.

It evaluates multiple PointNetPPFeatureExtractor configurations on exactly
the same examples and produces:

    results/
        selected_indices.json
        summary.csv
        pointnet_visualizations/
            000_raw_input.png
            <config>/layer1/feature_*.png
            <config>/layer2/feature_*.png
            <config>/mrg/*.png

Supply a trained checkpoint to visualize learned activations. Without one,
the script only visualizes initial/random features.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.m4human_dataset import RF3DPoseDataset
from my_models.feature_extract_vol_3 import (
    PointNetPPFeatureExtractor,
    three_nn_interpolate,
)


REQUIRED_CACHE_FILES = (
    "calib.lmdb",
    "indicator.lmdb",
    "params.lmdb",
    "radar_comp.lmdb",
    "radar_pc.lmdb",
    "indeces.pkl.gz",
)
VISUALIZATION_AUX_KEYS = (
    "input_xyz",
    "stage1_xyz",
    "stage1_features",
    "stage2_xyz",
    "stage2_features",
    "mrg_lower_features",
    "mrg_raw_features",
)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_worker_init_fn(seed: int):
    def _init(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init


# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------

def default_configs():
    """
    Three useful comparisons:

      baseline:
          larger encoder + transforms + robust grouping + MRG

      light:
          fewer sampled centers and smaller feature dimension

      no_transforms:
          same PointNet++ capacity as baseline but without T-Nets

    Edit these or use --configs_json.
    """
    return [
        {
            "name": "baseline",
            "kwargs": {
                "feature_dim": 256,
                "first_sample_count": 256,
                "second_sample_count": 64,
                "use_input_transform": True,
                "use_feature_transform": True,
                "robust_grouping": True,
                "radius_scale": 2.0,
                "first_sampling_mode": "hybrid",
                "second_sampling_mode": "fps",
                "use_mrg_second_stage": True,
                "mrg_lower_neighbor_count": 16,
                "mrg_raw_neighbor_count": 64,
                "mrg_lower_radius": 0.5,
                "mrg_raw_radius": 1.0,
                "num_segmentation_classes": None,
            },
        },
        {
            "name": "light",
            "kwargs": {
                "feature_dim": 128,
                "first_sample_count": 128,
                "second_sample_count": 32,
                "use_input_transform": True,
                "use_feature_transform": True,
                "robust_grouping": True,
                "radius_scale": 2.0,
                "first_sampling_mode": "hybrid",
                "second_sampling_mode": "fps",
                "use_mrg_second_stage": True,
                "mrg_lower_neighbor_count": 16,
                "mrg_raw_neighbor_count": 32,
                "mrg_lower_radius": 0.5,
                "mrg_raw_radius": 1.0,
                "num_segmentation_classes": None,
            },
        },
        {
            "name": "no_transforms",
            "kwargs": {
                "feature_dim": 256,
                "first_sample_count": 256,
                "second_sample_count": 64,
                "use_input_transform": False,
                "use_feature_transform": False,
                "robust_grouping": True,
                "radius_scale": 2.0,
                "first_sampling_mode": "hybrid",
                "second_sampling_mode": "fps",
                "use_mrg_second_stage": True,
                "mrg_lower_neighbor_count": 16,
                "mrg_raw_neighbor_count": 64,
                "mrg_lower_radius": 0.5,
                "mrg_raw_radius": 1.0,
                "num_segmentation_classes": None,
            },
        },
    ]


def load_configs(path):
    if path is None:
        return default_configs()

    with open(path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    if not isinstance(configs, list):
        raise ValueError("--configs_json must contain a JSON list")

    for config in configs:
        if "name" not in config or "kwargs" not in config:
            raise ValueError(
                'Each config needs {"name": ..., "kwargs": {...}}'
            )

    return configs


def load_checkpoint(path, model, device):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Unsupported checkpoint format: {path}"
        )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing:
        print(f"  missing checkpoint keys: {len(missing)}")

    if unexpected:
        print(f"  unexpected checkpoint keys: {len(unexpected)}")


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def build_dataset(args):
    cache_dir = args.cache_dir.expanduser().resolve()
    processed_cache = cache_dir / "rf3dpose_all"
    missing = [
        name for name in REQUIRED_CACHE_FILES
        if not (processed_cache / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{cache_dir} is not a complete RF3DPose processed-cache subset; "
            f"missing from {processed_cache}: {', '.join(missing)}. "
            "Pass the destination directory created by "
            "create_rf3dpose_10gb_subset.py using --cache_dir."
        )

    dataset = RF3DPoseDataset(
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
    print(f"Loaded RF3DPoseDataset with {len(dataset)} examples.")
    return dataset


def get_radar_points(item):
    if not isinstance(item, dict):
        raise TypeError(
            "Dataset item must be a dictionary containing 'radar_points'"
        )

    if "radar_points" not in item:
        raise KeyError(
            "Dataset item does not contain 'radar_points'"
        )

    return torch.as_tensor(
        item["radar_points"]
    )


def normalize_item(
    radar_points,
    temporal_window,
    in_channels,
    num_points,
    seed,
):
    """
    Normalize the common dataset representation to (T,P,C).

    Supported:
        (T,P,C)
        (P,C) -> treated as T=1

    The training loop used the first three channels for xyz, so here we
    keep exactly in_channels for the feature extractor.
    """
    x = torch.as_tensor(radar_points)

    # Remove only singleton dimensions.
    while x.ndim > 3 and 1 in x.shape:
        dim = next(
            i for i, size in enumerate(x.shape)
            if size == 1
        )
        x = x.squeeze(dim)

    if x.ndim == 2:
        x = x.unsqueeze(0)

    if x.ndim != 3:
        raise ValueError(
            f"Expected radar_points with shape (T,P,C), got {tuple(x.shape)}"
        )

    # Try to repair a (P,T,C) layout when T is known and unambiguous.
    if (
        x.shape[0] != temporal_window
        and x.shape[1] == temporal_window
    ):
        x = x.transpose(0, 1)

    # For a real dataset, trust the actual temporal dimension if the
    # command-line value does not match; the model comparison still uses
    # exactly the same frames for every configuration.
    if x.shape[0] != temporal_window:
        print(
            "Warning: dataset sample has temporal dimension "
            f"{x.shape[0]}, while --temporal_window={temporal_window}. "
            "Using the dataset value for this sample."
        )

    if x.shape[-1] < in_channels:
        raise ValueError(
            f"Dataset sample has {x.shape[-1]} channels, "
            f"but model requires {in_channels}."
        )

    x = x[..., :in_channels].float()
    # RF3DPose normalizes padded points to x=y=0, z=-1.5.
    valid_points = x[..., :3].abs().sum(dim=-1).gt(1e-6).any(dim=0)
    point_indices = torch.nonzero(valid_points, as_tuple=False).flatten()
    if point_indices.numel() == 0:
        raise ValueError("Radar sample contains no non-padding points.")
    generator = torch.Generator().manual_seed(seed)
    point_indices = point_indices[
        torch.randperm(point_indices.numel(), generator=generator)
    ]
    selected = point_indices[:num_points]
    if selected.numel() < num_points:
        repeated = point_indices[
            torch.randint(
                point_indices.numel(),
                (num_points - selected.numel(),),
                generator=generator,
            )
        ]
        selected = torch.cat((selected, repeated))
    x = x.index_select(1, selected)

    return x


class FixedSubsetDataset(Dataset):
    """
    Materializes only the radar tensor and preserves the original dataset
    index for reproducibility.
    """

    def __init__(
        self,
        dataset,
        indices,
        temporal_window,
        in_channels,
        num_points,
        seed,
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.temporal_window = temporal_window
        self.in_channels = in_channels
        self.num_points = num_points
        self.seed = seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        dataset_index = self.indices[i]
        item = self.dataset[dataset_index]

        x = normalize_item(
            get_radar_points(item),
            temporal_window=self.temporal_window,
            in_channels=self.in_channels,
            num_points=self.num_points,
            seed=self.seed + dataset_index,
        )

        return {
            "radar_points": x,
            "dataset_index": dataset_index,
        }


def choose_fixed_indices(dataset, count, seed):
    count = min(count, len(dataset))

    generator = torch.Generator()
    generator.manual_seed(seed)

    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:count]

    return indices.tolist()


def collate_batch(batch):
    return {
        "radar_points": torch.stack(
            [item["radar_points"] for item in batch],
            dim=0,
        ),
        "dataset_indices": [
            item["dataset_index"]
            for item in batch
        ],
    }


# ---------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


def extract_model_features(
    model,
    loader,
    device,
    seed,
):
    """
    Deterministically evaluate one configuration.

    We reset the RNG at the start so hybrid/random sampling uses the same
    stochastic sequence for each configuration.
    """
    model.eval()

    seed_everything(seed)

    aux_batches = {}

    elapsed = 0.0

    with torch.no_grad():
        for batch in loader:
            x = batch["radar_points"].to(
                device,
                non_blocking=True,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            _, aux = model(x, return_aux=True)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed += (
                time.perf_counter() - start
            )

            for key in VISUALIZATION_AUX_KEYS:
                value = aux.get(key)
                if isinstance(value, torch.Tensor):
                    aux_batches.setdefault(key, []).append(value.detach().cpu())

    return {
        "aux": {
            key: torch.cat(values, dim=0)
            for key, values in aux_batches.items()
        },
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------


def set_equal_3d_axes(ax, xyz):
    if xyz.size == 0:
        return

    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)

    center = (mins + maxs) / 2.0
    half_range = (
        maxs - mins
    ).max() / 2.0

    half_range = max(
        half_range,
        1e-6,
    )

    ax.set_xlim(
        center[0] - half_range,
        center[0] + half_range,
    )
    ax.set_ylim(
        center[1] - half_range,
        center[1] + half_range,
    )
    ax.set_zlim(
        center[2] - half_range,
        center[2] + half_range,
    )


def _scatter_cloud(ax, xyz, values=None, title="", cmap="coolwarm"):
    if values is None:
        artist = ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=10,
            c="steelblue",
            alpha=0.8,
        )
    else:
        low, high = np.percentile(values, [2, 98])
        if low == high:
            low, high = float(values.min()), float(values.max()) + 1e-6
        artist = ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=18,
            c=values,
            cmap=cmap,
            vmin=low,
            vmax=high,
        )
        ax.figure.colorbar(artist, ax=ax, shrink=0.7, label="Activation")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_equal_3d_axes(ax, xyz)


def _top_variable_channels(features, count):
    variation = features.float().std(dim=0, unbiased=False)
    return torch.argsort(variation, descending=True)[: min(count, features.shape[-1])]


def save_raw_input(xyz, path, sample_index, frame_index):
    frame_xyz = xyz[sample_index, frame_index].numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    _scatter_cloud(ax, frame_xyz, title="Raw mmWave radar points")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_stage_activation_maps(
    xyz,
    centers,
    features,
    out_dir,
    sample_index,
    frame_index,
    top_k,
    stage_name,
):
    source_xyz = centers[sample_index, frame_index].unsqueeze(0)
    source_features = features[sample_index, frame_index].unsqueeze(0)
    query_xyz = xyz[sample_index, frame_index].unsqueeze(0)
    point_features = three_nn_interpolate(
        query_xyz=query_xyz,
        source_xyz=source_xyz,
        source_features=source_features,
    )[0]
    raw_xyz = query_xyz[0].numpy()
    out_dir.mkdir(parents=True, exist_ok=True)

    for channel in _top_variable_channels(source_features[0], top_k).tolist():
        activation = point_features[:, channel].numpy()
        fig = plt.figure(figsize=(11, 5))
        input_ax = fig.add_subplot(121, projection="3d")
        activation_ax = fig.add_subplot(122, projection="3d")
        _scatter_cloud(input_ax, raw_xyz, title="Input points")
        _scatter_cloud(
            activation_ax,
            raw_xyz,
            values=activation,
            title=f"{stage_name} feature channel {channel}",
        )
        fig.suptitle(
            f"{stage_name} activation — sample {sample_index}, frame {frame_index}"
        )
        fig.tight_layout()
        fig.savefig(
            out_dir
            / (
                f"feature_{channel:03d}_sample_{sample_index:03d}_"
                f"frame_{frame_index:03d}.png"
            ),
            dpi=180,
        )
        plt.close(fig)


def save_mrg_branch_maps(result, out_dir, sample_index, frame_index):
    aux = result["aux"]
    centers = aux["stage2_xyz"][sample_index, frame_index]
    branches = {
        "lower_resolution": aux.get("mrg_lower_features"),
        "raw_points": aux.get("mrg_raw_features"),
        "combined": aux.get("stage2_features"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    for branch_name, branch_batch in branches.items():
        if branch_batch is None:
            continue
        branch_features = branch_batch[sample_index, frame_index]
        channel = int(_top_variable_channels(branch_features, 1)[0])
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        _scatter_cloud(
            ax,
            centers.numpy(),
            values=branch_features[:, channel].numpy(),
            title=f"MRG {branch_name} branch — channel {channel}",
        )
        fig.tight_layout()
        fig.savefig(
            out_dir
            / (
                f"{branch_name}_sample_{sample_index:03d}_"
                f"frame_{frame_index:03d}.png"
            ),
            dpi=180,
        )
        plt.close(fig)


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------

def save_summary(
    results,
    configs,
    out_dir,
):
    rows = []

    for config in configs:
        name = config["name"]
        result = results[name]
        rows.append(
            {
                "name": name,
                "parameters": result["parameters"],
                "forward_seconds": result["elapsed"],
                "kwargs": json.dumps(
                    config["kwargs"],
                    sort_keys=True,
                ),
            }
        )

    csv_path = out_dir / "summary.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Saved summary: {csv_path}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare PointNet++ hyperparameter configurations on a "
            "fixed M4Human subsample."
        )
    )

    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=ROOT_DIR / "rf3dpose_10gb",
        help="Destination directory produced by create_rf3dpose_10gb_subset.py.",
    )

    parser.add_argument(
        "--smplx_model_path",
        type=Path,
        default=ROOT_DIR / "models",
        help="Directory containing the SMPL-X model files.",
    )

    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
    )

    parser.add_argument(
        "--scale_id",
        choices=("p1", "p2", "p3"),
        default="p3",
    )

    parser.add_argument(
        "--split_id",
        choices=("s1", "s2", "s3"),
        default="s2",
    )

    parser.add_argument(
        "--in_channels",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num_points",
        type=int,
        default=256,
        help="Maximum points per frame used for the feature extractor.",
    )

    parser.add_argument(
        "--temporal_window",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--configs_json",
        type=str,
        default=None,
        help="JSON file containing a list of named model configurations.",
    )

    parser.add_argument(
        "--checkpoints_json",
        type=str,
        default=None,
        help="JSON mapping config names to feature-extractor checkpoint paths.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Evaluate one trained baseline checkpoint instead of comparing random configs.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="pointnet_test_results",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--frames_to_visualize",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--visualize_examples",
        type=int,
        default=1,
        help="Number of selected examples to render as point-cloud visualizations.",
    )

    parser.add_argument(
        "--num_visual_features",
        type=int,
        default=16,
        help="Number of high-variation channels to visualize at each stage.",
    )

    args = parser.parse_args()

    if args.in_channels < 3:
        parser.error("--in_channels must be at least 3 for XYZ coordinates.")
    if args.num_points <= 0:
        parser.error("--num_points must be greater than zero.")
    if args.temporal_window <= 0:
        parser.error("--temporal_window must be greater than zero.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be greater than zero.")
    if args.num_examples <= 0:
        parser.error("--num_examples must be greater than zero.")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative.")
    if args.frames_to_visualize <= 0:
        parser.error("--frames_to_visualize must be greater than zero.")
    if args.visualize_examples <= 0:
        parser.error("--visualize_examples must be greater than zero.")
    if args.num_visual_features <= 0:
        parser.error("--num_visual_features must be greater than zero.")

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    dataset = build_dataset(
        args
    )

    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    selected_indices = choose_fixed_indices(
        dataset,
        args.num_examples,
        args.seed,
    )

    with open(
        out_dir / "selected_indices.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            selected_indices,
            f,
            indent=2,
        )

    fixed_dataset = FixedSubsetDataset(
        dataset=dataset,
        indices=selected_indices,
        temporal_window=args.temporal_window,
        in_channels=args.in_channels,
        num_points=args.num_points,
        seed=args.seed,
    )

    loader = DataLoader(
        fixed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=make_worker_init_fn(
            args.seed
        ),
        collate_fn=collate_batch,
    )

    # -------------------------------------------------------------
    # Configurations
    # -------------------------------------------------------------

    configs = load_configs(
        args.configs_json
    )

    checkpoints = {}

    if args.checkpoints_json is not None:
        with open(
            args.checkpoints_json,
            "r",
            encoding="utf-8",
        ) as f:
            checkpoints = json.load(f)

    if args.checkpoint is not None:
        if args.checkpoints_json is not None:
            parser.error("Use either --checkpoint or --checkpoints_json, not both.")
        if args.configs_json is None:
            configs = [config for config in configs if config["name"] == "baseline"]
        if len(configs) != 1:
            parser.error(
                "--checkpoint requires one configuration; use the default "
                "baseline or provide a one-entry --configs_json file."
            )
        checkpoints[configs[0]["name"]] = str(
            args.checkpoint.expanduser().resolve()
        )

    results = {}

    # -------------------------------------------------------------
    # Evaluate each configuration
    # -------------------------------------------------------------

    for config in configs:
        name = config["name"]

        kwargs = dict(
            config["kwargs"]
        )

        kwargs.setdefault(
            "in_channels",
            args.in_channels,
        )

        print(
            f"\n[{name}]"
        )

        # Reset seed BEFORE construction so parameter initialization is
        # deterministic between runs.
        seed_everything(
            args.seed
        )

        model = PointNetPPFeatureExtractor(
            **kwargs
        ).to(device)

        parameter_count = count_parameters(
            model
        )

        checkpoint_path = checkpoints.get(
            name
        )

        if checkpoint_path:
            print(
                f"Loading checkpoint: {checkpoint_path}"
            )

            load_checkpoint(
                checkpoint_path,
                model,
                device,
            )

        print(
            f"Parameters: {parameter_count:,}"
        )

        # Reset again so stochastic hybrid/random sampling is comparable.
        seed_everything(
            args.seed
        )

        result = extract_model_features(
            model=model,
            loader=loader,
            device=device,
            seed=args.seed,
        )

        result["model"] = model
        results[name] = result

        print(
            f"Forward time: {result['elapsed']:.4f} s"
        )

    # Only retain diagnostic representations after inference; do not write
    # large embedding/input arrays that are not used by the visualizations.
    for result in results.values():
        result["parameters"] = count_parameters(result["model"])
        del result["model"]

    # Visualize raw points, stage activations, and MRG branches.
    sample_count = min(
        args.visualize_examples,
        len(fixed_dataset),
    )
    visualization_dir = out_dir / "pointnet_visualizations"

    for sample_index in range(
        sample_count
    ):
        # Determine T from the fixed sample.
        sample = fixed_dataset[
            sample_index
        ]["radar_points"]

        frame_count = min(
            args.frames_to_visualize,
            sample.shape[0],
        )

        for frame_index in range(
            frame_count
        ):
            if sample_index == 0 and frame_index == 0:
                save_raw_input(
                    results[next(iter(results))]["aux"]["input_xyz"],
                    visualization_dir / "000_raw_input.png",
                    sample_index,
                    frame_index,
                )

            for name, result in results.items():
                safe_name = name.replace("/", "_").replace(" ", "_")
                model_dir = visualization_dir / safe_name
                save_stage_activation_maps(
                    xyz=result["aux"]["input_xyz"],
                    centers=result["aux"]["stage1_xyz"],
                    features=result["aux"]["stage1_features"],
                    out_dir=model_dir / "layer1",
                    sample_index=sample_index,
                    frame_index=frame_index,
                    top_k=args.num_visual_features,
                    stage_name="Layer 1",
                )
                save_stage_activation_maps(
                    xyz=result["aux"]["input_xyz"],
                    centers=result["aux"]["stage2_xyz"],
                    features=result["aux"]["stage2_features"],
                    out_dir=model_dir / "layer2",
                    sample_index=sample_index,
                    frame_index=frame_index,
                    top_k=args.num_visual_features,
                    stage_name="Layer 2",
                )
                save_mrg_branch_maps(
                    result,
                    model_dir / "mrg",
                    sample_index,
                    frame_index,
                )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    save_summary(
        results=results,
        configs=configs,
        out_dir=out_dir,
    )

    print(
        "\nEvaluation complete."
    )
    print(
        f"Results: {out_dir.resolve()}"
    )


if __name__ == "__main__":
    main()