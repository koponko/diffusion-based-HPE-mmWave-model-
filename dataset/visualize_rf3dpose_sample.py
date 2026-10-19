#!/usr/bin/env python3
"""Visualize radar input and ground truth for a cached RF3DPose sample.

Example:
python dataset/visualize_rf3dpose_sample.py \
  --cache-dir /home/koponko/M4Human/rf3dpose_10gb \
  --split train --scale p3 --split-id s2 --index 0 \
  --smplx-model-path models/smplx \
  --output-dir visualizations
"""
from __future__ import annotations

import argparse
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INDEX_NAME = "indeces.pkl.gz"
DEFAULT_RADAR_SHAPE = (121, 111, 31)
DATABASE_NAMES = (
    "calib",
    "indicator",
    "params",
    "radar_comp",
    "radar_pc",
    "image",
    "bbox",
    "rgb_feat",
)


def resolve_cache_dir(path: Path) -> Path:
    if (path / INDEX_NAME).is_file():
        return path
    nested = path / "rf3dpose_all"
    if (nested / INDEX_NAME).is_file():
        return nested
    raise FileNotFoundError(
        f"Could not find {INDEX_NAME} in {path} or {nested}. "
        "Pass the cache directory or its parent as --cache-dir."
    )


def parse_indicator(values: list[str] | None) -> tuple[int, int, int] | None:
    if values is None:
        return None
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            "--indicator requires three values: PERSON ACTION FRAME (e.g. p1 10 42)."
        )
    person_text = values[0].lower()
    if person_text.startswith("p"):
        person_text = person_text[1:]
    try:
        person, action, frame = int(person_text), int(values[1]), int(values[2])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--indicator values must be integers, with an optional 'p' before PERSON."
        ) from error
    if person <= 0 or action <= 0 or frame < 0:
        raise argparse.ArgumentTypeError(
            "PERSON and ACTION must be positive; FRAME must be non-negative."
        )
    return person, action, frame


def read_split_indicators(cache_dir: Path, scale: str, split_id: str, split: str):
    with gzip.open(cache_dir / INDEX_NAME, "rb") as stream:
        index = pickle.load(stream)
    try:
        indicators = index[scale][split_id][split]
    except KeyError as error:
        raise SystemExit(
            f"Saved split index has no {scale}/{split_id}/{split} entry. "
            f"Available scales: {list(index)}."
        ) from error
    return [tuple(int(part) for part in item) for item in indicators]


def load_record(cache_dir: Path, indicator: tuple[int, int, int], radar_shape):
    import lmdb
    import msgpack

    from dataset.lmdb_utils import unpack_dict_np, unpack_ndarray

    key = str(list(indicator)).encode()
    values = {}
    for name in ("radar_comp", "radar_pc", "params", "calib"):
        path = cache_dir / f"{name}.lmdb"
        if not path.is_file():
            raise FileNotFoundError(f"Required cache database is missing: {path}")
        env = lmdb.open(
            str(path), subdir=False, readonly=True, lock=False, readahead=False
        )
        try:
            with env.begin() as txn:
                value = txn.get(key)
        finally:
            env.close()
        if value is None:
            raise KeyError(f"Record {indicator} is missing from {path.name}.")
        values[name] = value

    sparse_volume = unpack_ndarray(values["radar_comp"])
    if sparse_volume.ndim != 2 or sparse_volume.shape[1] != 4:
        raise ValueError(
            f"Expected sparse radar volume with shape (N, 4), got {sparse_volume.shape}."
        )
    radar_volume = np.zeros(radar_shape, dtype=sparse_volume.dtype)
    if sparse_volume.size:
        coordinates = sparse_volume[:, :3].astype(np.int64)
        radar_volume[tuple(coordinates.T)] = sparse_volume[:, 3]
    radar_points = unpack_ndarray(values["radar_pc"]).copy()
    params = unpack_dict_np(values["params"])
    calibration = unpack_dict_np(values["calib"])

    result = {
        "radar_volume": radar_volume,
        "radar_points": radar_points,
        "params": params,
        "calibration": calibration,
        "image": None,
        "optional_databases": {},
    }

    for name in ("image", "bbox", "rgb_feat"):
        path = cache_dir / f"{name}.lmdb"
        if not path.is_file():
            continue
        env = lmdb.open(
            str(path), subdir=False, readonly=True, lock=False, readahead=False
        )
        try:
            with env.begin() as txn:
                value = txn.get(key)
                entry_count = env.stat()["entries"]
        finally:
            env.close()
        result["optional_databases"][name] = entry_count
        if name == "image" and value is not None:
            result["image"] = unpack_ndarray(value)

    indicator_path = cache_dir / "indicator.lmdb"
    if not indicator_path.is_file():
        raise FileNotFoundError(f"Required cache database is missing: {indicator_path}")
    indicator_env = lmdb.open(
        str(indicator_path), subdir=False, readonly=True, lock=False, readahead=False
    )
    try:
        with indicator_env.begin() as txn:
            indicator_value = txn.get(key)
    finally:
        indicator_env.close()
    if indicator_value is None:
        raise KeyError(f"Record {indicator} is missing from indicator.lmdb.")
    stored_indicator = tuple(msgpack.unpackb(indicator_value, raw=False))
    if stored_indicator != indicator:
        raise ValueError(
            f"Indicator key/value mismatch: requested {indicator}, stored {stored_indicator}."
        )
    return result


def reconstruct_ground_truth(params, calibration, smplx_model_path, normalize):
    try:
        import smplx
        import torch
    except ImportError as error:
        raise SystemExit(
            "SMPL-X mesh reconstruction requires torch and smplx. Install them "
            "in this Python environment, or use --no-mesh to visualize radar "
            "and the cached ground-truth parameters without mesh reconstruction."
        ) from error

    from dataset.data_loader_Load_data import gender_info
    from dataset.data_loader_camera_calibration import (
        calibrate_param_to_radar,
        compute_joints_from_vertices,
        generate_vertices_in_radar_space,
    )

    person_id = int(params.get("person_id", 0))
    subject_name = f"P{person_id}"
    if subject_name not in gender_info:
        raise ValueError(f"No gender metadata exists for subject P{person_id}.")
    gender_value = gender_info[subject_name]
    gender_name = "male" if gender_value else "female"
    model = smplx.create(
        model_path=str(smplx_model_path),
        model_type="smplx",
        gender=gender_name,
    )
    model.eval()
    model_map = {gender_name: model}
    params = dict(params)
    params["gender"] = gender_value
    radar_params = calibrate_param_to_radar(params, calibration)
    if normalize:
        radar_params["trans"] = np.asarray(radar_params["trans"]).copy()
        radar_params["trans"][..., 2] -= 1.5
    with torch.inference_mode():
        vertices_mm = generate_vertices_in_radar_space(model_map, radar_params)
        joints_m = compute_joints_from_vertices(
            model_map, vertices_mm, radar_params["gender"]
        )
    return vertices_mm / 1000.0, np.asarray(joints_m), model.faces


def transform_annotated_joints(params, calibration, normalize):
    joints_vicon = np.asarray(params["joints"], dtype=np.float64)
    if joints_vicon.ndim != 2 or joints_vicon.shape[1] != 3:
        raise ValueError(
            f"Expected annotated joints with shape (J, 3), got {joints_vicon.shape}."
        )
    joints_camera = (
        calibration["vicon_to_cam_rotmatrix"] @ joints_vicon.T
    ).T + np.asarray(calibration["vicon_to_cam_tvec"]) / 1000.0
    joints_radar = (
        np.linalg.inv(calibration["radar_to_cam_rotmatrix"])
        @ (joints_camera - calibration["radar_to_cam_tvec"]).T
    ).T
    if normalize:
        joints_radar[:, 2] -= 1.5
    return joints_radar


def add_equalish_3d_limits(ax, points: np.ndarray, padding: float = 0.15):
    if points.size == 0:
        return
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    center = (mins + maxs) / 2
    radius = max(float(np.max(maxs - mins)) / 2, 0.1) * (1 + padding)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def render_sample(
    indicator,
    record,
    vertices_m,
    joints_m,
    faces,
    output_path: Path,
    show: bool,
    normalize: bool,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    volume = record["radar_volume"]
    points = record["radar_points"].copy()
    if normalize and points.size:
        points[:, 2] -= 1.5
    finite_rows = np.all(np.isfinite(points), axis=1)
    nonzero_points = points[finite_rows & np.any(points != 0, axis=1)]
    image = record["image"]
    if image is not None and image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
        if image.shape[-1] == 1:
            image = image[..., 0]

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    axes = [
        fig.add_subplot(2, 3, 1),
        fig.add_subplot(2, 3, 2),
        fig.add_subplot(2, 3, 3, projection="3d"),
        fig.add_subplot(2, 3, 4, projection="3d"),
        fig.add_subplot(2, 3, 5, projection="3d"),
        fig.add_subplot(2, 3, 6),
    ]
    fig.suptitle(
        f"RF3DPose sample P{indicator[0]} / action {indicator[1]} / frame {indicator[2]}",
        fontsize=16,
    )

    projection_axes = (
        (axes[0], np.max(volume, axis=2).T, "Radar volume: max over Z (XY)"),
        (axes[1], np.max(volume, axis=1).T, "Radar volume: max over Y (XZ)"),
    )
    for ax, projection, title in projection_axes:
        img = ax.imshow(projection, origin="lower", aspect="auto", cmap="inferno")
        ax.set_title(title)
        ax.set_xlabel("Grid index")
        ax.set_ylabel("Grid index")
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label="Radar value")

    point_ax = axes[2]
    if len(nonzero_points):
        scatter = point_ax.scatter(
            nonzero_points[:, 0],
            nonzero_points[:, 1],
            nonzero_points[:, 2],
            c=nonzero_points[:, 3] if nonzero_points.shape[1] > 3 else "tab:blue",
            cmap="viridis",
            s=2,
            alpha=0.75,
        )
        if nonzero_points.shape[1] > 3:
            fig.colorbar(scatter, ax=point_ax, fraction=0.046, pad=0.04, label="Intensity")
        add_equalish_3d_limits(point_ax, nonzero_points[:, :3])
    point_ax.set_title(f"Radar point cloud ({len(nonzero_points):,} nonzero points)")
    point_ax.set_xlabel("X (m)")
    point_ax.set_ylabel("Y (m)")
    point_ax.set_zlabel("Z (m)")

    overlay_ax = axes[3]
    if len(nonzero_points):
        overlay_ax.scatter(
            nonzero_points[:, 0],
            nonzero_points[:, 1],
            nonzero_points[:, 2],
            c="black",
            s=1,
            alpha=0.3,
            label="Radar points",
        )
    if vertices_m is not None and len(vertices_m) and len(faces):
        mesh = Poly3DCollection(
            vertices_m[faces],
            facecolor="tomato",
            edgecolor="darkred",
            linewidth=0.05,
            alpha=0.32,
        )
        overlay_ax.add_collection3d(mesh)
        overlay_ax.scatter(
            vertices_m[::30, 0],
            vertices_m[::30, 1],
            vertices_m[::30, 2],
            c="red",
            s=1,
            alpha=0.4,
        )
    if len(nonzero_points) and vertices_m is not None and len(vertices_m):
        add_equalish_3d_limits(
            overlay_ax, np.concatenate((nonzero_points[:, :3], vertices_m), axis=0)
        )
    elif vertices_m is not None and len(vertices_m):
        add_equalish_3d_limits(overlay_ax, vertices_m)
    overlay_ax.set_title(
        "Ground-truth SMPL-X mesh + radar points"
        if vertices_m is not None
        else "Radar points (mesh reconstruction disabled)"
    )
    overlay_ax.set_xlabel("X (m)")
    overlay_ax.set_ylabel("Y (m)")
    overlay_ax.set_zlabel("Z (m)")
    if len(nonzero_points):
        overlay_ax.legend(loc="upper left")

    joints_ax = axes[4]
    if len(nonzero_points):
        joints_ax.scatter(
            nonzero_points[:, 0],
            nonzero_points[:, 1],
            nonzero_points[:, 2],
            c="lightgray",
            s=1,
            alpha=0.25,
        )
    if joints_m is not None and len(joints_m):
        joints_ax.scatter(
            joints_m[:, 0],
            joints_m[:, 1],
            joints_m[:, 2],
            c=np.arange(len(joints_m)),
            cmap="turbo",
            s=22,
            depthshade=False,
        )
        for joint_id, xyz in enumerate(joints_m):
            joints_ax.text(*xyz, str(joint_id), fontsize=7)
        if len(nonzero_points):
            add_equalish_3d_limits(
                joints_ax, np.concatenate((nonzero_points[:, :3], joints_m), axis=0)
            )
        else:
            add_equalish_3d_limits(joints_ax, joints_m)
    joints_ax.set_title(
        "Ground-truth SMPL-X joints (index labels)"
        if joints_m is not None
        else "Ground-truth joints (mesh reconstruction disabled)"
    )
    joints_ax.set_xlabel("X (m)")
    joints_ax.set_ylabel("Y (m)")
    joints_ax.set_zlabel("Z (m)")

    info_ax = axes[5]
    info_ax.axis("off")
    params = record["params"]
    calibration = record["calibration"]
    lines = [
        "Ground-truth parameters",
        *[
            f"  {name}: shape {np.asarray(params[name]).shape}"
            for name in ("joints", "betas", "pose_body", "trans", "root_orient")
            if name in params
        ],
        f"  trans: {np.asarray(params['trans']).reshape(-1)[:3]}",
        f"  root_orient: {np.asarray(params['root_orient']).reshape(-1)[:3]}",
        "",
        "Radar / calibration",
        f"  radar volume shape: {volume.shape}",
        f"  nonzero voxels: {np.count_nonzero(volume):,}",
        f"  raw point array: {record['radar_points'].shape}",
        f"  radar value range: [{float(np.min(volume)):.4g}, {float(np.max(volume)):.4g}]",
        f"  calibration fields: {', '.join(calibration)}",
        "",
        "Other cache databases",
        *[
            f"  {name}.lmdb: {count:,} entries"
            for name, count in record["optional_databases"].items()
        ],
    ]
    if vertices_m is None:
        lines.extend(("", "SMPL-X mesh reconstruction disabled."))
    if image is not None:
        lines.extend(("", f"RGB image cache found: shape {image.shape}"))
    else:
        lines.extend(("", "No decodable image.lmdb record for this frame."))
    info_ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        wrap=True,
    )
    if image is not None:
        inset = info_ax.inset_axes([0.48, 0.02, 0.5, 0.36])
        inset.imshow(image)
        inset.set_title("Optional RGB image")
        inset.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(f"Saved visualization: {output_path}")
    print(f"Indicator: {indicator}")
    print(f"Radar volume: {volume.shape}, {np.count_nonzero(volume):,} nonzero voxels")
    print(f"Radar points: {len(nonzero_points):,} nonzero rows")
    if vertices_m is None:
        print("Ground-truth SMPL-X mesh: skipped")
    else:
        print(f"Ground truth: {len(vertices_m):,} mesh vertices, {len(joints_m):,} joints")
    if show:
        plt.show()
    plt.close(fig)


def render_gif(
    indicators,
    cache_dir: Path,
    radar_shape,
    output_path: Path,
    duration_ms: int,
    normalize: bool,
    smplx_model_path: Path,
    no_mesh: bool,
):
    from PIL import Image
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    frames = []
    for indicator in indicators:
        record = load_record(cache_dir, indicator, radar_shape)
        volume = record["radar_volume"]
        points = record["radar_points"].copy()
        if normalize and points.size:
            points[:, 2] -= 1.5
        finite = np.all(np.isfinite(points), axis=1)
        visible_points = points[finite & np.any(points != 0, axis=1)]

        vertices_m = faces = None
        joints_m = transform_annotated_joints(
            record["params"], record["calibration"], normalize
        )
        if not no_mesh:
            record["params"]["person_id"] = indicator[0]
            vertices_m, _, faces = reconstruct_ground_truth(
                record["params"],
                record["calibration"],
                smplx_model_path,
                normalize,
            )

        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        ax_xy = fig.add_subplot(1, 3, 1)
        ax_xz = fig.add_subplot(1, 3, 2)
        ax_3d = fig.add_subplot(1, 3, 3, projection="3d")
        fig.suptitle(
            f"P{indicator[0]} / action {indicator[1]} / frame {indicator[2]}",
            fontsize=15,
        )

        ax_xy.imshow(np.max(volume, axis=2).T, origin="lower", cmap="inferno", aspect="auto")
        ax_xy.set_title("Radar projection (XY)")
        ax_xy.set_xlabel("X grid index")
        ax_xy.set_ylabel("Y grid index")

        ax_xz.imshow(np.max(volume, axis=1).T, origin="lower", cmap="inferno", aspect="auto")
        ax_xz.set_title("Radar projection (XZ)")
        ax_xz.set_xlabel("X grid index")
        ax_xz.set_ylabel("Z grid index")

        combined = [visible_points[:, :3]] if len(visible_points) else []
        if len(visible_points):
            ax_3d.scatter(
                visible_points[:, 0],
                visible_points[:, 1],
                visible_points[:, 2],
                c="black",
                s=2,
                alpha=0.55,
                label="Radar points",
            )
        if vertices_m is not None and faces is not None:
            ax_3d.add_collection3d(
                Poly3DCollection(
                    vertices_m[faces],
                    facecolor="tomato",
                    edgecolor="darkred",
                    linewidth=0.03,
                    alpha=0.25,
                )
            )
            combined.append(vertices_m)
        if joints_m is not None and len(joints_m):
            ax_3d.scatter(
                joints_m[:, 0],
                joints_m[:, 1],
                joints_m[:, 2],
                c="blue",
                s=9,
                label="GT joints",
            )
            combined.append(joints_m)
        if combined:
            add_equalish_3d_limits(ax_3d, np.concatenate(combined, axis=0))
        ax_3d.set_title(
            "Radar + ground truth" if vertices_m is not None else "Radar point cloud"
        )
        ax_3d.set_xlabel("X (m)")
        ax_3d.set_ylabel("Y (m)")
        ax_3d.set_zlabel("Z (m)")
        if len(visible_points) or (joints_m is not None and len(joints_m)):
            ax_3d.legend(loc="upper left")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
        frames.append(Image.fromarray(rgba).convert("P", palette=Image.Palette.ADAPTIVE))
        plt.close(fig)

    if not frames:
        raise ValueError("Cannot create a GIF without at least one frame.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"Saved GIF ({len(frames)} frames): {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Cache directory itself or its parent (contains rf3dpose_all/).",
    )
    parser.add_argument("--scale", choices=("p1", "p2", "p3"), default="p3")
    parser.add_argument("--split-id", choices=("s1", "s2", "s3"), default="s2")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--index", type=int, default=0, help="Position in the saved split.")
    parser.add_argument(
        "--indicator",
        nargs=3,
        metavar=("PERSON", "ACTION", "FRAME"),
        help="Select an exact indicator instead of --index, e.g. p1 10 42.",
    )
    parser.add_argument("--radar-shape", nargs=3, type=int, default=DEFAULT_RADAR_SHAPE)
    parser.add_argument(
        "--smplx-model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "smplx",
        help="SMPL-X model directory used to reconstruct ground-truth mesh.",
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help="Skip SMPL-X reconstruction; plot radar and display cached GT parameters.",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Also make a short GIF from consecutive frames of the same action.",
    )
    parser.add_argument(
        "--gif-frames",
        type=int,
        default=8,
        help="Maximum number of consecutive frames in the GIF (default: 8).",
    )
    parser.add_argument(
        "--gif-duration-ms",
        type=int,
        default=250,
        help="Display duration per GIF frame in milliseconds (default: 250).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualizations"),
        help="Directory for the saved PNG.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not apply the loader's default Z offset of -1.5 m.",
    )
    parser.add_argument("--show", action="store_true", help="Also open an interactive window.")
    args = parser.parse_args()
    if args.index < 0:
        parser.error("--index must be non-negative.")
    if args.gif_frames <= 0:
        parser.error("--gif-frames must be positive.")
    if args.gif_duration_ms <= 0:
        parser.error("--gif-duration-ms must be positive.")
    if any(size <= 0 for size in args.radar_shape):
        parser.error("--radar-shape values must be positive.")

    cache_dir = resolve_cache_dir(args.cache_dir.expanduser().resolve())
    indicators = read_split_indicators(cache_dir, args.scale, args.split_id, args.split)
    try:
        exact_indicator = parse_indicator(args.indicator)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if exact_indicator is not None:
        if exact_indicator not in indicators:
            raise SystemExit(
                f"Indicator {exact_indicator} is not in the selected "
                f"{args.scale}/{args.split_id}/{args.split} split."
            )
        sample_indicator = exact_indicator
    else:
        if args.index >= len(indicators):
            raise SystemExit(
                f"Index {args.index} is out of range for a split with "
                f"{len(indicators)} records."
            )
        sample_indicator = indicators[args.index]

    record = load_record(cache_dir, sample_indicator, tuple(args.radar_shape))
    record["params"]["person_id"] = sample_indicator[0]
    normalize = not args.no_normalize
    if args.no_mesh:
        vertices_m = faces = None
        joints_m = transform_annotated_joints(
            record["params"], record["calibration"], normalize
        )
    else:
        vertices_m, _, faces = reconstruct_ground_truth(
            record["params"],
            record["calibration"],
            args.smplx_model_path.expanduser().resolve(),
            normalize,
        )
        joints_m = transform_annotated_joints(
            record["params"], record["calibration"], normalize
        )
    filename = (
        f"p{sample_indicator[0]}_action{sample_indicator[1]}_"
        f"frame{sample_indicator[2]}_{args.scale}_{args.split_id}_{args.split}.png"
    )
    render_sample(
        sample_indicator,
        record,
        vertices_m,
        joints_m,
        faces,
        args.output_dir.expanduser().resolve() / filename,
        args.show,
        normalize,
    )
    if args.gif:
        sequence = [sample_indicator]
        indicators_set = set(indicators)
        for frame_id in range(sample_indicator[2] + 1, sample_indicator[2] + args.gif_frames):
            next_indicator = (sample_indicator[0], sample_indicator[1], frame_id)
            if next_indicator not in indicators_set:
                break
            sequence.append(next_indicator)
        gif_path = (
            args.output_dir.expanduser().resolve()
            / f"{Path(filename).stem}.gif"
        )
        render_gif(
            sequence,
            cache_dir,
            tuple(args.radar_shape),
            gif_path,
            args.gif_duration_ms,
            normalize,
            args.smplx_model_path.expanduser().resolve(),
            args.no_mesh,
        )


if __name__ == "__main__":
    main()
