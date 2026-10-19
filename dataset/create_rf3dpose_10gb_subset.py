#!/usr/bin/env python3
"""Create a size-limited, loadable subset of an RF3DPose LMDB cache.

The source may be either the cache directory itself or its parent. The output
is a cache parent containing ``rf3dpose_all/``, ready to use as ``cache_dir``
with ``RF3DPoseDataset``. Select subjects and/or action IDs to fit the size
limit; every frame in each selected person/action pair is retained.

Example:
python create_rf3dpose_10gb_subset.py \
  --source /data/rf3dpose_all \
  --destination /data/rf3dpose_subset \
  --target-gib 10 \
  --select-people p1 p3 \
  --select-actions 2 7
"""
from __future__ import annotations

import argparse
import gzip
import pickle
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

REQUIRED_DATABASES = {
    "calib.lmdb",
    "indicator.lmdb",
    "params.lmdb",
    "radar_comp.lmdb",
    "radar_pc.lmdb",
}
INDEX_NAME = "indeces.pkl.gz"


def resolve_cache_path(path: Path) -> Path:
    """Accept either a cache directory or its parent directory."""
    if (path / INDEX_NAME).is_file():
        return path
    nested = path / "rf3dpose_all"
    if (nested / INDEX_NAME).is_file():
        return nested
    return path


def selected_actions(
    cache_dir: Path,
    people: set[int] | None = None,
    actions: set[int] | None = None,
):
    import lmdb
    import msgpack

    database_paths = sorted(cache_dir.glob("*.lmdb"))
    present = {p.name for p in database_paths}
    missing = sorted(REQUIRED_DATABASES - present)
    if missing:
        raise SystemExit(
            f"{cache_dir} is not a supported RF3DPose cache; missing: "
            + ", ".join(missing)
        )

    sizes_by_key: dict[bytes, int] = defaultdict(int)
    for path in database_paths:
        env = lmdb.open(
            str(path), subdir=False, readonly=True, lock=False, readahead=False
        )
        try:
            with env.begin() as txn:
                for key, value in txn.cursor():
                    sizes_by_key[key] += len(key) + len(value)
        finally:
            env.close()

    indicator_env = lmdb.open(
        str(cache_dir / "indicator.lmdb"),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
    )
    groups: dict[tuple[int, int], list[tuple[int, bytes, int]]] = defaultdict(list)
    try:
        with indicator_env.begin() as txn:
            for key, value in txn.cursor():
                indicator = tuple(msgpack.unpackb(value, raw=False))
                if len(indicator) != 3 or not all(
                    isinstance(part, int) for part in indicator
                ):
                    raise SystemExit(f"Invalid indicator record in {cache_dir}: {value!r}")
                person, action, frame = indicator
                record_size = sizes_by_key.get(key)
                if record_size is None:
                    raise SystemExit(f"Missing cache records for indicator {indicator}.")
                groups[(person, action)].append((frame, key, record_size))
    finally:
        indicator_env.close()

    for frames in groups.values():
        frames.sort(key=lambda item: item[0])

    matching_groups = {
        key: frames
        for key, frames in groups.items()
        if (people is None or key[0] in people)
        and (actions is None or key[1] in actions)
    }
    if not matching_groups:
        available_people = sorted({person for person, _ in groups})
        available_actions = sorted({action for _, action in groups})
        raise SystemExit(
            "No actions matched the requested filters. "
            f"Available subject IDs: {available_people}; "
            f"available action IDs: {available_actions}."
        )

    selected: list[tuple[tuple[int, int, int], bytes]] = []
    used_bytes = 0
    for action_key in sorted(matching_groups):
        for frame, key, record_size in matching_groups[action_key]:
            selected.append(((action_key[0], action_key[1], frame), key))
            used_bytes += record_size

    return selected, used_bytes, database_paths, len(matching_groups)


def parse_person_id(value: str) -> int:
    normalized = value.lower()
    if normalized.startswith("p"):
        normalized = normalized[1:]
    try:
        person_id = int(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid subject ID {value!r}; use a number or p<number>"
        ) from error
    if person_id <= 0:
        raise argparse.ArgumentTypeError("subject IDs must be positive.")
    return person_id


def parse_action_id(value: str) -> int:
    try:
        action_id = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid action ID {value!r}; use a positive integer"
        ) from error
    if action_id <= 0:
        raise argparse.ArgumentTypeError("action IDs must be positive.")
    return action_id


def trim_indices(source_index: Path, destination_index: Path, selected):
    selected_indicators = {indicator for indicator, _ in selected}
    with gzip.open(source_index, "rb") as stream:
        index = pickle.load(stream)

    for schemes in index.values():
        for split_data in schemes.values():
            for split_name in ("train", "val", "test"):
                split_data[split_name] = [
                    item
                    for item in split_data[split_name]
                    if tuple(item) in selected_indicators
                ]

    with gzip.open(destination_index, "wb") as stream:
        pickle.dump(index, stream, protocol=pickle.HIGHEST_PROTOCOL)


def write_subset(cache_dir: Path, staged_cache: Path, database_paths, selected):
    import lmdb

    selected_keys = {key for _, key in selected}
    for source_path in database_paths:
        source_env = lmdb.open(
            str(source_path), subdir=False, readonly=True, lock=False, readahead=False
        )
        destination_path = staged_cache / source_path.name
        destination_env = lmdb.open(
            str(destination_path), subdir=False, map_size=1 << 40
        )
        copied = 0
        try:
            with source_env.begin() as source_txn:
                output_txn = destination_env.begin(write=True)
                for key, value in source_txn.cursor():
                    if key not in selected_keys:
                        continue
                    if not output_txn.put(key, value, overwrite=False):
                        raise RuntimeError(f"Duplicate key while copying {source_path.name}: {key!r}")
                    copied += 1
                    if copied % 5000 == 0:
                        output_txn.commit()
                        output_txn = destination_env.begin(write=True)
                output_txn.commit()
        finally:
            source_env.close()
            destination_env.sync()
            destination_env.close()

        if source_path.name in REQUIRED_DATABASES and copied != len(selected):
            raise RuntimeError(
                f"{source_path.name} contained {copied} of {len(selected)} selected records."
            )

    trim_indices(
        cache_dir / INDEX_NAME,
        staged_cache / INDEX_NAME,
        selected,
    )


def cache_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="Cache directory or its parent (the latter contains rf3dpose_all/).",
    )
    parser.add_argument(
        "--destination",
        required=True,
        help="Output cache parent; it will contain rf3dpose_all/.",
    )
    parser.add_argument("--target-gib", type=float, default=10.0)
    parser.add_argument(
        "--select-people",
        nargs="+",
        type=parse_person_id,
        metavar="ID",
        help="Subject IDs to include, e.g. p1 p3 or 1 3. Default: all subjects.",
    )
    parser.add_argument(
        "--select-actions",
        nargs="+",
        type=parse_action_id,
        metavar="ID",
        help=(
            "Action IDs to include for the selected subjects, e.g. 2 7. "
            "Default: all actions."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the destination only after the new subset has been verified.",
    )
    args = parser.parse_args()
    if args.target_gib <= 0:
        parser.error("--target-gib must be greater than zero.")

    source = resolve_cache_path(Path(args.source).expanduser().resolve())
    destination = Path(args.destination).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Cache directory does not exist: {source}")
    output_cache = destination / "rf3dpose_all"
    if (
        destination == source
        or destination in source.parents
        or source in destination.parents
    ):
        raise SystemExit("Destination must not be the source cache or its parent.")
    if output_cache == source or output_cache in source.parents:
        raise SystemExit("Output cache must not replace the source cache.")

    if destination.exists():
        if not args.overwrite and any(destination.iterdir()):
            raise SystemExit(f"Destination is not empty: {destination}. Use --overwrite.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    budget = int(args.target_gib * 1024**3)
    selected, estimated_size, databases, action_count = selected_actions(
        source,
        people=set(args.select_people) if args.select_people else None,
        actions=set(args.select_actions) if args.select_actions else None,
    )
    if estimated_size > budget:
        raise SystemExit(
            f"The {action_count} selected complete actions need approximately "
            f"{estimated_size / 1024**3:.3f} GiB of record data, exceeding the "
            f"{args.target_gib:.3f} GiB limit. Select fewer subjects/actions or "
            "increase --target-gib; frames are never dropped from a selected action."
        )

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    ) as temp_dir:
        staged_destination = Path(temp_dir) / destination.name
        staged_cache = staged_destination / "rf3dpose_all"
        staged_cache.mkdir(parents=True)
        write_subset(source, staged_cache, databases, selected)
        readme_path = staged_destination / "README.txt"
        readme_path.write_text(
            f"""RF3DPose processed-cache subset

Source cache: {source}
Size: pending verification
Limit: {args.target_gib:.6f} GiB
Actions: {action_count:,}
Frames: {len(selected):,}

Use this directory as cache_dir when creating RF3DPoseDataset. It is a
processed LMDB subset, not a raw pX/actions dataset.
"""
        )
        actual_size = cache_size(staged_destination)
        if actual_size > budget:
            raise SystemExit(
                f"The selected complete actions need {actual_size / 1024**3:.3f} "
                f"GiB on disk, exceeding the {args.target_gib:.3f} GiB limit. "
                "Select fewer subjects/actions or increase --target-gib; "
                "frames are never dropped from a selected action."
            )

        readme_path.write_text(
            f"""RF3DPose processed-cache subset

Source cache: {source}
Size: {actual_size:,} bytes ({actual_size / 1024**3:.6f} GiB)
Limit: {args.target_gib:.6f} GiB
Actions: {action_count:,}
Frames: {len(selected):,}

Use this directory as cache_dir when creating RF3DPoseDataset. It is a
processed LMDB subset, not a raw pX/actions dataset.
"""
        )
        actual_size = cache_size(staged_destination)
        if actual_size > budget:
            raise SystemExit(
                f"The selected complete actions and metadata need "
                f"{actual_size / 1024**3:.3f} GiB on disk, exceeding the "
                f"{args.target_gib:.3f} GiB limit. Select fewer subjects/actions "
                "or increase --target-gib."
            )

        if destination.exists():
            if any(destination.iterdir()) and not args.overwrite:
                raise SystemExit(
                    f"Destination is not empty: {destination}. Use --overwrite."
                )
            shutil.rmtree(destination)
        shutil.move(str(staged_destination), str(destination))

    print("DONE")
    print(f"Subset: {destination}")
    print(f"Size:   {actual_size / 1024**3:.6f} GiB (limit {args.target_gib:.6f} GiB)")
    print(f"Actions: {action_count}")
    print(f"Frames: {len(selected):,}")
    print(f"Estimated data bytes: {estimated_size:,}")


if __name__ == "__main__":
    main()
