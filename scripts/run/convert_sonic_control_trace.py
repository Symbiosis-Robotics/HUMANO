"""Create self-contained training and control Parquets from raw SONIC data.

The raw dataset and its SONIC NPZ (or legacy JSONL) sidecars remain intact.
Visual frames are joined to their exact SONIC control tick, while a second
Parquet preserves every control tick. Videos and metadata are copied as regular
files, never symlinks.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRAINING_COLUMNS = (
    "observation.state",
    "action.wbc",
    "action.motion_token",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "sonic_run_id",
    "sonic_tick",
    "robot_state_seq",
)

TRACE_VECTOR_COLUMNS = {
    "encoder_obs": ("sonic.encoder_obs", 1762),
    "token_state": ("sonic.token_state", 64),
    "decoder_obs": ("sonic.decoder_obs", 994),
    "decoder_raw_action": ("sonic.decoder_raw_action", 29),
    "decoder_target_action": ("sonic.decoder_target_action", 29),
}

TRACE_FEATURES = {
    output_name: {"dtype": "float64", "shape": [width]}
    for output_name, width in TRACE_VECTOR_COLUMNS.values()
}


def _scalar_int(value: Any, *, column: str, location: str) -> int:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{location}: {column} must be scalar, got {value!r}")
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: {column} must be an integer, got {value!r}") from exc


def _trace_vector(row: Mapping[str, Any], key: str, width: int, *, location: str) -> list[float]:
    value = row.get(key)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != width:
        actual = len(value) if isinstance(value, (list, tuple)) else type(value).__name__
        raise ValueError(f"{location}: {key} must have length {width}, got {actual}")
    return [float(item) for item in value]


def _read_trace_rows(trace_path: Path) -> list[dict[str, Any]]:
    if not trace_path.is_file():
        raise FileNotFoundError(f"Missing SONIC trace sidecar: {trace_path}")

    if trace_path.suffix == ".npz":
        with np.load(trace_path, allow_pickle=False) as trace:
            schema = str(trace["schema"]) if "schema" in trace else "sonic_control_trace_v1"
            if schema != "sonic_control_trace_v1":
                raise ValueError(
                    f"{trace_path}: {schema} is a replay trace; this converter requires "
                    "original atomic SONIC control ticks"
                )
            count = len(trace["sonic_tick"])
            rows = []
            for row_index in range(count):
                row = {
                    key: int(trace[key][row_index])
                    for key in ("sonic_run_id", "sonic_tick", "robot_state_seq")
                }
                for key in TRACE_VECTOR_COLUMNS:
                    if bool(trace[f"{key}_valid"][row_index]):
                        row[key] = trace[key][row_index].tolist()
                    else:
                        row[key] = []
                rows.append(row)
        if not rows:
            raise ValueError(f"SONIC trace sidecar is empty: {trace_path}")
        return rows

    rows = []
    with trace_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{trace_path}:{line_number}: trace row must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"SONIC trace sidecar is empty: {trace_path}")
    return rows


def _validate_trace_rows(
    rows: list[dict[str, Any]], trace_path: Path
) -> dict[tuple[int, int], dict[str, Any]]:
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    previous_run_id: int | None = None
    previous_tick: int | None = None
    previous_state_seq: int | None = None

    for row_index, row in enumerate(rows):
        location = f"{trace_path}:row {row_index}"
        run_id = _scalar_int(row.get("sonic_run_id"), column="sonic_run_id", location=location)
        tick = _scalar_int(row.get("sonic_tick"), column="sonic_tick", location=location)
        state_seq = _scalar_int(
            row.get("robot_state_seq"), column="robot_state_seq", location=location
        )
        key = (run_id, tick)
        if key in lookup:
            raise ValueError(f"{location}: duplicate SONIC trace key {key}")
        lookup[key] = row

        for trace_key, (_, width) in TRACE_VECTOR_COLUMNS.items():
            _trace_vector(row, trace_key, width, location=location)

        if run_id == previous_run_id:
            if tick != previous_tick + 1:
                raise ValueError(
                    f"{location}: non-continuous sonic_tick: {previous_tick} -> {tick}"
                )
            if state_seq != previous_state_seq + 1:
                raise ValueError(
                    f"{location}: non-continuous robot_state_seq: "
                    f"{previous_state_seq} -> {state_seq}"
                )
        previous_run_id = run_id
        previous_tick = tick
        previous_state_seq = state_seq

    return lookup


def _merge_visual_frames(
    frame: pd.DataFrame,
    trace_lookup: Mapping[tuple[int, int], Mapping[str, Any]],
    parquet_path: Path,
) -> pd.DataFrame:
    merged = frame.loc[:, list(TRAINING_COLUMNS)].copy()
    merged_trace = {output_name: [] for output_name, _ in TRACE_VECTOR_COLUMNS.values()}

    for row_index, (raw_run_id, raw_tick, raw_state_seq) in enumerate(
        zip(frame["sonic_run_id"], frame["sonic_tick"], frame["robot_state_seq"], strict=True)
    ):
        location = f"{parquet_path}:row {row_index}"
        run_id = _scalar_int(raw_run_id, column="sonic_run_id", location=location)
        tick = _scalar_int(raw_tick, column="sonic_tick", location=location)
        state_seq = _scalar_int(raw_state_seq, column="robot_state_seq", location=location)
        trace_row = trace_lookup.get((run_id, tick))
        if trace_row is None:
            raise ValueError(f"{location}: has no matching SONIC trace row for {(run_id, tick)}")
        trace_state_seq = _scalar_int(
            trace_row.get("robot_state_seq"), column="robot_state_seq", location=location
        )
        if state_seq != trace_state_seq:
            raise ValueError(
                f"{location}: robot_state_seq {state_seq} does not match trace {trace_state_seq}"
            )

        for trace_key, (output_name, width) in TRACE_VECTOR_COLUMNS.items():
            merged_trace[output_name].append(
                _trace_vector(trace_row, trace_key, width, location=location)
            )

    for output_name, values in merged_trace.items():
        merged[output_name] = values
    # The training token must be the exact token embedded in the matched
    # decoder observation, not a separately sampled or backfilled value.
    merged["action.motion_token"] = merged["sonic.token_state"]
    return merged


def _update_feature_metadata(output_root: Path) -> None:
    info_path = output_root / "meta" / "info.json"
    if not info_path.is_file():
        return
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.setdefault("features", {})
    features.update(TRACE_FEATURES)
    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")


def convert_dataset(dataset_root: Path, output_root: Path, overwrite: bool = False) -> int:
    dataset_root = dataset_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root == dataset_root:
        raise ValueError("output_root must be different from dataset_root")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)

    parquet_files = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files under {dataset_root / 'data'}")

    episode_count = 0
    for source in parquet_files:
        frame = pd.read_parquet(source)
        missing = [column for column in TRAINING_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{source} is not a trace-aware dataset; missing {missing}")
        npz_path = dataset_root / "sonic" / f"{source.stem}.npz"
        trace_path = (
            npz_path
            if npz_path.is_file()
            else dataset_root / "sonic_control_trace" / f"{source.stem}.jsonl"
        )
        trace_rows = _read_trace_rows(trace_path)
        trace_lookup = _validate_trace_rows(trace_rows, trace_path)
        merged = _merge_visual_frames(frame, trace_lookup, source)

        destination = output_root / "data" / source.relative_to(dataset_root / "data")
        destination.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(destination, index=False)

        if trace_path.suffix == ".npz":
            control_destination = output_root / "sonic" / trace_path.name
            control_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trace_path, control_destination)
        else:
            control_destination = (
                output_root / "sonic_control_trace" / source.parent.name / f"{source.stem}.parquet"
            )
            control_destination.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_records(trace_rows).to_parquet(control_destination, index=False)
        episode_count += 1

    for directory in ("meta", "videos"):
        source = dataset_root / directory
        if source.is_dir():
            shutil.copytree(source, output_root / directory, symlinks=False)

    _update_feature_metadata(output_root)

    return episode_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    count = convert_dataset(args.dataset_root, args.output_root, overwrite=args.overwrite)
    print(f"Converted {count} episodes with exact SONIC traces to {args.output_root}")


if __name__ == "__main__":
    main()
