"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

import copy
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.utils import serialize_dict
from transforms3d.quaternions import qmult

_PSI0_OFFICE_CHAIR_TASK_UID = "g1_wholebody_push_office_chair_teleop"
_OFFICE_CHAIR_SPATIAL_KEY = "articulated_0"
_OFFICE_CHAIR_TABLE_FACING_CORRECTION = np.array(
    [np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)], dtype=np.float64
)


def _orient_office_chair_toward_table(env_conf: dict[str, Any]) -> dict[str, Any]:
    if env_conf.get("uid") != _PSI0_OFFICE_CHAIR_TASK_UID:
        return env_conf

    oriented = copy.deepcopy(env_conf)
    try:
        chair_state = oriented["dr_state_dict"]["spatial"][
            _OFFICE_CHAIR_SPATIAL_KEY
        ]
        source_orientation = np.asarray(chair_state["quaternion"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Psi0 office-chair environment config is missing a valid articulated pose"
        ) from exc

    if source_orientation.shape != (4,) or np.linalg.norm(source_orientation) == 0:
        raise ValueError("Psi0 office-chair quaternion must contain four non-zero values")

    source_orientation = source_orientation / np.linalg.norm(source_orientation)
    table_facing_orientation = qmult(
        source_orientation,
        _OFFICE_CHAIR_TABLE_FACING_CORRECTION,
    )
    chair_state["quaternion"] = (
        table_facing_orientation / np.linalg.norm(table_facing_orientation)
    ).tolist()
    return oriented


def get_episode_lerobot(dataset, eps_idx, data_format=None):
    def _to_int(value):
        item = getattr(value, "item", None)
        if callable(item):
            return int(item())
        return int(value)

    from_idx = _to_int(dataset.episode_data_index["from"][eps_idx])
    to_idx = _to_int(dataset.episode_data_index["to"][eps_idx])
    episode = [dataset[i] for i in range(from_idx, to_idx)]

    env_conf = _orient_office_chair_toward_table(
        json.loads(dataset.meta.episodes[eps_idx]["environment_config"])
    )
    scene_state = env_conf["dr_state_dict"].get("scene")
    if scene_state is not None:
        scene_state["uid"] = scene_state["uid"].replace(
            "102344280", "scene3"
        )  # FIXME
    # import pickle; pickle.dump(env_conf, open(f"env_conf_{eps_idx}.pkl", "wb"))
    # import pickle; env_conf = pickle.loads(open(f"env_conf_{eps_idx}.pkl", "rb").read())
    return env_conf, episode


@dataclass(frozen=True)
class PostprocessDatasetConfig:
    dataset_paths: tuple[Path, ...]
    output_path: Path
    remove_discarded: bool = True
    remove_stale_smpl: bool = True


@dataclass(frozen=True)
class PostprocessSummary:
    output_path: Path
    input_datasets: int
    kept_episodes: int
    discarded_episodes: int
    removed_frames: int


@dataclass(frozen=True)
class _DatasetSource:
    root: Path
    info: dict[str, Any]
    modality: dict[str, Any]
    tasks: dict[int, str]
    episodes: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _ProcessedEpisode:
    source: _DatasetSource
    metadata: dict[str, Any]
    dataframe: pd.DataFrame
    source_episode_index: int
    valid_indices: tuple[int, ...] | None
    was_discarded: bool


def build_stale_mask(smpl_poses: np.ndarray) -> np.ndarray:
    """Mark zero SMPL frames and the repeated frames immediately before them."""
    if smpl_poses.ndim != 2 or len(smpl_poses) == 0:
        raise ValueError("SMPL poses must be a non-empty two-dimensional array")
    zero_frames = np.all(smpl_poses == 0, axis=1)
    remove = zero_frames.copy()
    for zero_index in np.flatnonzero(zero_frames):
        frame_index = int(zero_index) - 1
        while (
            frame_index > 0
            and not zero_frames[frame_index]
            and np.array_equal(smpl_poses[frame_index], smpl_poses[frame_index - 1])
        ):
            remove[frame_index] = True
            frame_index -= 1
    return remove


def read_dataset_list(dataset_list: Path) -> tuple[Path, ...]:
    if not dataset_list.is_file():
        raise ValueError(f"Dataset list does not exist: {dataset_list}")
    paths = []
    for line in dataset_list.read_text(encoding="utf-8").splitlines():
        value = line.split("#", maxsplit=1)[0].strip()
        if value:
            paths.append(Path(value).expanduser())
    if not paths:
        raise ValueError(f"Dataset list contains no paths: {dataset_list}")
    return tuple(paths)


def postprocess_datasets(config: PostprocessDatasetConfig) -> PostprocessSummary:
    return _postprocess_datasets(config, frozenset())


def postprocess_dataset_with_discarded_episodes(
    config: PostprocessDatasetConfig,
    discarded_episode_indices: tuple[int, ...],
) -> PostprocessSummary:
    if len(config.dataset_paths) != 1:
        raise ValueError("Explicit discards require exactly one dataset path")
    return _postprocess_datasets(config, frozenset(discarded_episode_indices))


def _postprocess_datasets(
    config: PostprocessDatasetConfig,
    explicit_discarded_episode_indices: frozenset[int],
) -> PostprocessSummary:
    dataset_paths = tuple(path.expanduser().resolve() for path in config.dataset_paths)
    output_path = config.output_path.expanduser().resolve()
    _validate_output_path(dataset_paths, output_path)
    sources = tuple(_load_dataset(path) for path in dataset_paths)
    _validate_compatible_sources(sources)
    processed, discarded_episodes, removed_frames = _postprocess_sources(
        sources, config, explicit_discarded_episode_indices
    )
    if not processed:
        raise ValueError("No valid episodes remain after postprocessing")
    _write_output_dataset(output_path, sources[0], processed, config.remove_discarded)
    return PostprocessSummary(
        output_path=output_path,
        input_datasets=len(sources),
        kept_episodes=len(processed),
        discarded_episodes=discarded_episodes,
        removed_frames=removed_frames,
    )


def _validate_output_path(dataset_paths: tuple[Path, ...], output_path: Path) -> None:
    if not dataset_paths:
        raise ValueError("At least one dataset path is required")
    if output_path.exists():
        raise ValueError(f"Output path already exists: {output_path}")
    for dataset_path in dataset_paths:
        if not dataset_path.is_dir():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        if output_path == dataset_path or dataset_path in output_path.parents:
            raise ValueError("Output path must not be inside an input dataset")


def _load_dataset(root: Path) -> _DatasetSource:
    metadata_dir = root / "meta"
    required_paths = (
        metadata_dir / "info.json",
        metadata_dir / "modality.json",
        metadata_dir / "tasks.jsonl",
        metadata_dir / "episodes.jsonl",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise ValueError(f"Dataset metadata is incomplete: {', '.join(missing)}")
    info = _read_json(metadata_dir / "info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError(
            f"Unsupported LeRobot version in {root}: {info.get('codebase_version')!r}"
        )
    tasks = _load_tasks(metadata_dir / "tasks.jsonl")
    episodes = tuple(_read_jsonl(metadata_dir / "episodes.jsonl"))
    if not episodes:
        raise ValueError(f"Dataset has no episodes: {root}")
    return _DatasetSource(
        root=root,
        info=info,
        modality=_read_json(metadata_dir / "modality.json"),
        tasks=tasks,
        episodes=episodes,
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in {path}")
    return rows


def _load_tasks(path: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    for row in _read_jsonl(path):
        task_index = row.get("task_index")
        task = row.get("task")
        if not isinstance(task_index, int) or not isinstance(task, str):
            raise ValueError(f"Invalid task metadata in {path}: {row}")
        if task_index in tasks and tasks[task_index] != task:
            raise ValueError(f"Conflicting task index {task_index} in {path}")
        tasks[task_index] = task
    return tasks


def _validate_compatible_sources(sources: tuple[_DatasetSource, ...]) -> None:
    reference = sources[0]
    for source in sources[1:]:
        _validate_matching_value(reference, source, "robot_type", "robot type")
        _validate_matching_value(reference, source, "fps", "FPS")
        _validate_matching_value(reference, source, "features", "feature schema")
        _validate_matching_value(reference, source, "script_config", "script_config")
        if _canonical_json(reference.modality) != _canonical_json(source.modality):
            raise ValueError(
                f"modality metadata mismatch: {reference.root} != {source.root}"
            )


def _validate_matching_value(
    reference: _DatasetSource, source: _DatasetSource, key: str, label: str
) -> None:
    if _canonical_json(reference.info.get(key)) != _canonical_json(
        source.info.get(key)
    ):
        raise ValueError(f"{label} mismatch: {reference.root} != {source.root}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _postprocess_sources(
    sources: tuple[_DatasetSource, ...],
    config: PostprocessDatasetConfig,
    explicit_discarded_episode_indices: frozenset[int],
) -> tuple[list[_ProcessedEpisode], int, int]:
    processed: list[_ProcessedEpisode] = []
    discarded_episodes = 0
    removed_frames = 0
    for source in sources:
        discarded = set(source.info.get("discarded_episode_indices", [])) | set(
            explicit_discarded_episode_indices
        )
        for metadata in source.episodes:
            episode, removed = _postprocess_episode(source, metadata, config, discarded)
            removed_frames += removed
            if episode is None:
                discarded_episodes += 1
            else:
                processed.append(episode)
    return processed, discarded_episodes, removed_frames


def _postprocess_episode(
    source: _DatasetSource,
    metadata: dict[str, Any],
    config: PostprocessDatasetConfig,
    discarded: set[int],
) -> tuple[_ProcessedEpisode | None, int]:
    episode_index = metadata.get("episode_index")
    if not isinstance(episode_index, int):
        raise ValueError(f"Invalid episode metadata in {source.root}: {metadata}")
    was_discarded = episode_index in discarded
    if was_discarded and config.remove_discarded:
        return None, 0
    dataframe = _load_episode_dataframe(source, episode_index, metadata)
    _validate_episode_videos(source, episode_index, len(dataframe))
    valid_indices: tuple[int, ...] | None = None
    removed_frames = 0
    if config.remove_stale_smpl and "teleop.smpl_pose" in dataframe:
        poses = np.vstack(
            [
                np.asarray(value, dtype=np.float32)
                for value in dataframe["teleop.smpl_pose"]
            ]
        )
        stale_mask = build_stale_mask(poses)
        if stale_mask.all():
            stale_reason = (
                "all-zero SMPL data" if np.all(poses == 0) else "only stale SMPL data"
            )
            raise ValueError(
                f"Episode {episode_index} in {source.root} has {stale_reason}; "
                "rerun with --no-remove-stale-smpl"
            )
        if stale_mask.any():
            valid_indices = tuple(np.flatnonzero(~stale_mask).tolist())
            removed_frames = int(stale_mask.sum())
            dataframe = (
                dataframe.iloc[list(valid_indices)].copy().reset_index(drop=True)
            )
    return (
        _ProcessedEpisode(
            source=source,
            metadata=copy.deepcopy(metadata),
            dataframe=dataframe.copy(),
            source_episode_index=episode_index,
            valid_indices=valid_indices,
            was_discarded=was_discarded,
        ),
        removed_frames,
    )


def _load_episode_dataframe(
    source: _DatasetSource, episode_index: int, metadata: dict[str, Any]
) -> pd.DataFrame:
    parquet_path = _data_path(source.root, source.info, episode_index)
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Missing parquet for episode {episode_index}: {parquet_path}"
        )
    dataframe = pd.read_parquet(parquet_path)
    expected_length = metadata.get("length")
    if not isinstance(expected_length, int) or len(dataframe) != expected_length:
        raise ValueError(
            f"Episode metadata length does not match parquet: {parquet_path}"
        )
    return dataframe


def _validate_episode_videos(
    source: _DatasetSource, episode_index: int, frame_count: int
) -> None:
    for video_key in _video_keys(source.info):
        video_path = _video_path(source.root, source.info, episode_index, video_key)
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Missing video for episode {episode_index}: {video_path}"
            )
        if _video_frame_count(video_path) != frame_count:
            raise ValueError(f"Video frame count does not match parquet: {video_path}")


def _data_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    pattern = info.get("data_path")
    if not isinstance(pattern, str):
        raise ValueError("Dataset info is missing data_path")
    return root / pattern.format(
        episode_chunk=episode_index // int(info.get("chunks_size", 1000)),
        episode_index=episode_index,
    )


def _video_path(
    root: Path, info: dict[str, Any], episode_index: int, video_key: str
) -> Path:
    pattern = info.get("video_path")
    if not isinstance(pattern, str):
        raise ValueError("Dataset info is missing video_path")
    return root / pattern.format(
        episode_chunk=episode_index // int(info.get("chunks_size", 1000)),
        episode_index=episode_index,
        video_key=video_key,
    )


def _write_output_dataset(
    output_path: Path,
    reference: _DatasetSource,
    processed: list[_ProcessedEpisode],
    remove_discarded: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_path.parent, prefix=f".{output_path.name}."
    ) as temp_dir:
        staging = Path(temp_dir) / output_path.name
        staging.mkdir()
        _write_processed_episodes(staging, reference, processed)
        _write_dataset_metadata(staging, reference, processed, remove_discarded)
        staging.rename(output_path)


def _write_processed_episodes(
    destination: Path, reference: _DatasetSource, processed: list[_ProcessedEpisode]
) -> None:
    task_indices = _merged_task_indices(processed)
    global_index = 0
    for episode_index, episode in enumerate(processed):
        dataframe = _rewritten_dataframe(
            episode.dataframe,
            episode.source.tasks,
            task_indices,
            episode_index,
            global_index,
            int(reference.info["fps"]),
        )
        parquet_path = _data_path(destination, reference.info, episode_index)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(parquet_path)
        _write_episode_videos(destination, reference, episode, episode_index)
        global_index += len(dataframe)


def _merged_task_indices(processed: list[_ProcessedEpisode]) -> dict[str, int]:
    task_indices: dict[str, int] = {}
    for episode in processed:
        for task_index in sorted(episode.source.tasks):
            task = episode.source.tasks[task_index]
            if task not in task_indices:
                task_indices[task] = len(task_indices)
    return task_indices


def _rewritten_dataframe(
    dataframe: pd.DataFrame,
    source_tasks: dict[int, str],
    merged_tasks: dict[str, int],
    episode_index: int,
    global_index: int,
    fps: int,
) -> pd.DataFrame:
    if "task_index" not in dataframe:
        raise ValueError("Episode parquet is missing task_index")
    task_values = dataframe["task_index"].to_numpy()
    remapped = []
    for task_value in task_values:
        source_task = source_tasks.get(int(task_value))
        if source_task is None:
            raise ValueError(f"Unknown task index in episode parquet: {task_value}")
        remapped.append(merged_tasks[source_task])
    frame_count = len(dataframe)
    return dataframe.assign(
        episode_index=np.full(frame_count, episode_index, dtype=np.int64),
        index=np.arange(global_index, global_index + frame_count, dtype=np.int64),
        frame_index=np.arange(frame_count, dtype=np.int64),
        timestamp=np.arange(frame_count, dtype=np.float32) / fps,
        task_index=np.asarray(remapped, dtype=np.int64),
    )


def _write_episode_videos(
    destination: Path,
    reference: _DatasetSource,
    episode: _ProcessedEpisode,
    episode_index: int,
) -> None:
    for video_key in _video_keys(reference.info):
        source_path = _video_path(
            episode.source.root,
            episode.source.info,
            episode.source_episode_index,
            video_key,
        )
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Missing video for episode {episode.source_episode_index}: {source_path}"
            )
        destination_path = _video_path(
            destination, reference.info, episode_index, video_key
        )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if episode.valid_indices is None:
            shutil.copy2(source_path, destination_path)
        else:
            _filter_video(
                source_path,
                destination_path,
                episode.valid_indices,
                int(episode.metadata["length"]),
                int(reference.info["fps"]),
            )


def _video_keys(info: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        key for key, value in info["features"].items() if value.get("dtype") == "video"
    )


def _filter_video(
    source_path: Path,
    destination_path: Path,
    valid_indices: tuple[int, ...],
    original_frame_count: int,
    fps: int,
) -> None:
    with av.open(str(source_path)) as source:
        frames = [frame.to_ndarray(format="rgb24") for frame in source.decode(video=0)]
    if len(frames) != original_frame_count:
        raise ValueError(f"Video frame count does not match parquet: {source_path}")
    selected = [frames[index] for index in valid_indices]
    with av.open(str(destination_path), mode="w") as destination:
        stream = destination.add_stream("libx264", rate=fps)
        stream.width = selected[0].shape[1]
        stream.height = selected[0].shape[0]
        stream.pix_fmt = "yuv420p"
        for image in selected:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                destination.mux(packet)
        for packet in stream.encode():
            destination.mux(packet)


def _video_frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def _write_dataset_metadata(
    destination: Path,
    reference: _DatasetSource,
    processed: list[_ProcessedEpisode],
    remove_discarded: bool,
) -> None:
    task_indices = _merged_task_indices(processed)
    episode_rows = []
    stats_rows = []
    global_index = 0
    for episode_index, episode in enumerate(processed):
        dataframe = _rewritten_dataframe(
            episode.dataframe,
            episode.source.tasks,
            task_indices,
            episode_index,
            global_index,
            int(reference.info["fps"]),
        )
        global_index += len(dataframe)
        metadata = copy.deepcopy(episode.metadata)
        metadata["episode_index"] = episode_index
        metadata["length"] = len(dataframe)
        episode_rows.append(metadata)
        stats_rows.append(
            {
                "episode_index": episode_index,
                "stats": _episode_stats(dataframe, reference.info),
            }
        )
    info = copy.deepcopy(reference.info)
    info.update(
        {
            "total_episodes": len(processed),
            "total_frames": global_index,
            "total_tasks": len(task_indices),
            "total_videos": len(processed) * len(_video_keys(reference.info)),
            "total_chunks": (len(processed) - 1) // int(reference.info["chunks_size"])
            + 1,
            "splits": {"train": f"0:{len(processed)}"},
        }
    )
    if remove_discarded:
        info.pop("discarded_episode_indices", None)
    else:
        info["discarded_episode_indices"] = [
            index for index, episode in enumerate(processed) if episode.was_discarded
        ]
    _write_json(destination / "meta" / "info.json", info)
    _write_json(destination / "meta" / "modality.json", reference.modality)
    _write_jsonl(
        destination / "meta" / "tasks.jsonl",
        [{"task_index": index, "task": task} for task, index in task_indices.items()],
    )
    _write_jsonl(destination / "meta" / "episodes.jsonl", episode_rows)
    _write_jsonl(destination / "meta" / "episodes_stats.jsonl", stats_rows)


def _episode_stats(dataframe: pd.DataFrame, info: dict[str, Any]) -> dict[str, Any]:
    features = {
        key: value
        for key, value in info["features"].items()
        if value.get("dtype") not in {"image", "video"}
    }
    episode_data = {}
    for key in features:
        if key not in dataframe:
            raise ValueError(
                f"Episode parquet is missing feature required by metadata: {key}"
            )
        values = dataframe[key].to_list()
        first = values[0]
        episode_data[key] = (
            np.stack(values)
            if isinstance(first, (list, np.ndarray))
            else np.asarray(values)
        )
    return serialize_dict(compute_episode_stats(episode_data, features))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=4), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
