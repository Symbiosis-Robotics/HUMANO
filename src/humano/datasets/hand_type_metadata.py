from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from humano.robots.hands import get_hand_spec

HAND_TYPE_FEATURE_NAME = "hand_type"
HAND_TYPE_FEATURE_DTYPE = "string"
HAND_TYPE_FEATURE_SHAPE = (1,)

_HAND_TYPE_BY_HAND_DOF = {6: {"inspire", "brainco"}, 7: {"dex3"}}
_HAND_TYPE_BY_TOTAL_DOF = {29 + 2 * dof: names for dof, names in _HAND_TYPE_BY_HAND_DOF.items()}


class HandTypeMetadataError(ValueError):
    """Raised when a dataset's hand type cannot be determined safely."""


@dataclass(frozen=True)
class BackfillResult:
    dataset_root: Path
    hand_type: str
    parquet_files: int
    updated_parquet_files: int
    updated_info: bool


def hand_type_feature(*, json_compatible: bool = False) -> dict[str, Any]:
    shape: list[int] | tuple[int, ...]
    shape = list(HAND_TYPE_FEATURE_SHAPE) if json_compatible else HAND_TYPE_FEATURE_SHAPE
    return {"dtype": HAND_TYPE_FEATURE_DTYPE, "shape": shape}


def _feature_width(features: Mapping[str, Any], key: str) -> int | None:
    feature = features.get(key)
    if not isinstance(feature, Mapping):
        return None
    shape = feature.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 1:
        return None
    width = shape[0]
    return int(width) if isinstance(width, int) else None


def infer_hand_type_from_features(
    features: Mapping[str, Any], *, hand_type: str | None = None
) -> str:
    signals: list[tuple[str, set[str]]] = []
    for key in ("teleop.left_hand_joints", "teleop.right_hand_joints"):
        width = _feature_width(features, key)
        if width in _HAND_TYPE_BY_HAND_DOF:
            signals.append((key, _HAND_TYPE_BY_HAND_DOF[width]))
    for key in ("observation.state", "action.wbc", "sim.robot_ctrl"):
        width = _feature_width(features, key)
        if width in _HAND_TYPE_BY_TOTAL_DOF:
            signals.append((key, _HAND_TYPE_BY_TOTAL_DOF[width]))

    if not signals:
        raise HandTypeMetadataError(
            "could not infer hand type from state/action or hand feature shapes"
        )
    inferred = set.intersection(*(names for _, names in signals))
    if hand_type is not None:
        inferred &= {hand_type}
    if not inferred:
        details = ", ".join(f"{key}={sorted(names)}" for key, names in signals)
        raise HandTypeMetadataError(f"conflicting hand type signals: {details}")
    if len(inferred) != 1:
        raise HandTypeMetadataError(
            "ambiguous six-joint hand: specify hand_type=inspire or brainco; "
            "feature width alone cannot distinguish them"
        )
    return inferred.pop()


def _validate_registered_feature(features: Mapping[str, Any]) -> bool:
    feature = features.get(HAND_TYPE_FEATURE_NAME)
    if feature is None:
        return False
    expected = hand_type_feature(json_compatible=True)
    if feature != expected:
        raise HandTypeMetadataError(
            f"invalid {HAND_TYPE_FEATURE_NAME} feature: expected {expected}, got {feature}"
        )
    return True


def _read_info(info_path: Path) -> dict[str, Any]:
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HandTypeMetadataError(f"failed to read {info_path}: {exc}") from exc
    features = info.get("features")
    if not isinstance(features, dict):
        raise HandTypeMetadataError(f"{info_path} does not contain a features object")
    return info


def _parquet_hand_type(path: Path) -> tuple[bool, str | None]:
    schema = pq.read_schema(path)
    if HAND_TYPE_FEATURE_NAME not in schema.names:
        return False, None
    if schema.field(HAND_TYPE_FEATURE_NAME).type != pa.string():
        raise HandTypeMetadataError(f"{path} has non-string {HAND_TYPE_FEATURE_NAME} column")
    hand_type_column = pq.read_table(path, columns=[HAND_TYPE_FEATURE_NAME])[HAND_TYPE_FEATURE_NAME]
    recorded_values = hand_type_column.to_pylist()
    if any(value is None for value in recorded_values):
        raise HandTypeMetadataError(f"{path} contains null {HAND_TYPE_FEATURE_NAME} values")
    values = set(recorded_values)
    if len(values) > 1:
        raise HandTypeMetadataError(f"{path} contains multiple hand types: {sorted(values)}")
    return True, next(iter(values), None)


def _dataset_recorded_hand_type(dataset_root: Path) -> str | None:
    """Read explicit labels from saved rows and episode robot configurations."""
    values = set()
    for path in dataset_root.rglob("*.parquet"):
        _, value = _parquet_hand_type(path)
        if value is not None:
            values.add(value)
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        for line in episodes_path.read_text().splitlines():
            config = json.loads(line).get("environment_config")
            if config is None:
                continue
            config = json.loads(config) if isinstance(config, str) else config
            value = config.get("robot_cfg", {}).get("hand_type")
            if value is not None:
                values.add(value)
    if len(values) > 1:
        raise HandTypeMetadataError(f"conflicting recorded hand types: {sorted(values)}")
    return next(iter(values), None)


def validate_hand_assembly(robot_cfg: Mapping[str, Any], hand_type: str) -> None:
    expected = get_hand_spec(hand_type).assembly_id
    if expected is not None and robot_cfg.get("hand_assembly") != expected:
        raise HandTypeMetadataError(
            f"Recorded {hand_type} hand_assembly={robot_cfg.get('hand_assembly')!r} "
            f"does not match {expected!r}; use data recorded with this assembly"
        )


def _updated_parquet_metadata(
    metadata: Mapping[bytes, bytes] | None,
) -> dict[bytes, bytes] | None:
    if not metadata:
        return None
    updated = dict(metadata)
    if b"pandas" in updated:
        pandas_metadata = json.loads(updated[b"pandas"])
        columns = pandas_metadata.setdefault("columns", [])
        if not any(column.get("field_name") == HAND_TYPE_FEATURE_NAME for column in columns):
            columns.append(
                {
                    "name": HAND_TYPE_FEATURE_NAME,
                    "field_name": HAND_TYPE_FEATURE_NAME,
                    "pandas_type": "unicode",
                    "numpy_type": "object",
                    "metadata": None,
                }
            )
        updated[b"pandas"] = json.dumps(pandas_metadata).encode("utf-8")
    if b"huggingface" in updated:
        huggingface_metadata = json.loads(updated[b"huggingface"])
        features = huggingface_metadata.setdefault("info", {}).setdefault("features", {})
        features[HAND_TYPE_FEATURE_NAME] = {
            "dtype": HAND_TYPE_FEATURE_DTYPE,
            "_type": "Value",
        }
        updated[b"huggingface"] = json.dumps(huggingface_metadata).encode("utf-8")
    return updated


def _backup_file(path: Path, *, repository_root: Path, backup_root: Path) -> None:
    try:
        relative_path = path.resolve().relative_to(repository_root.resolve())
    except ValueError as exc:
        raise HandTypeMetadataError(f"{path} is outside repository root {repository_root}") from exc
    backup_path = backup_root / relative_path
    if backup_path.exists():
        raise FileExistsError(f"backup already exists: {backup_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(path, backup_path)
    except OSError:
        shutil.copy2(path, backup_path)


def _write_info(info_path: Path, info: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=info_path.parent,
        prefix=f".{info_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        json.dump(info, stream, indent=4)
        stream.write("\n")
    try:
        shutil.copystat(info_path, temporary_path)
        os.replace(temporary_path, info_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_parquet(path: Path, hand_type: str) -> None:
    original = pq.read_table(path)
    hand_types = pa.array([hand_type] * original.num_rows, type=pa.string())
    updated = original.append_column(HAND_TYPE_FEATURE_NAME, hand_types)
    updated = updated.replace_schema_metadata(_updated_parquet_metadata(original.schema.metadata))
    parquet_file = pq.ParquetFile(path)
    row_group_size = max(
        (
            parquet_file.metadata.row_group(index).num_rows
            for index in range(parquet_file.num_row_groups)
        ),
        default=None,
    )
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
    try:
        pq.write_table(
            updated,
            temporary_path,
            compression="snappy",
            row_group_size=row_group_size,
        )
        reloaded = pq.read_table(temporary_path)
        if reloaded.num_rows != original.num_rows:
            raise HandTypeMetadataError(f"row count changed while updating {path}")
        if not reloaded.select(original.column_names).equals(original):
            raise HandTypeMetadataError(f"existing parquet values changed while updating {path}")
        if set(reloaded[HAND_TYPE_FEATURE_NAME].to_pylist()) != {hand_type}:
            raise HandTypeMetadataError(f"failed to verify {HAND_TYPE_FEATURE_NAME} in {path}")
        shutil.copystat(path, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def backfill_dataset(
    dataset_root: Path,
    *,
    repository_root: Path,
    backup_root: Path,
    write: bool,
    hand_type: str | None = None,
) -> BackfillResult:
    info_path = dataset_root / "meta" / "info.json"
    info = _read_info(info_path)
    features = info["features"]
    recorded = _dataset_recorded_hand_type(dataset_root)
    if hand_type is not None and recorded is not None and hand_type != recorded:
        raise HandTypeMetadataError(f"recorded hand_type={recorded}, requested {hand_type}")
    hand_type = infer_hand_type_from_features(features, hand_type=hand_type or recorded)
    info_registered = _validate_registered_feature(features)
    parquet_files = tuple(sorted(dataset_root.rglob("*.parquet")))
    parquet_to_update: list[Path] = []
    for parquet_path in parquet_files:
        has_hand_type, recorded_hand_type = _parquet_hand_type(parquet_path)
        if not has_hand_type:
            parquet_to_update.append(parquet_path)
        elif recorded_hand_type not in (None, hand_type):
            raise HandTypeMetadataError(
                f"{parquet_path} contains {recorded_hand_type}, expected {hand_type}"
            )

    info_needs_update = not info_registered
    if write and (info_needs_update or parquet_to_update):
        backup_paths = ([info_path] if info_needs_update else []) + parquet_to_update
        for path in backup_paths:
            _backup_file(path, repository_root=repository_root, backup_root=backup_root)
        for parquet_path in parquet_to_update:
            _write_parquet(parquet_path, hand_type)
        if info_needs_update:
            updated_info = dict(info)
            updated_features = dict(features)
            updated_features[HAND_TYPE_FEATURE_NAME] = hand_type_feature(json_compatible=True)
            updated_info["features"] = updated_features
            _write_info(info_path, updated_info)

    return BackfillResult(
        dataset_root=dataset_root,
        hand_type=hand_type,
        parquet_files=len(parquet_files),
        updated_parquet_files=len(parquet_to_update),
        updated_info=info_needs_update,
    )
