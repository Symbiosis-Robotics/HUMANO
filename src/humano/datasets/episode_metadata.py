from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _load_lerobot_episodes(data_dir: Path) -> dict[int, Any]:
    parquet_files = sorted((data_dir / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir / 'data'}")

    episodes: dict[int, Any] = {}
    for parquet_file in parquet_files:
        df = pq.read_table(parquet_file).to_pandas()
        for ep_idx in df["episode_index"].unique():
            episodes[int(ep_idx)] = df[df["episode_index"] == ep_idx].reset_index(drop=True)
    return episodes


def _load_episode_configs(data_dir: Path) -> dict[int, dict[str, Any]]:
    meta_file = data_dir / "meta" / "episodes.jsonl"
    if not meta_file.is_file():
        return {}

    configs: dict[int, dict[str, Any]] = {}
    with meta_file.open("r") as f:
        for line in f:
            entry = json.loads(line)
            if "episode_index" not in entry or "environment_config" not in entry:
                continue
            configs[int(entry["episode_index"])] = json.loads(entry["environment_config"])
    return configs
