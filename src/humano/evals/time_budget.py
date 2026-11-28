"""Load task evaluation limits and generate training-based budgets for custom suites."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from humano.evals.api import DEFAULT_MAX_EPISODE_STEPS

TIME_BUDGET_MARGIN = 1.1


def _episode_seconds(path: Path, fps: float, length: int) -> float:
    import pyarrow.parquet as pq

    if "sim.time" not in pq.read_schema(path).names:
        raise ValueError(f"Missing sim.time in training episode: {path}")
    times = pq.read_table(path, columns=["sim.time"])["sim.time"].to_pylist()
    if len(times) != length or not times:
        raise ValueError(f"Training episode length mismatch or empty: {path}")
    if not all(math.isfinite(value) for value in times):
        raise ValueError(f"Non-finite sim.time in training episode: {path}")
    if any(right <= left for left, right in zip(times, times[1:], strict=False)):
        raise ValueError(f"Non-increasing sim.time in training episode: {path}")
    return times[-1] - times[0] + 1.0 / fps


def _training_task_budget(dataset: Path) -> dict:
    info = json.loads((dataset / "meta/info.json").read_text())
    fps = float(info["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Training FPS must be positive and finite: {dataset}")
    split = info["splits"]["train"]
    start, stop = (int(value) for value in split.split(":"))
    discarded = set(info.get("discarded_episode_indices", []))
    candidates = []
    for line in (dataset / "meta/episodes.jsonl").read_text().splitlines():
        episode = json.loads(line)
        index = episode["episode_index"]
        if not start <= index < stop or index in discarded:
            continue
        if episode.get("success") is False or episode.get("partial_reason"):
            continue
        path = dataset / info["data_path"].format(
            episode_chunk=index // info["chunks_size"], episode_index=index
        )
        candidates.append((_episode_seconds(path, fps, episode["length"]), index))
    if not candidates:
        raise ValueError(f"No valid curated training episodes: {dataset}")
    seconds, index = max(candidates)
    return {
        "max_episode_seconds": seconds * TIME_BUDGET_MARGIN,
        "longest_training_episode_seconds": seconds,
        "longest_training_episode_index": index,
        "training_episode_count": len(candidates),
        "training_fps": fps,
        "training_split": split,
    }


def freeze_suite_time_budgets(train_root: Path, suite_config: Path) -> dict:
    """Write a frozen manifest into a suite; runtime never re-reads training data."""
    payload = json.loads(suite_config.read_text())
    tasks = {
        task: _training_task_budget(train_root / suite_config.stem / task)
        for task in payload["tasks"]
    }
    frozen = {
        "schema_version": 1,
        "margin": TIME_BUDGET_MARGIN,
        "duration_source": "sim.time_last_minus_first_plus_frame_period",
        "episode_selection": (
            "curated train split; exclude success=false, partial_reason, discarded_episode_indices"
        ),
        "trim_status": "no start/end labels; full retained demonstration",
        "tasks": tasks,
    }
    result = {**payload, "evaluation_time_budget": frozen}
    suite_config.write_text(json.dumps(result, indent=2) + "\n")
    return result


def load_task_time_budget(
    env_id: str, data_dir: Path, suite_config: Path | None = None
) -> dict | None:
    """Require frozen limits for configured suites, leaving custom data unchanged."""
    if suite_config is None:
        suite_config = Path(__file__).resolve().parents[3] / "configs/suites"
        suite_config = suite_config / f"{data_dir.parent.name}.json"
        if not suite_config.exists():
            return None
    payload = json.loads(suite_config.read_text())
    task = env_id.removeprefix("humano/").removesuffix("-v0")
    frozen = payload.get("evaluation_time_budget")
    if frozen is None or task not in frozen["tasks"]:
        raise ValueError(f"Missing frozen evaluation_time_budget for {task}: {suite_config}")
    record = {key: value for key, value in frozen.items() if key != "tasks"}
    task_budget = {**record, **frozen["tasks"][task], "suite_config": str(suite_config.resolve())}
    if "max_steps_200hz" in task_budget:
        return {**task_budget, "max_episode_seconds": task_budget["max_steps_200hz"] / 200}
    return task_budget


def episode_step_limit(
    max_episode_steps: int | None, time_budget: dict | None, step_seconds: float
) -> int:
    """Translate physical seconds to executed env.step calls; explicit steps win."""
    if max_episode_steps is not None:
        if type(max_episode_steps) is not int or max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be a positive integer")
        return max_episode_steps
    if time_budget is None:
        return DEFAULT_MAX_EPISODE_STEPS
    if not math.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("step_seconds must be positive and finite")
    seconds = float(time_budget["max_episode_seconds"])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("max_episode_seconds must be positive and finite")
    ratio = seconds / step_seconds
    nearest = round(ratio)
    if math.isclose(ratio, nearest, rel_tol=1e-12, abs_tol=1e-9):
        ratio = nearest
    return math.ceil(ratio)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--suite-config", type=Path, required=True)
    args = parser.parse_args()
    freeze_suite_time_budgets(args.train_root, args.suite_config)


if __name__ == "__main__":
    main()
