"""Evaluate every task listed in a HUMANO evaluation suite."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import typer

from humano.evals.api import DEFAULT_NUM_EPISODES, EvalConfig, resolve_wbc_controller
from humano.evals.metrics import summarize_episode_metrics, summarize_evaluation_health
from humano.evals.time_budget import episode_step_limit, load_task_time_budget


def _load_suite_payload(suite_config_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(suite_config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid suite JSON: {suite_config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Suite config must be an object: {suite_config_path}")
    return payload


def load_suite_tasks(suite_config_path: Path) -> tuple[str, ...]:
    """Load and validate the stable task membership of one suite."""
    payload = _load_suite_payload(suite_config_path)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Suite config needs a non-empty tasks list: {suite_config_path}")
    if not all(isinstance(task, str) and task for task in tasks):
        raise ValueError(f"Suite tasks must be non-empty strings: {suite_config_path}")
    if len(tasks) != len(set(tasks)):
        raise ValueError(f"Suite config contains duplicate tasks: {suite_config_path}")
    return tuple(tasks)


def load_suite_task_kwargs(suite_config_path: Path) -> dict[str, Any]:
    """Load constructor arguments shared by every task in a suite."""
    payload = _load_suite_payload(suite_config_path)
    task_kwargs = payload.get("task_kwargs", {})
    if not isinstance(task_kwargs, dict) or not all(
        isinstance(name, str) and name for name in task_kwargs
    ):
        raise ValueError(f"Suite task_kwargs must be an object: {suite_config_path}")
    unsupported = set(task_kwargs) - {"movement"}
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported suite task kwargs: {names}")
    if "movement" in task_kwargs and not isinstance(task_kwargs["movement"], bool):
        raise ValueError("Suite task_kwargs.movement must be a boolean")
    return dict(task_kwargs)


def _task_data_dir(data_root: Path, suite_name: str, task_name: str) -> Path:
    candidates = (data_root / suite_name / task_name, data_root / suite_name / f"{task_name}-v0")
    for data_dir in candidates:
        metadata = data_dir / "meta" / "info.json"
        if metadata.is_file():
            return data_dir
    expected = ", ".join(str(path / "meta" / "info.json") for path in candidates)
    raise FileNotFoundError(f"LeRobot metadata not found in: {expected}")


def _write_summary(eval_dir: Path, summary: Mapping[str, Any]) -> None:
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fieldnames = [
        "task",
        "env_id",
        "status",
        "successful_episodes",
        "episodes",
        "success_rate",
        "fall_rate",
        "timeout_rate",
        "median_success_completion_time_s",
        "median_success_dimensionless_jerk",
        "metrics_path",
        "evaluation_health_path",
        "log_path",
        "error",
    ]
    with (eval_dir / "summary.csv").open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["tasks"])


def _task_command(config: EvalConfig, run_options: Mapping[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "humano.cli.eval_task",
        config.env_id,
        config.policy,
        config.split,
    ]
    option_pairs = (
        ("--host", config.host),
        ("--port", config.port),
        ("--data-format", config.data_format),
        ("--sim-mode", config.sim_mode),
        ("--eval-dir", config.eval_dir),
        ("--max-episode-steps", config.max_episode_steps),
        ("--time-budget-file", config.time_budget_file),
        ("--num-episodes", config.num_episodes),
        ("--episode-start", config.episode_start),
        ("--data-dir", config.data_dir),
        ("--success-criteria", config.success_criteria),
        ("--num-workers", config.num_workers),
        ("--exocentric-camera", config.exocentric_camera),
        ("--exocentric-video-key", config.exocentric_video_key),
    )
    for flag, value in option_pairs:
        if value is not None:
            command.extend([flag, str(value)])
    command.extend(
        [
            "--headless" if config.headless else "--no-headless",
            "--save-video" if config.save_video else "--no-save-video",
            (
                "--save-exocentric-video"
                if config.save_exocentric_video
                else "--no-save-exocentric-video"
            ),
            "--async-video" if config.async_video else "--no-async-video",
            "--fast" if run_options.get("fast", False) else "--no-fast",
            "--movement" if run_options.get("movement", False) else "--no-movement",
            "--hand-type",
            str(run_options.get("hand_type", "dex3")),
        ]
    )
    if config.wbc_controller is not None:
        command.extend(["--wbc-controller", config.wbc_controller])
    for option_name in ("env_step_dt", "sonic_control_dt", "seed"):
        value = run_options.get(option_name)
        if value is not None:
            command.extend([f"--{option_name.replace('_', '-')}", str(value)])
    return command


def _read_task_stats(stats_path: Path) -> dict[str, bool]:
    if not stats_path.is_file():
        raise RuntimeError(f"Task eval did not write stats: {stats_path}")
    stats: dict[str, bool] = {}
    for line in stats_path.read_text().splitlines():
        key, separator, value = line.partition(":")
        if not separator or not key.startswith("episode_"):
            continue
        normalized = value.strip().lower()
        if normalized not in {"true", "false"}:
            raise ValueError(f"Invalid episode result in {stats_path}: {line}")
        stats[key] = normalized == "true"
    if not stats:
        raise RuntimeError(f"Task eval produced no episode results: {stats_path}")
    return stats


def _read_task_metrics(metrics_path: Path) -> dict[str, dict[str, Any]]:
    if not metrics_path.is_file():
        raise RuntimeError(f"Task eval did not write metrics: {metrics_path}")
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(metrics_path.read_text().splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid metric JSON at {metrics_path}:{line_number}") from exc
        if not isinstance(record, dict) or not isinstance(record.get("episode"), str):
            raise ValueError(f"Metric record needs an episode id at {metrics_path}:{line_number}")
        if "fall_detected" not in record and "fell" in record:
            record = {**record, "fall_detected": record["fell"]}
        required = {
            "success",
            "fall_detected",
            "timed_out",
            "termination_reason",
            "completion_time_s",
            "dimensionless_jerk",
        }
        numeric_values = (
            record.get("completion_time_s"),
            record.get("dimensionless_jerk"),
        )
        if (
            not required.issubset(record)
            or not isinstance(record["success"], bool)
            or not isinstance(record["fall_detected"], bool)
            or not isinstance(record["timed_out"], bool)
            or not isinstance(record["termination_reason"], str)
            or record["termination_reason"]
            not in {
                "success",
                "timeout",
                "fall_down",
                "wall_collision",
                "policy_exhausted",
                "ended_without_success",
            }
            or any(
                value is not None
                and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0.0
                )
                for value in numeric_values
            )
        ):
            raise ValueError(f"Invalid metric record at {metrics_path}:{line_number}")
        episode = record["episode"]
        if episode in records:
            raise ValueError(f"Duplicate metric record for {episode} in {metrics_path}")
        records = {
            **records,
            episode: {key: value for key, value in record.items() if key != "episode"},
        }
    if not records:
        raise RuntimeError(f"Task eval produced no metric records: {metrics_path}")
    return records


def _resumable_task_results(
    config: EvalConfig,
) -> tuple[dict[str, bool], dict[str, dict[str, Any]]] | None:
    try:
        stats = _read_task_stats(Path(config.eval_dir) / "eval_stats.txt")
        episode_metrics = _read_task_metrics(Path(config.eval_dir) / "eval_metrics.jsonl")
    except (RuntimeError, ValueError):
        return None
    expected = {
        f"episode_{index}"
        for index in range(config.episode_start, config.episode_start + config.num_episodes)
    }
    if set(stats) != expected or set(episode_metrics) != expected:
        return None
    time_budget = (
        load_task_time_budget(
            config.env_id,
            Path(config.data_dir),
            Path(config.time_budget_file) if config.time_budget_file else None,
        )
        if config.max_episode_steps is None
        else None
    )
    controller = resolve_wbc_controller(config.policy, config.wbc_controller)
    for record in episode_metrics.values():
        if record.get("wbc_controller", record.get("mt_controller", "sonic")) != controller:
            return None
        if controller in ("amo", "twist2"):
            key = "amo_weights_dir" if controller == "amo" else "twist2_model_path"
            weights = os.getenv("HUMANO_" + key.upper())
            if not weights or record.get(key) != str(Path(weights).expanduser().resolve()):
                return None
        step_seconds = record.get("env_step_seconds")
        if (
            not isinstance(step_seconds, (int, float))
            or not math.isfinite(step_seconds)
            or step_seconds <= 0
        ):
            return None
        if record.get("evaluation_time_budget") != time_budget:
            return None
        expected_cap = episode_step_limit(config.max_episode_steps, time_budget, step_seconds)
        if record.get("max_episode_steps") != expected_cap:
            return None
    return stats, episode_metrics


def _run_task_process(
    config: EvalConfig, run_options: Mapping[str, Any]
) -> tuple[dict[str, bool], dict[str, dict[str, Any]], str]:
    task_eval_dir = Path(config.eval_dir)
    task_eval_dir.mkdir(parents=True, exist_ok=True)
    stats_path = task_eval_dir / "eval_stats.txt"
    stats_path.unlink(missing_ok=True)
    metrics_path = task_eval_dir / "eval_metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    log_path = task_eval_dir / "eval_latest.log"
    child_env = os.environ.copy()
    child_env["HUMANO_LOG_FILE"] = str(log_path)
    completed = subprocess.run(_task_command(config, run_options), check=False, env=child_env)
    if completed.returncode != 0:
        raise RuntimeError(f"Task eval exited with code {completed.returncode}; see {log_path}")
    return _read_task_stats(stats_path), _read_task_metrics(metrics_path), str(log_path)


def evaluate_suite(
    config: EvalConfig,
    suite_config_path: Path,
    data_root: Path,
    *,
    run_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate suite tasks sequentially in isolated Isaac Sim processes."""
    task_names = load_suite_tasks(suite_config_path)
    suite_task_kwargs = load_suite_task_kwargs(suite_config_path)
    suite_name = suite_config_path.stem
    frozen_budget = _load_suite_payload(suite_config_path).get("evaluation_time_budget")
    task_data_dirs = {
        task_name: _task_data_dir(data_root, suite_name, task_name) for task_name in task_names
    }
    options = {**dict(run_options or {}), **suite_task_kwargs}
    eval_dir = Path(config.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    task_summaries: list[dict[str, Any]] = []
    suite_episode_metrics: dict[str, dict[str, Any]] = {}

    for task_name in task_names:
        env_id = f"humano/{task_name}-v0"
        task_config = replace(
            config,
            env_id=env_id,
            data_dir=str(task_data_dirs[task_name]),
            eval_dir=str(eval_dir / "tasks" / task_name),
            time_budget_file=str(suite_config_path.resolve())
            if frozen_budget
            else config.time_budget_file,
        )
        try:
            resumable = _resumable_task_results(task_config) if options.get("resume") else None
            if resumable is None:
                stats, episode_metrics, log_path = _run_task_process(task_config, options)
            else:
                stats, episode_metrics = resumable
                log_path = str(Path(task_config.eval_dir) / "eval_latest.log")
        except (OSError, RuntimeError, ValueError, typer.Exit) as exc:
            task_summaries.append(
                {
                    "task": task_name,
                    "env_id": env_id,
                    "status": "failed",
                    "successful_episodes": 0,
                    "episodes": 0,
                    "success_rate": None,
                    "fall_rate": None,
                    "timeout_rate": None,
                    "median_success_completion_time_s": None,
                    "median_success_dimensionless_jerk": None,
                    "metrics_path": "",
                    "evaluation_health_path": "",
                    "log_path": "",
                    "error": str(exc) or type(exc).__name__,
                }
            )
            continue
        metric_summary = summarize_episode_metrics(episode_metrics)
        task_summaries.append(
            {
                "task": task_name,
                "env_id": env_id,
                "status": "completed",
                "successful_episodes": sum(stats.values()),
                "episodes": len(stats),
                "success_rate": sum(stats.values()) / len(stats),
                **metric_summary,
                "metrics_path": str(Path(task_config.eval_dir) / "eval_metrics.jsonl"),
                "evaluation_health_path": str(
                    Path(task_config.eval_dir) / "evaluation_health.json"
                ),
                "log_path": log_path,
                "error": "",
            }
        )
        suite_episode_metrics = {
            **suite_episode_metrics,
            **{f"{task_name}/{episode}": record for episode, record in episode_metrics.items()},
        }

    completed_tasks = [item for item in task_summaries if item["status"] == "completed"]
    successful_episodes = sum(item["successful_episodes"] for item in completed_tasks)
    episodes = sum(item["episodes"] for item in completed_tasks)
    suite_metric_summary = summarize_episode_metrics(suite_episode_metrics)
    expected_episode_ids = tuple(
        f"{task_name}/episode_{episode_index}"
        for task_name in task_names
        for episode_index in range(config.episode_start, config.episode_start + config.num_episodes)
    )
    evaluation_health = {
        **summarize_evaluation_health(
            expected_episode_ids=expected_episode_ids,
            episode_metrics=suite_episode_metrics,
        ),
        "requested_tasks": len(task_names),
        "completed_tasks": len(completed_tasks),
        "failed_tasks": len(task_summaries) - len(completed_tasks),
    }
    evaluation_health_path = eval_dir / "evaluation_health.json"
    evaluation_health_path.write_text(
        json.dumps(evaluation_health, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "suite": suite_name,
        "evaluation_time_budget": frozen_budget if config.max_episode_steps is None else None,
        "max_episode_steps_override": config.max_episode_steps,
        "policy": config.policy,
        "tasks": task_summaries,
        "completed_tasks": len(completed_tasks),
        "failed_tasks": len(task_summaries) - len(completed_tasks),
        "successful_episodes": successful_episodes,
        "episodes": episodes,
        "success_rate": successful_episodes / episodes if episodes else None,
        **suite_metric_summary,
        "evaluation_health_path": str(evaluation_health_path),
    }
    _write_summary(eval_dir, summary)
    return summary


def main(
    suite_config: Annotated[Path, typer.Option()],
    data_root: Annotated[Path, typer.Option()],
    policy: Annotated[str, typer.Option()],
    host: Annotated[str, typer.Option()] = "172.17.0.1",
    port: Annotated[int, typer.Option()] = 21000,
    data_format: Annotated[str, typer.Option()] = "lerobot",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = False,
    eval_dir: Annotated[str, typer.Option()] = "data/evals_sonic",
    max_episode_steps: Annotated[int | None, typer.Option(min=1)] = None,
    num_episodes: Annotated[int, typer.Option()] = DEFAULT_NUM_EPISODES,
    episode_start: Annotated[int, typer.Option()] = 0,
    success_criteria: Annotated[float, typer.Option()] = 0.7,
    save_video: Annotated[bool, typer.Option("--save-video/--no-save-video")] = True,
    num_workers: Annotated[int, typer.Option()] = 1,
    save_exocentric_video: Annotated[
        bool, typer.Option("--save-exocentric-video/--no-save-exocentric-video")
    ] = False,
    exocentric_camera: Annotated[str, typer.Option()] = "franka_camera",
    exocentric_video_key: Annotated[str, typer.Option()] = "exocentric",
    async_video: Annotated[bool, typer.Option("--async-video/--no-async-video")] = True,
    env_step_dt: Annotated[float | None, typer.Option()] = None,
    sonic_control_dt: Annotated[float | None, typer.Option()] = None,
    fast: Annotated[bool, typer.Option("--fast/--no-fast")] = False,
    wbc_controller: Annotated[str | None, typer.Option("--wbc-controller")] = None,
    hand_type: Annotated[str, typer.Option()] = "dex3",
    seed: Annotated[int | None, typer.Option()] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = False,
) -> None:
    summary = evaluate_suite(
        EvalConfig(
            env_id="",
            policy=policy,
            wbc_controller=wbc_controller,
            host=host,
            port=port,
            data_format=data_format,
            sim_mode=sim_mode,
            headless=headless,
            eval_dir=eval_dir,
            max_episode_steps=max_episode_steps,
            num_episodes=num_episodes,
            episode_start=episode_start,
            success_criteria=success_criteria,
            save_video=save_video,
            num_workers=num_workers,
            save_exocentric_video=save_exocentric_video,
            exocentric_camera=exocentric_camera,
            exocentric_video_key=exocentric_video_key,
            async_video=async_video,
            seed=seed,
        ),
        suite_config,
        data_root,
        run_options={
            "env_step_dt": env_step_dt,
            "sonic_control_dt": sonic_control_dt,
            "fast": fast,
            "hand_type": hand_type,
            "seed": seed,
            "resume": resume,
        },
    )
    if summary["failed_tasks"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
