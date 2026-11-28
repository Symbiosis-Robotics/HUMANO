"""Run isolated policy services and evaluation episode shards."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

STOP_TIMEOUT_SECONDS = 15


def split_episodes(start: int, count: int, jobs: int) -> list[tuple[int, int]]:
    if start < 0 or count < 1 or not 1 <= jobs <= count:
        raise ValueError("Require episode_start >= 0 and 1 <= parallel_jobs <= num_episodes")
    size, extra = divmod(count, jobs)
    return [(start + i * size + min(i, extra), size + int(i < extra)) for i in range(jobs)]


def validate_parallel(args: argparse.Namespace) -> None:
    split_episodes(args.episode_start, args.num_episodes, args.parallel_jobs)
    if args.external_server or args.num_workers != 1 or args.sonic_mode != "native":
        raise ValueError(
            "Parallel jobs require owned policy servers, num_workers=1 and native SONIC"
        )
    if args.host not in {"localhost", "127.0.0.1"}:
        raise ValueError("Parallel jobs require a local policy client host")
    if args.server_host not in {"localhost", "127.0.0.1", "0.0.0.0"}:
        raise ValueError("Parallel jobs require a local policy server host")
    if args.wait_tries < 1 or args.wait_sleep_s <= 0:
        raise ValueError("Policy readiness wait parameters must be positive")


def plan_jobs(args: argparse.Namespace, build_commands: Callable) -> list[dict]:
    validate_parallel(args)
    reservations = []
    jobs = []
    try:
        for index, (start, count) in enumerate(
            split_episodes(args.episode_start, args.num_episodes, args.parallel_jobs)
        ):
            reservation = socket.socket()
            reservations.append(reservation)
            reservation.bind(("127.0.0.1", args.port + index if args.port else 0))
            port = reservation.getsockname()[1]
            directory = Path(args.eval_dir).expanduser().resolve() / f"job_{index:03d}"
            gpu_id = (
                args.gpu_ids[index]
                if getattr(args, "gpu_ids", None)
                else getattr(args, "gpu_id", None)
            )
            child = argparse.Namespace(
                **{
                    **vars(args),
                    "gpu_id": gpu_id,
                    "episode_start": start,
                    "num_episodes": count,
                    "port": port,
                    "eval_dir": str(directory),
                }
            )
            server, evaluation = build_commands(child)
            jobs.append(
                {
                    "job": index,
                    "gpu_id": gpu_id,
                    "episode_start": start,
                    "num_episodes": count,
                    "port": port,
                    "eval_dir": str(directory),
                    "status": "pending",
                    "server_command": server,
                    "eval_command": evaluation,
                    "result_paths": [],
                }
            )
    finally:
        for reservation in reservations:
            reservation.close()
    return jobs


def _start(command: str, log: Path) -> subprocess.Popen:
    with log.open("w") as stream:
        return subprocess.Popen(
            ["bash", "-lc", command],
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _stop(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def _ready(jobs: list[dict], servers: list[subprocess.Popen], args: argparse.Namespace) -> None:
    pending = set(range(len(jobs)))
    for _ in range(args.wait_tries):
        for index in tuple(pending):
            if servers[index].poll() is not None:
                raise RuntimeError(f"Policy server job {index} exited before becoming ready")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{jobs[index]['port']}/health", timeout=1
                ) as response:
                    health = json.load(response)
                if health["status"] != "ok":
                    raise RuntimeError(f"Policy server job {index} is unhealthy")
                pending.remove(index)
            except (urllib.error.URLError, TimeoutError):
                continue
        if not pending:
            return
        time.sleep(args.wait_sleep_s)
    raise TimeoutError(f"Policy servers did not become ready: {sorted(pending)}")


def verify_results(job: dict, suite_config: str | None) -> list[str]:
    from humano.cli.eval_suite import _read_task_metrics, load_suite_tasks

    root = Path(job["eval_dir"])
    paths = [root / "eval_metrics.jsonl"]
    if suite_config:
        tasks = load_suite_tasks(Path(suite_config))
        paths = [root / "tasks" / task / "eval_metrics.jsonl" for task in tasks]
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text())
        if summary["failed_tasks"] or summary["completed_tasks"] != len(tasks):
            raise RuntimeError(f"Incomplete suite results: {summary_path}")
    expected = {
        f"episode_{i}"
        for i in range(job["episode_start"], job["episode_start"] + job["num_episodes"])
    }
    for path in paths:
        if set(_read_task_metrics(path)) != expected:
            raise RuntimeError(f"Episode coverage differs from assigned shard: {path}")
    return [str(path) for path in paths] + ([str(root / "summary.json")] if suite_config else [])


def _wait_evals(
    jobs: list[dict],
    evaluators: list[subprocess.Popen],
    servers: list[subprocess.Popen],
    args: argparse.Namespace,
) -> None:
    pending = set(range(len(jobs)))
    while pending:
        for index in tuple(pending):
            code = evaluators[index].poll()
            if code is None:
                if servers[index].poll() is not None:
                    raise RuntimeError(f"Policy server job {index} exited during evaluation")
                continue
            if code:
                jobs[index]["status"] = "failed"
                raise RuntimeError(f"Evaluation job {index} exited with code {code}")
            jobs[index]["result_paths"] = verify_results(jobs[index], args.suite_config)
            jobs[index]["status"] = "completed"
            pending.remove(index)
        if pending:
            time.sleep(args.wait_sleep_s)


def _interrupt(_signal: int, _frame: object) -> None:
    raise KeyboardInterrupt


def run_parallel(args: argparse.Namespace, build_commands: Callable) -> None:
    jobs = plan_jobs(args, build_commands)
    manifest = {
        "parallel_jobs": args.parallel_jobs,
        "episode_start": args.episode_start,
        "num_episodes": args.num_episodes,
        "status": "planned",
        "jobs": jobs,
    }
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        return
    root = Path(args.eval_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "parallel_manifest.json"
    if manifest_path.exists() or any(Path(job["eval_dir"]).exists() for job in jobs):
        raise FileExistsError("Parallel output already exists; choose a new --eval-dir")
    processes = []
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in handlers:
            signal.signal(sig, _interrupt)
        for job in jobs:
            Path(job["eval_dir"]).mkdir()
            processes.append(
                _start(job["server_command"], Path(job["eval_dir"]) / "policy_server.log")
            )
        _ready(jobs, processes, args)
        servers = list(processes)
        for job in jobs:
            processes.append(
                _start(job["eval_command"], Path(job["eval_dir"]) / "eval_process.log")
            )
            job["status"] = "running"
        manifest["status"] = "running"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        _wait_evals(jobs, processes[len(servers) :], servers, args)
        manifest["status"] = "completed"
    finally:
        _stop(processes)
        if manifest["status"] != "completed":
            manifest["status"] = "incomplete"
            manifest["error"] = str(sys.exc_info()[1])
            for job in jobs:
                if job["status"] not in {"completed", "failed"}:
                    job["status"] = "incomplete"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
