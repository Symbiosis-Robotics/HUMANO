from __future__ import annotations

import copy
import hashlib
import importlib
import json
import logging
import multiprocessing as mp
import os
import random
import shlex
import signal
import subprocess
from collections import defaultdict
from multiprocessing.connection import wait

os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"
import pickle
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import gymnasium as gym
import mujoco
import numpy as np
import torch
import typer
import tyro
from gymnasium import spaces
from rich.live import Live

import humano.envs as _  # noqa: F401
from humano.envs.wrappers import VideoRecorder
from humano.evals.api import (
    DEFAULT_NUM_EPISODES,
    EvalConfig,
    EvalResult,
    resolve_wbc_controller,
)
from humano.evals.eval_scene_protocol import (
    EVAL_SCENE_SELECTION_SCHEMA as MERGED_EVAL_SCENE_SELECTION_SCHEMA,
)
from humano.evals.eval_scene_protocol import resolve_eval_scene_selection
from humano.evals.metrics import (
    FallDetector,
    core_ground_contact_body,
    dimensionless_jerk,
    episode_timed_out,
    summarize_episode_metrics,
    summarize_evaluation_health,
)
from humano.evals.ood_physics import (
    EVAL_ROBOT_INITIAL_PARAMS,
    EVAL_TASK_ROBOT_INITIAL_STATE,
)
from humano.evals.time_budget import episode_step_limit, load_task_time_budget
from humano.evals.tui import (
    WorkerProgress,
    make_console,
    render_progress,
    restore_cursor,
    update_progress,
)
from humano.robots.hands import get_hand_spec
from humano.scenes.bev_cache import resolve_project_path
from humano.scenes.scene_selection import (
    EVAL_SCENE_SELECTION_SCHEMA as LEGACY_EVAL_SCENE_SELECTION_SCHEMA,
)
from humano.utils import snake_to_pascal
from humano.utils.config import (
    configure_logging,
    route_prints_to_logger,
    route_stderr_to_logger,
)
from humano.wbc.sonic import (
    SONIC_HANDOFF_BIRTH_ROOT_Z,
)

LOGGER = logging.getLogger(__name__)
_ORIGINAL_PRINT = print
_PRINT_LOGGER: logging.Logger | None = None
REPO_ROOT = Path(__file__).resolve().parents[3]


class EvalSceneWallCollisionError(RuntimeError):
    def __init__(self, *, phase: str, step: int, detail: str) -> None:
        self.phase = phase
        self.step = step
        super().__init__(
            f"Eval scene wall collision detected during {phase} at step {step}: {detail}"
        )


def _episode_outcome(
    *,
    env_success: bool,
    fall_detected: bool,
    timed_out: bool,
    policy_exhausted: bool,
    policy_wall_collision: bool,
) -> tuple[bool, str]:
    if fall_detected:
        return False, "fall_down"
    if policy_wall_collision:
        return False, "wall_collision"
    if env_success:
        return True, "success"
    if timed_out:
        return False, "timeout"
    if policy_exhausted:
        return False, "policy_exhausted"
    return False, "ended_without_success"


def _message_from_print_args(*args: Any, **kwargs: Any) -> str:
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    return (sep.join(str(arg) for arg in args) + end).rstrip("\n")


def _log_print(*args: Any, **kwargs: Any) -> None:
    file = kwargs.get("file")
    if file not in (None, sys.stdout, sys.stderr):
        _ORIGINAL_PRINT(*args, **kwargs)
        return
    if _PRINT_LOGGER is None:
        _ORIGINAL_PRINT(*args, **kwargs)
        return
    _PRINT_LOGGER.info(_message_from_print_args(*args, **kwargs))


print = _log_print


class ExocentricVideoObservation(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        camera_name: str = "franka_camera",
        observation_key: str = "exocentric",
    ):
        super().__init__(env)
        self.camera_name = camera_name
        self.observation_key = observation_key
        self._warned_missing_camera = False
        self._warned_no_space = False
        self._ensured_camera = False
        self._step_count = 0

        observation_space = getattr(env, "observation_space", None)
        if isinstance(observation_space, spaces.Dict):
            spaces_dict = dict(observation_space.spaces)
            source_space = self._find_image_space(spaces_dict)
            if source_space is not None:
                spaces_dict[observation_key] = spaces.Box(
                    0,
                    255,
                    shape=source_space.shape,
                    dtype=source_space.dtype,
                )
                self.observation_space = spaces.Dict(spaces_dict)
            else:
                self._warned_no_space = True

    @staticmethod
    def _find_image_space(spaces_dict: dict[str, spaces.Space]) -> spaces.Box | None:
        for subspace in spaces_dict.values():
            if (
                isinstance(subspace, spaces.Box)
                and len(subspace.shape) == 3
                and subspace.shape[-1] == 3
            ):
                return subspace
        return None

    def _exocentric_frame(self, observation: dict[str, Any]) -> Any | None:
        frame_sets = (
            getattr(self.unwrapped, "_last_mujoco_frame", None),
            getattr(self.unwrapped, "_last_isaac_frame", None),
        )
        for frame_set in frame_sets:
            if isinstance(frame_set, dict) and self.camera_name in frame_set:
                return frame_set[self.camera_name]

        self._ensure_exocentric_camera()
        if self._ensured_camera:
            rendered = self.unwrapped.render()
            if isinstance(rendered, dict) and self.camera_name in rendered:
                return rendered[self.camera_name]

        rendered = self.unwrapped.render()
        if isinstance(rendered, dict) and self.camera_name in rendered:
            return rendered[self.camera_name]

        if not self._warned_missing_camera:
            available = []
            for frame_set in frame_sets:
                if isinstance(frame_set, dict):
                    available.extend(frame_set.keys())
            if isinstance(rendered, dict):
                available.extend(rendered.keys())
            print(
                "[SonicEval] Exocentric camera "
                f"{self.camera_name!r} not found; available cameras: {sorted(set(available))}. "
                "Writing black frames for the exocentric video."
            )
            self._warned_missing_camera = True

        for value in observation.values():
            if (
                hasattr(value, "shape")
                and hasattr(value, "dtype")
                and len(value.shape) == 3
                and value.shape[-1] == 3
            ):
                return value * 0
        return None

    def _ensure_exocentric_camera(self) -> None:
        if self._ensured_camera:
            return
        mujoco_sim = getattr(self.unwrapped, "mujoco", None)
        ensure_camera = getattr(mujoco_sim, "ensure_fixed_camera", None)
        if ensure_camera is None:
            return

        lookat = [0.0, 0.0, 0.38]
        azimuth = 100.0
        elevation = -30.0
        distance = 3.0
        azimuth_rad = np.deg2rad(azimuth)
        elevation_rad = np.deg2rad(elevation)
        position = (
            np.asarray(lookat, dtype=np.float64)
            + np.array(
                [
                    distance * np.cos(elevation_rad) * np.cos(azimuth_rad),
                    distance * np.cos(elevation_rad) * np.sin(azimuth_rad),
                    -distance * np.sin(elevation_rad),
                ],
                dtype=np.float64,
            )
        ).tolist()
        fovy = 95.0
        resolution = (640, 360)
        ensure_camera(
            self.camera_name,
            position=position,
            lookat=lookat,
            fovy=fovy,
            resolution=resolution,
        )
        self._ensured_camera = True
        print(
            "[SonicEval] Created MuJoCo exocentric camera "
            f"{self.camera_name!r} position={position} lookat={lookat} fovy={fovy}."
        )

    def _with_exocentric(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.observation_key in observation:
            return observation
        frame = self._exocentric_frame(observation)
        if frame is None:
            return observation
        observation = dict(observation)
        observation[self.observation_key] = frame
        return observation

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._with_exocentric(observation), info

    def step(self, action):
        t_start = time.monotonic()
        observation, reward, terminated, truncated, info = self.env.step(action)
        t_inner = time.monotonic()
        result = self._with_exocentric(observation), reward, terminated, truncated, info
        t_exo = time.monotonic()
        self._step_count += 1
        if self._step_count <= 3 or self._step_count % 200 == 0:
            inner_ms = (t_inner - t_start) * 1000.0
            exo_ms = (t_exo - t_inner) * 1000.0
            print(
                f"[ExocentricVideoObs] step={self._step_count} "
                f"inner_env={inner_ms:.1f}ms exo_merge={exo_ms:.1f}ms",
                flush=True,
            )
        return result


class ManagedProcess:
    def __init__(
        self,
        name: str,
        cmd: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
    ):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.env = env
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self._log_handle = None

    def _close_log_handle(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None

    def start(self) -> None:
        print(f"[SonicEval] Starting {self.name}: {' '.join(shlex.quote(x) for x in self.cmd)}")
        stdout = None
        stderr = None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(str(self.log_path), "a")
            stdout = self._log_handle
            stderr = subprocess.STDOUT
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                cwd=str(self.cwd) if self.cwd else None,
                env=self.env,
                start_new_session=True,
                stdout=stdout,
                stderr=stderr,
            )
        except Exception:
            self._close_log_handle()
            raise

    def check_started(self) -> bool:
        if self.proc is None:
            print(f"[SonicEval] {self.name} failed to start.")
            return False
        code = self.proc.poll()
        if code is None:
            print(f"[SonicEval] {self.name} started successfully.")
            return True
        self.proc = None
        if code == 0:
            print(f"[SonicEval] {self.name} exited normally.")
            return False
        self._close_log_handle()
        raise RuntimeError(f"{self.name} exited early with code {code}")

    def stop(self) -> None:
        if self.proc is None:
            self._close_log_handle()
            return
        print(f"[SonicEval] Stopping {self.name}")
        try:
            pgid = os.getpgid(self.proc.pid)
        except ProcessLookupError:
            self.proc = None
            self._close_log_handle()
            return
        if self.proc.poll() is not None:
            self.proc = None
            self._close_log_handle()
            return
        for sig, timeout in (
            (signal.SIGINT, 5),
            (signal.SIGTERM, 3),
            (signal.SIGKILL, 3),
        ):
            try:
                os.killpg(pgid, sig)
                self.proc.wait(timeout=timeout)
                self.proc = None
                self._close_log_handle()
                return
            except Exception:
                pass
        self.proc = None
        self._close_log_handle()


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        env.pop(key, None)
    env.setdefault("ROS_LOCALHOST_ONLY", "1")
    env.setdefault("SONIC_SKIP_MOTION_SWITCHER", "1")
    env.setdefault("SONIC_DISABLE_AUDIO", "1")
    return env


def _apply_eval_scene_hidden_paths(
    isaac_sim: Any,
    scene: Any,
    hidden_paths: list[str],
) -> dict[str, list[str]]:
    from pxr import UsdGeom

    root_path = isaac_sim._indexed_scene_prim_path(scene)
    hidden_collision_paths = isaac_sim._indexed_collision_hidden.setdefault(scene.cache_key, set())
    applied = []
    missing = []
    for source_path in sorted(set(hidden_paths)):
        prim_path = source_path.replace("/World", root_path, 1)
        prim = isaac_sim.world.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
            missing.append(source_path)
            continue
        UsdGeom.Imageable(prim).MakeInvisible()
        hidden_collision_paths.add(prim_path)
        applied.append(source_path)
    return {
        "applied_hidden_prim_paths": applied,
        "missing_hidden_prim_paths": missing,
    }


def _prepare_eval_scene(
    sonic_env: Any,
    environment_config: dict[str, Any],
    sim_mode: str,
) -> tuple[dict[str, Any], Any | None, list[str] | None, bool]:
    selection = environment_config.get("eval_scene")
    if selection is None:
        return environment_config, None, None, False
    schema = selection.get("schema")
    if schema == MERGED_EVAL_SCENE_SELECTION_SCHEMA:
        runtime_selection = resolve_eval_scene_selection(
            selection,
            repository_root=REPO_ROOT,
        )
    elif schema == LEGACY_EVAL_SCENE_SELECTION_SCHEMA:
        runtime_selection = selection
    else:
        raise ValueError(f"Unsupported eval scene schema: {selection.get('schema')!r}")
    if "isaac" not in sim_mode or sonic_env.isaac is None:
        raise ValueError("Eval scene selections require an Isaac-enabled sim mode")

    scene_index = resolve_project_path(runtime_selection["scene_index"])
    if (
        hashlib.sha256(scene_index.read_bytes()).hexdigest()
        != runtime_selection["scene_index_hash"]
    ):
        raise ValueError(f"Eval scene index content changed: {scene_index}")
    from humano.evals.indexed_environment import (
        _install_indexed_scene_randomizer,
        _prepare_indexed_environment_config,
    )
    from humano.scenes.indexed import (
        align_indexed_background_scene,
        load_indexed_background_scene,
    )

    scene = load_indexed_background_scene(scene_index, runtime_selection["scene_uid"])
    if scene.revision != runtime_selection["scene_revision"]:
        raise ValueError(
            f"Scene revision changed for {scene.uid}: "
            f"expected {runtime_selection['scene_revision']}, found {scene.revision}"
        )
    if selection.get("hidden_prim_paths_final") is not True:
        raise ValueError(
            "Eval scene selections must set hidden_prim_paths_final=True; "
            "regenerate eval data with frozen hidden prim paths before evaluation."
        )
    placement = runtime_selection["placement"]
    final_hidden_paths = list(runtime_selection.get("hidden_prim_paths", []))
    finalized_scene_selection = True
    scene = align_indexed_background_scene(
        scene,
        translation_xy=tuple(float(value) for value in placement["translation_xy"]),
        yaw_rad=float(placement["yaw_rad"]),
        hidden_anchor_prim_path=None,
        hide_all_background_assets=False,
        additional_hidden_prim_paths=tuple(final_hidden_paths),
    )
    hidden_paths_to_replay = final_hidden_paths
    scene.metadata["final_eval_scene_hidden_prim_paths"] = final_hidden_paths
    prepared = _prepare_indexed_environment_config(environment_config)
    scene_state = prepared.get("dr_state_dict", {}).get("scene")
    has_task_table = isinstance(scene_state, dict) and isinstance(scene_state.get("table"), dict)
    if has_task_table:
        _install_indexed_scene_randomizer(sonic_env, scene)
        return (
            prepared,
            None,
            hidden_paths_to_replay,
            finalized_scene_selection,
        )
    sonic_env.task.dr.randomizers.pop("scene", None)
    return (
        prepared,
        scene,
        hidden_paths_to_replay,
        finalized_scene_selection,
    )


def _make_eval_scene_wall_guard(sonic_env: Any, scene: Any | None):
    if scene is None:
        return lambda _phase, _step: None
    from humano.scenes.indexed import IndexedBackgroundScene

    if not isinstance(scene, IndexedBackgroundScene):
        raise TypeError(f"Expected IndexedBackgroundScene for eval wall checks, got {type(scene)}")
    from humano.evals.indexed_environment import (
        _moving_geom_ids,
        _validate_current_indexed_scene,
    )

    geom_ids = _moving_geom_ids(sonic_env.mujoco)

    def validate(phase: str, step: int) -> None:
        try:
            _validate_current_indexed_scene(scene, sonic_env.mujoco, geom_ids)
        except ValueError as exc:
            raise EvalSceneWallCollisionError(phase=phase, step=step, detail=str(exc)) from exc

    return validate


def _prepend_env_path(env: dict[str, str], key: str, *paths: Path | str) -> None:
    existing = [part for part in env.get(key, "").split(":") if part]
    prefix = [str(path) for path in paths if path and Path(path).exists()]
    env[key] = ":".join(prefix + [part for part in existing if part not in prefix])


def _deploy_runtime_env(deploy_root: Path) -> dict[str, str]:
    env = _clean_env()
    env.setdefault(
        "FASTRTPS_DEFAULT_PROFILES_FILE",
        str(deploy_root / "src/g1/g1_deploy_onnx_ref/config/fastrtps_profile.xml"),
    )
    env.setdefault("CUDAToolkit_ROOT", os.environ.get("CUDAToolkit_ROOT", "/usr/local/cuda"))
    env.setdefault("CUDA_HOME", os.environ.get("CUDA_HOME", env["CUDAToolkit_ROOT"]))
    tensor_rt_root = os.environ.get("TensorRT_ROOT") or str(Path.home() / "TensorRT")
    env.setdefault("TensorRT_ROOT", tensor_rt_root)
    _prepend_env_path(
        env,
        "LD_LIBRARY_PATH",
        "/opt/onnxruntime/lib",
        Path(env["CUDAToolkit_ROOT"]) / "lib64",
        Path(env["CUDAToolkit_ROOT"]) / "lib",
        Path(tensor_rt_root) / "lib",
        deploy_root / "thirdparty/unitree_sdk2/thirdparty/lib/x86_64",
    )
    return env


def _resolve_deploy_path(deploy_root: Path, path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return deploy_root / path


def _check_file(deploy_root: Path, path_like: str, name: str) -> None:
    path = _resolve_deploy_path(deploy_root, path_like)
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")


def _check_dir(deploy_root: Path, path_like: str, name: str) -> None:
    path = _resolve_deploy_path(deploy_root, path_like)
    if not path.is_dir():
        raise FileNotFoundError(f"{name} not found: {path}")


def _build_deploy_process(
    *,
    deploy_root: Path,
    deploy_interface: str,
    checkpoint: str,
    motion_data: str,
    obs_config: str,
    planner: str,
    zmq_host: str,
    zmq_port: int,
    feedback_port: int,
    logs_dir: str | None,
    log_path: Path | None = None,
    sonic_control_dt: float | None = None,
) -> ManagedProcess:
    binary = Path("target/release/g1_deploy_onnx_ref")
    _check_file(deploy_root, binary.as_posix(), "SONIC deploy binary")
    _check_file(deploy_root, f"{checkpoint}_decoder.onnx", "SONIC decoder model")
    _check_file(deploy_root, f"{checkpoint}_encoder.onnx", "SONIC encoder model")
    _check_file(deploy_root, obs_config, "SONIC observation config")
    _check_file(deploy_root, planner, "SONIC planner model")
    _check_dir(deploy_root, motion_data, "SONIC motion data")

    args = [
        f"./{binary.as_posix()}",
        deploy_interface,
        f"{checkpoint}_decoder.onnx",
        motion_data,
        "--obs-config",
        obs_config,
        "--encoder-file",
        f"{checkpoint}_encoder.onnx",
        "--planner-file",
        planner,
        "--input-type",
        "zmq_manager",
        "--output-type",
        "all",
        "--zmq-host",
        zmq_host,
        "--zmq-port",
        str(zmq_port),
        "--zmq-out-port",
        str(feedback_port),
        "--zmq-verbose",
        "--disable-crc-check",
    ]
    if logs_dir:
        args.extend(["--logs-dir", logs_dir])
    if sonic_control_dt is not None:
        args.extend(["--control-dt", f"{sonic_control_dt:.4f}"])
    shell_args = [
        "bash",
        "-lc",
        f"source scripts/setup_env.sh && exec {' '.join(shlex.quote(a) for a in args)}",
    ]
    return ManagedProcess("SONIC deploy", shell_args, cwd=deploy_root, log_path=log_path)


SONIC_TOKEN_POLICIES = {
    "sonic_policy",
    "gr00t_n17",
    "psi0",
    "act",
    "diffusion",
    "pi05",
    "flow_matching",
}


def _policy_module_name(policy: str) -> str:
    normalized = policy.lower().replace("-", "_").replace(".", "")
    if normalized == "remote":
        return "sonic_policy"
    if normalized in {"gr00t_n17", "gr00tn17", "gr00t_n1d7", "gr00tn1d7"}:
        return "gr00t_n17"
    if normalized in {"pi05", "pi_05"}:
        return "pi05"
    if normalized == "flow_matching":
        return "flow_matching"
    return normalized


def _is_sonic_token_policy(policy: str) -> bool:
    return _policy_module_name(policy) in SONIC_TOKEN_POLICIES


def _append_eval_stats_line(eval_dir: str, line: str) -> None:
    path = Path(eval_dir) / "eval_stats.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", buffering=1) as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _append_eval_metrics_record(eval_dir: str, record: dict[str, Any]) -> None:
    path = Path(eval_dir) / "eval_metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", buffering=1) as metrics_file:
        metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
        metrics_file.flush()
        os.fsync(metrics_file.fileno())


def _net_flag(interface: str, name: str) -> int | None:
    path = Path("/sys/class/net") / interface / name
    try:
        return int(path.read_text().strip(), 0)
    except Exception:
        return None


def _pick_sonic_interface(configured: str | None = None) -> str:
    forced = (
        os.environ.get("SONIC_DDS_INTERFACE") or os.environ.get("HUMANO_SONIC_INTERFACE") or ""
    ).strip()
    if forced:
        return forced

    configured = (configured or "").strip()
    if configured and configured != "lo":
        return configured

    sys_net = Path("/sys/class/net")
    if not sys_net.is_dir():
        return configured or "lo"

    skip_prefixes = ("br-", "docker", "veth", "virbr", "tailscale")
    candidates: list[tuple[int, str]] = []
    for iface_path in sys_net.iterdir():
        iface = iface_path.name
        if iface == "lo" or iface.startswith(skip_prefixes):
            continue
        flags = _net_flag(iface, "flags")
        if flags is None:
            continue
        is_up = bool(flags & 0x1)
        supports_multicast = bool(flags & 0x1000)
        if not (is_up and supports_multicast):
            continue
        operstate = (iface_path / "operstate").read_text(errors="ignore").strip()
        priority = 0 if operstate == "up" else 1
        candidates.append((priority, iface))

    if candidates:
        candidates.sort()
        return candidates[0][1]
    return configured or "lo"


def _make_sonic_config(hand_spec=None) -> dict[str, Any]:
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    if hand_spec is not None:
        config.wbc_version = hand_spec.wbc_version
    sonic_config = config.load_wbc_yaml()
    if hand_spec is not None:
        sonic_config["WBC_VERSION"] = hand_spec.wbc_version
    selected_interface = _pick_sonic_interface(str(sonic_config.get("INTERFACE", "")))
    sonic_config["ENV_NAME"] = "humano"
    sonic_config["INTERFACE"] = selected_interface
    sonic_config["SONIC_ZMQ_HOST"] = "localhost"
    sonic_config["SONIC_ZMQ_PORT"] = int(os.environ.get("SONIC_ZMQ_PORT", "5556"))
    sonic_config["FEEDBACK_PORT"] = int(os.environ.get("SONIC_ZMQ_OUT_PORT", "5557"))
    sonic_config["ENABLE_ELASTIC_BAND"] = True
    sonic_config["SUSPEND_ONLY_AT_START"] = True
    sonic_config.setdefault("ELASTIC_BAND_Z", 1.0)
    # HACK: total Phase-1 bootstrap duration; native pre-stabilisation is the first 40 frames.
    sonic_config.setdefault("SONIC_BOOTSTRAP_STABILIZE_STEPS", 500)
    # HACK: decoder warmup runs after bootstrap while holding the native SONIC command.
    sonic_config.setdefault("GR00T_DECODER_WARMUP_STEPS", 150)
    print(f"[SonicEval] Using DDS interface: {selected_interface}")
    return sonic_config


def _configure_policy_sonic_start(policy: str, sonic_config: dict[str, Any]) -> None:
    if _policy_module_name(policy) in ("amo", "twist2"):
        sonic_config.update(ENABLE_ELASTIC_BAND=False, SUSPEND_ONLY_AT_START=False)
        return
    if not _is_sonic_token_policy(policy):
        return

    sonic_config["DECOUPLED_WBC_BOOTSTRAP"] = True
    sonic_config["START_ON_GROUND"] = False
    sonic_config["GROUND_START_HOLD_BEFORE_LOWCMD"] = False
    sonic_config["ENABLE_ELASTIC_BAND"] = False
    sonic_config["SUSPEND_ONLY_AT_START"] = False


def _get_sonic_mode() -> str:
    """Return the SONIC motion-tracking backend mode.

    Reads ``HUMANO_SONIC_MODE`` env var set by ``launch_evaluation.py``.
    Values: ``"none"`` | ``"native"``; legacy C++ modes are disabled.
    """
    mode = os.environ.get("HUMANO_SONIC_MODE", "native")
    if mode not in ("none", "native"):
        raise ValueError("HUMANO_SONIC_MODE must be native (or none for non-SONIC policies)")
    return mode


def _run_eval_worker_entry(
    worker_result_path: str,
    worker_id: int,
    num_workers: int,
    worker_kwargs: dict[str, Any],
    log_path: str | None,
    progress_conn: Any | None,
):
    def report(payload: dict[str, Any]) -> None:
        if progress_conn is not None:
            progress_conn.send((worker_id, payload))

    try:
        global _PRINT_LOGGER
        configure_logging(log_file=log_path, console=False)
        _PRINT_LOGGER = LOGGER
        with route_prints_to_logger(LOGGER), route_stderr_to_logger(LOGGER):
            try:
                _run_eval_worker(
                    **worker_kwargs,
                    worker_id=worker_id,
                    num_workers=num_workers,
                    worker_result_path=worker_result_path,
                    progress_reporter=report,
                )
            except BaseException:
                report(
                    {
                        "event": "worker_error",
                        "message": "see eval log",
                    }
                )
                with open(worker_result_path, "wb") as f:
                    pickle.dump(("err", worker_id, traceback.format_exc()), f)
    finally:
        if progress_conn is not None:
            try:
                progress_conn.close()
            except Exception:
                pass


def _is_robot_initial_ood(environment_config: dict[str, Any]) -> bool:
    return environment_config.get("ood", {}).get("dimension") in {"robot_init", "robot_initial"}


def _eval_robot_initial_params(environment_config: dict[str, Any]) -> dict[str, Any] | None:
    if not _is_robot_initial_ood(environment_config):
        return None
    params = environment_config.get("ood", {}).get("params")
    if not isinstance(params, dict):
        raise ValueError("robot_initial OOD eval requires saved params")
    required = ("robot_key", "translation_xy_m", "yaw_rad", "accepted_pose")
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"robot_initial OOD params missing {missing}")
    return {key: copy.deepcopy(params[key]) for key in required}


def _yaw_quaternion_eval(quaternion: np.ndarray, yaw: float) -> list[float]:
    w, x, y, z = quaternion
    c, s = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return [
        float(c * w - s * z),
        float(c * x - s * y),
        float(c * y + s * x),
        float(c * z + s * w),
    ]


def _restore_robot_initial_base_pose(spatial: dict[str, Any], params: dict[str, Any]) -> None:
    robot_key = params["robot_key"]
    accepted_pose = params["accepted_pose"]
    if not isinstance(accepted_pose, dict):
        raise ValueError("robot_initial OOD eval requires accepted_pose params")
    translation = np.asarray(params["translation_xy_m"], dtype=float)
    if translation.shape != (2,) or not np.all(np.isfinite(translation)):
        raise ValueError("robot_initial translation_xy_m must be two finite values")
    yaw = float(params["yaw_rad"])
    if not np.isfinite(yaw):
        raise ValueError("robot_initial yaw_rad must be finite")
    base_pose = copy.deepcopy(accepted_pose)
    position = list(base_pose["position"])
    position[0] = float(position[0]) - float(translation[0])
    position[1] = float(position[1]) - float(translation[1])
    base_pose["position"] = position
    base_pose["quaternion"] = _yaw_quaternion_eval(
        np.asarray(base_pose["quaternion"], dtype=float),
        -yaw,
    )
    spatial[robot_key] = base_pose


def _eval_reset_state_dict(environment_config: dict[str, Any]) -> dict[str, Any]:
    state = copy.deepcopy(environment_config)
    robot_initial_params = _eval_robot_initial_params(state)
    spatial = state.get("dr_state_dict", {}).get("spatial")
    if robot_initial_params is not None:
        if not isinstance(spatial, dict):
            raise ValueError("robot_initial OOD eval requires saved spatial state")
        _restore_robot_initial_base_pose(spatial, robot_initial_params)
        state[EVAL_ROBOT_INITIAL_PARAMS] = robot_initial_params
    if "ood_robot_joint_qpos" in state or "ood_physics_state" in state:
        state.pop("ood_robot_joint_qpos", None)
        state[EVAL_TASK_ROBOT_INITIAL_STATE] = True
    return state


def _seed_eval_rng(seed: int | None, *, use_isaac: bool = False) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        if use_isaac:
            # Replicator is available only after SimulationApp has started.
            from omni.replicator.core import set_global_seed

            set_global_seed(seed)


def _reset_eval_episode(
    env: Any,
    environment_config: dict[str, Any],
    base_seed: int | None,
    episode_index: int,
    camera_randomizer: Any | None = None,
    *,
    use_isaac: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_dict = _eval_reset_state_dict(environment_config)
    if base_seed is None:
        return env.reset(options={"state_dict": state_dict})

    episode_seed = base_seed + episode_index
    _seed_eval_rng(episode_seed, use_isaac=use_isaac)
    if camera_randomizer is not None:
        camera_randomizer.reseed(episode_seed)
    return env.reset(seed=episode_seed, options={"state_dict": state_dict})


def _compensated_root_pose(
    root_pose: np.ndarray, drift_body_xy: np.ndarray, drift_yaw_rad: float
) -> np.ndarray:
    """Place the root so the measured startup drift ends at the dataset pose."""
    compensated = np.asarray(root_pose, dtype=np.float64).copy()
    w, x, y, z = compensated[3:7]
    initial_yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    compensated_yaw = initial_yaw - drift_yaw_rad
    c, s = np.cos(compensated_yaw), np.sin(compensated_yaw)
    dx, dy = np.asarray(drift_body_xy, dtype=np.float64)
    compensated[0] -= c * dx - s * dy
    compensated[1] -= s * dx + c * dy
    inverse_yaw = np.array([np.cos(drift_yaw_rad / 2), 0.0, 0.0, -np.sin(drift_yaw_rad / 2)])
    corrected_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(corrected_quat, inverse_yaw, compensated[3:7])
    compensated[3:7] = corrected_quat
    return compensated


def _run_eval_worker(
    env_id: Annotated[str, typer.Argument()],
    policy: Annotated[str, typer.Argument()],
    split: Annotated[str, typer.Argument()] = "eval",
    host: Annotated[str, typer.Option()] = "172.17.0.1",
    port: Annotated[int, typer.Option()] = 21000,
    data_format: Annotated[str, typer.Option()] = "lerobot",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = False,
    eval_dir: Annotated[str, typer.Option()] = "data/evals_sonic",
    max_episode_steps: Annotated[int | None, typer.Option()] = None,
    num_episodes: Annotated[int, typer.Option()] = DEFAULT_NUM_EPISODES,
    episode_start: Annotated[int, typer.Option()] = 0,
    data_dir: Annotated[str, typer.Option()] = "data/datagen",
    success_criteria: Annotated[float, typer.Option()] = 0.7,
    save_video: Annotated[bool, typer.Option("--save-video/--no-save-video")] = True,
    sonic_config: dict[str, Any] | None = None,
    deploy_root: str = "third_party/gear_sonic_deploy",
    deploy_interface: str = "lo",
    checkpoint: str = "policy/release/model",
    obs_config: str = "policy/release/observation_config.yaml",
    planner: str = "planner/target_vel/V2/planner_sonic.onnx",
    motion_data: str = "reference/example/",
    feedback_port: int = 5557,
    deploy_logs_dir: str | None = None,
    deploy_startup_s: float = 10.0,
    worker_id: int = 0,
    num_workers: int = 1,
    worker_result_path: str | None = None,
    progress_reporter: Callable[[dict[str, Any]], None] | None = None,
    time_budget: dict[str, Any] | None = None,
    # New parameters from HUMANO sync:
    async_video: bool = True,
    save_exocentric_video: bool = False,
    exocentric_camera: str = "franka_camera",
    exocentric_video_key: str = "exocentric",
    sonic_deploy_log_path: str | None = None,
    env_step_dt: float | None = None,
    sonic_control_dt: float | None = None,
    fast: bool = False,
    hand_type: str = "dex3",
    movement: bool = False,
    seed: int | None = None,
    wbc_controller: str | None = None,
):
    hand_spec = get_hand_spec(hand_type)

    def persist_payload(kind: str, payload: Any):
        if not worker_result_path:
            return
        with open(worker_result_path, "wb") as f:
            pickle.dump((kind, worker_id, payload), f)

    def report(event: str, **payload: Any) -> None:
        if progress_reporter is not None:
            progress_reporter({"event": event, **payload})

    if sonic_config is None:
        sonic_config = _make_sonic_config(hand_spec=hand_spec)
    controller = resolve_wbc_controller(policy, wbc_controller)
    policy_module_name = (
        controller if controller in ("amo", "twist2") else _policy_module_name(policy)
    )
    _configure_policy_sonic_start(policy_module_name, sonic_config)
    if save_exocentric_video:
        sonic_config["EXOCENTRIC_CAMERA"] = {
            "name": exocentric_camera,
            "azimuth": 100.0,
            "elevation": -30.0,
            "distance": 3.0,
            "lookat": [0.0, 0.0, 0.38],
            "fovy": 95.0,
            "resolution": [640, 360],
        }

    eval_sonic_mode = _get_sonic_mode()
    print(f"[SonicEval] SONIC mode: {eval_sonic_mode}")
    deploy_proc: ManagedProcess | None = None
    agent = None
    sim_dt = sonic_config["SIMULATE_DT"]
    control_dt = sim_dt  # 0.005 s / 200 Hz, matching the SONIC low-level loop.
    sonic_config["CONTROL_DT"] = control_dt
    _env_step_dt = env_step_dt if env_step_dt is not None else sim_dt
    if _env_step_dt != sim_dt:
        print(
            f"[SonicEval] env.step() interval overridden: "
            f"{_env_step_dt * 1000:.1f}ms (sim_dt={sim_dt * 1000:.1f}ms)"
        )
    print(
        f"[SonicEval] Timing -- sim_dt={sim_dt * 1000:.1f}ms "
        f"env_step_dt={_env_step_dt * 1000:.1f}ms "
        f"control_dt={control_dt * 1000:.1f}ms"
    )

    # -- Fast mode: only active in native sonic + mujoco_isaac --
    # 参照 SIMPLE: observation frames 在 GR00T 查询之间是占位零帧，
    # 因此强制禁用视频录制。
    if fast:
        if eval_sonic_mode != "native":
            print(
                f"[SonicEval] WARNING: --fast ignored (requires native SONIC mode, "
                f"current mode={eval_sonic_mode!r})"
            )
        elif "isaac" not in sim_mode:
            print(
                f"[SonicEval] WARNING: --fast ignored (requires mujoco_isaac sim mode, "
                f"current sim_mode={sim_mode!r})"
            )
        else:
            save_video = False
            save_exocentric_video = False
            print(
                "[SonicEval] Fast mode enabled: isaac sync only on GR00T query, "
                "video recording disabled"
            )

    eval_output_dir = os.path.join(eval_dir, policy, env_id.split("/")[1], split)
    os.makedirs(eval_output_dir, exist_ok=True)

    if episode_start < 0:
        raise ValueError(f"episode_start must be >= 0, got {episode_start}")

    if data_format != "lerobot":
        raise ValueError("HUMANO eval only supports --data-format lerobot.")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from humano.datasets.lerobot import get_episode_lerobot

    print("repoid+root", env_id, data_dir)
    data_root = Path(data_dir).expanduser()
    info_path = data_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"LeRobot dataset metadata not found: {info_path}. "
            "Pass --data-dir pointing to a local LeRobot dataset with saved episodes."
        )
    dataset = LeRobotDataset(repo_id=env_id, root=data_dir)
    dataset_size = dataset.num_episodes
    if dataset_size <= 0:
        raise ValueError(f"LeRobot dataset has no episodes: {data_root}")
    render_hz = dataset.meta.fps
    if policy_module_name in ("amo", "twist2"):
        from importlib import import_module

        controller_type = getattr(
            import_module(f"humano.wbc.{controller}"), f"{snake_to_pascal(controller)}Controller"
        )

        render_hz = round(1 / controller_type.control_dt)
        if fast:
            raise ValueError(
                f"{controller} evaluation requires regular camera updates; disable --fast"
            )
    if _is_sonic_token_policy(policy_module_name):
        # LeRobot fps is the GR00T/video/action sampling rate. The SONIC
        # decoder/control loop runs at CONTROL_DT (200 Hz by default), with
        # GR00T tokens held between dataset-fps updates.
        sonic_config["GR00T_TOKEN_TARGET_HZ"] = float(render_hz)
        render_hz = int(round(1.0 / control_dt))
        sonic_config["GR00T_POLICY_STEP_HZ"] = float(render_hz)
    print(f"loaded dataset with {dataset_size} episodes.")

    def get_episode(dataset_obj, idx):
        return get_episode_lerobot(dataset_obj, idx)

    global_episode_indices = list(range(episode_start, dataset_size))
    global_episode_indices = global_episode_indices[:num_episodes]
    episode_indices = global_episode_indices[worker_id::num_workers]
    print(
        f"Evaluating {len(episode_indices)} episodes "
        f"(worker={worker_id}/{num_workers}, requested_total={num_episodes}, "
        f"assigned={len(episode_indices)})."
    )
    report("worker_init", total_episodes=len(episode_indices), status="creating_env")

    raw_env = None
    deploy_proc: ManagedProcess | None = None
    agent = None
    try:
        setup_start_time = time.perf_counter()
        _seed_eval_rng(seed)
        if "isaac" in sim_mode:
            os.environ.setdefault(
                "HUMANO_ISAAC_EXPERIENCE", str(REPO_ROOT / "configs/isaac/humano_eval.kit")
            )
        if (
            "isaac" in sim_mode
            and eval_sonic_mode == "native"
            and _is_sonic_token_policy(policy_module_name)
        ):
            from humano.envs.base_dual_env import _preload_native_runtime

            _preload_native_runtime(wbc=True)
        print(f"Creating environment: {env_id}")
        make_kwargs = dict(
            sim_mode=sim_mode,
            render_hz=render_hz,
            headless=headless,
            success_criteria=success_criteria,
            sonic_config=sonic_config,
        )
        if "isaac" in sim_mode and not (fast and eval_sonic_mode == "native"):
            make_kwargs["isaac_discard_solver_result"] = True
        # Pass fast_mode through to SonicLocoManipEnv if active.
        if fast and eval_sonic_mode == "native" and "isaac" in sim_mode:
            make_kwargs["fast_mode"] = True
        if movement:
            make_kwargs["movement"] = True
        raw_env = gym.make(env_id, **make_kwargs, hand_type=hand_spec.name)
        sonic_env = raw_env.unwrapped  # type: ignore
        task = raw_env.unwrapped.task  # type: ignore[attr-defined]

        robot = task.robot
        viewer_update_every = max(1, int(round(robot.viewer_dt / robot.sim_dt)))

        # ---- Create agent ONCE (before episode loop) ----
        # This avoids DDS participant accumulation from recreating UnitreeSdk2Bridge.
        if policy_module_name == "sonic_policy":
            from humano.policy.client import HttpActionClient

            HttpActionClient(host, port).reset_session()
        policy_module = importlib.import_module(f"humano.policy.{policy_module_name}")
        agent_clazz = getattr(policy_module, f"{snake_to_pascal(policy_module_name)}Agent")

        agent_kwargs: dict[str, Any] = dict(
            sonic_config=sonic_config,
            zmq_port=int(sonic_config.get("SONIC_ZMQ_PORT", 5556)),
        )
        if policy_module_name in ("amo", "twist2"):
            agent_kwargs = {}
        elif eval_sonic_mode == "native":
            agent_kwargs["use_native_sonic"] = True
            agent_kwargs["native_provider"] = os.environ.get("HUMANO_SONIC_PROVIDER", "cpu")
            agent_kwargs["native_model_path"] = str(
                _resolve_deploy_path(
                    Path(deploy_root).expanduser().resolve(),
                    f"{checkpoint}_decoder.onnx",
                )
            )
            agent_kwargs["native_obs_config_path"] = str(
                _resolve_deploy_path(Path(deploy_root).expanduser().resolve(), obs_config)
            )
            agent_kwargs["native_control_dt"] = (
                sonic_control_dt if sonic_control_dt is not None else 0.02
            )
            agent_kwargs["native_env_step_dt"] = _env_step_dt
            agent_kwargs["policy_seed"] = seed
            if fast:
                agent_kwargs["fast_isaac_sync_fn"] = lambda: sonic_env.sync_isaac_and_render()

        agent = agent_clazz(
            task.robot,
            host,
            port,
            **agent_kwargs,
        )

        setup_seconds = time.perf_counter() - setup_start_time
        report(
            "worker_init",
            total_episodes=len(episode_indices),
            status="ready",
            setup_seconds=setup_seconds,
        )

        step_update_every = 5
        stats = defaultdict(bool)
        episode_metrics: dict[str, dict[str, Any]] = {}

        for eps_idx in episode_indices:
            env_conf, episode = get_episode(dataset, eps_idx)  # type: ignore[arg-type]
            (
                env_conf,
                pending_indexed_scene,
                hidden_paths_to_replay,
                finalized_scene_selection,
            ) = _prepare_eval_scene(
                sonic_env,
                env_conf,
                sim_mode,
            )
            task_id = f"episode_{eps_idx}"
            report("episode_start", episode=task_id)
            print(f"[SonicEval] Starting episode {task_id} (eps_idx={eps_idx})")

            # ---- Per-episode SONIC deploy (internal mode only) ----
            if eval_sonic_mode == "internal":
                deploy_interface_val = str(sonic_config.get("INTERFACE") or deploy_interface)
                deploy_log_path_obj = Path(sonic_deploy_log_path) if sonic_deploy_log_path else None
                deploy_proc = _build_deploy_process(
                    deploy_root=Path(deploy_root).expanduser().resolve(),
                    deploy_interface=deploy_interface_val,
                    checkpoint=checkpoint,
                    motion_data=motion_data,
                    obs_config=obs_config,
                    planner=planner,
                    zmq_host=str(sonic_config.get("SONIC_ZMQ_HOST", "localhost")),
                    zmq_port=int(sonic_config.get("SONIC_ZMQ_PORT", 5556)),
                    feedback_port=int(sonic_config.get("FEEDBACK_PORT", feedback_port)),
                    logs_dir=deploy_logs_dir,
                    log_path=deploy_log_path_obj,
                    sonic_control_dt=sonic_control_dt,
                )
                deploy_proc.start()
                if deploy_startup_s > 0:
                    time.sleep(deploy_startup_s)
                deploy_proc.check_started()
                print("[SonicEval] SONIC deploy started")

            if save_video:
                env = raw_env
                if save_exocentric_video:
                    print(
                        "[SonicEval] Recording exocentric video "
                        f"from camera {exocentric_camera!r} as {exocentric_video_key!r}."
                    )
                    env = ExocentricVideoObservation(
                        env,
                        camera_name=exocentric_camera,
                        observation_key=exocentric_video_key,
                    )
                    print("[SonicEval] Exocentric video recording initialized.")
                env = VideoRecorder(
                    env=env,
                    video_folder=eval_output_dir,
                    name_prefix=task_id,
                    framerate=render_hz,
                    write_png=False,
                    async_write=async_video,
                )
                print("[SonicEval] VideoRecorder initialized.")
            else:
                env = raw_env
                print("[SonicEval] Video recording disabled for this episode.")

            print(f"[SonicEval][DEBUG] Starting episode {task_id} (eps_idx={eps_idx})")
            observation, info = _reset_eval_episode(
                env,
                env_conf,
                seed,
                eps_idx,
                sonic_env.task.dr.get_randomizer("camera"),
                use_isaac=sonic_env.isaac is not None,
            )
            if (
                policy_module_name == "act"
                and headless
                and not save_video
                and sonic_env.isaac is not None
            ):
                # ACT consumes only the left image; headless Kit still has a viewport.
                # Apply after every reset because camera render products can be rebuilt.
                from omni.kit.viewport.utility import get_active_viewport

                right_camera = sonic_env.isaac.cameras.get("head_stereo_right")
                if right_camera is not None:
                    right_camera._render_product.hydra_texture.set_updates_enabled(False)
                viewport = get_active_viewport()
                if viewport is not None:
                    viewport.updates_enabled = False
                print(
                    "[SonicEval] ACT headless capture: unused right camera and viewport disabled."
                )
            drift_text = os.environ.get("HUMANO_ACT_START_POSE_DRIFT", "")
            if drift_text:
                if not (eval_sonic_mode == "native" and policy_module_name == "act"):
                    raise ValueError("ACT start-pose compensation requires native WBC history")
                drift = np.fromstring(drift_text, sep=",")
                if drift.shape != (3,):
                    raise ValueError("HUMANO_ACT_START_POSE_DRIFT must be body_dx,body_dy,yaw_rad")
                reset_pose = robot.mjData.qpos[:7].copy()
                robot.mjData.qpos[:7] = _compensated_root_pose(
                    reset_pose, drift[:2], float(drift[2])
                )
                mujoco.mj_forward(robot.mjModel, robot.mjData)
                observation, info = sonic_env._get_obs(), sonic_env._get_info()
                print(
                    "[SonicEval/start_pose_compensation] "
                    f"reset_root_pose={reset_pose.tolist()} "
                    f"compensated_root_pose={robot.mjData.qpos[:7].tolist()} "
                    f"drift_body_xy_yaw={drift.tolist()}",
                    flush=True,
                )
            if policy_module_name in ("amo", "twist2"):
                agent.check_step_dt(sonic_env.control_decimal * robot.mjModel.opt.timestep)
                robot.initialize_ground_standing_pose(log_prefix=f"[{controller}/reset]")
                observation, info = sonic_env._get_obs(), sonic_env._get_info()
            handoff_birth_root_z = None
            if eval_sonic_mode == "native" and _is_sonic_token_policy(policy_module_name):
                handoff_birth_root_z = SONIC_HANDOFF_BIRTH_ROOT_Z
                robot.mjData.qpos[2] = handoff_birth_root_z
                robot.mjData.qvel[:] = 0.0
                robot.mjData.qacc_warmstart[:] = 0.0
                mujoco.mj_forward(robot.mjModel, robot.mjData)
                observation, info = sonic_env._get_obs(), sonic_env._get_info()
            active_indexed_scene = pending_indexed_scene
            if pending_indexed_scene is not None:
                sonic_env.isaac.replace_indexed_background(pending_indexed_scene)
            elif env_conf.get("eval_scene") is not None:
                active_indexed_scene = sonic_env.task.layout.scene
            if active_indexed_scene is not None and finalized_scene_selection:
                if hidden_paths_to_replay:
                    active_indexed_scene.metadata["final_eval_scene_replay"] = (
                        _apply_eval_scene_hidden_paths(
                            sonic_env.isaac,
                            active_indexed_scene,
                            hidden_paths_to_replay,
                        )
                    )
            if active_indexed_scene is not None:
                observation = {**observation, **sonic_env.render()}
            wall_guard = _make_eval_scene_wall_guard(sonic_env, active_indexed_scene)
            wall_guard("reset", 0)
            print(
                f"[SonicEval][DEBUG] Reset episode {task_id} with {len(observation)} observation keys."
            )

            # ---- DDS handshake (external/internal modes) ----
            if eval_sonic_mode in ("external", "internal"):
                dds_wait_start = time.monotonic()
                while not agent._low_cmd_received():
                    agent._publish_low_state(info)
                    if time.monotonic() - dds_wait_start > deploy_startup_s:
                        raise RuntimeError(
                            f"DDS handshake timeout ({deploy_startup_s}s): "
                            "no LowCmd received from SONIC deploy."
                        )
                    time.sleep(0.01)
                deploy_init_s = 5.0
                t0 = time.monotonic()
                while time.monotonic() - t0 < deploy_init_s:
                    agent._publish_low_state(info)
                    time.sleep(0.01)
                print("[SonicEval] DDS handshake complete, SONIC deploy is sending LowCmd.")

            agent.reset(policy_seed=seed)
            from humano.evals.ood_language import resolve_eval_instruction

            instruction_source = (
                os.environ.get("HUMANO_EVAL_INSTRUCTION_SOURCE", "").strip().lower()
            )
            instruction = resolve_eval_instruction(
                task, env_conf, dataset.meta.episodes[eps_idx], instruction_source
            )
            if instruction_source or env_conf.get("ood", {}).get("dimension") == "language":
                print(
                    f"[SonicEval] instruction_source={instruction_source or 'ood'} instruction={instruction!r}"
                )

            if policy_module_name in {"gr00t_n17", "sonic_policy"} and hasattr(
                agent, "_wbc_policy"
            ):
                agent._wbc_policy.lower_body_policy.use_policy_action = True

            # ---- Phase 1: SONIC_BOOTSTRAP_STABILIZE ----
            sim_cnt = 0
            bootstrap_steps = int(
                getattr(
                    agent,
                    "bootstrap_stabilize_steps",
                    sonic_config.get("SONIC_BOOTSTRAP_STABILIZE_STEPS", 500),
                )
            )
            while sim_cnt < bootstrap_steps:
                step_start = time.monotonic()
                if hasattr(agent, "get_bootstrap_stabilize_action"):
                    action = agent.get_bootstrap_stabilize_action(
                        observation, info=info, instruction=instruction
                    )
                elif hasattr(agent, "get_decoupled_stabilize_action"):
                    action = agent.get_decoupled_stabilize_action(
                        observation, info=info, instruction=instruction
                    )
                else:
                    action = agent.get_stabilize_action(
                        observation, info=info, instruction=instruction
                    )
                observation, *_, info = env.step(action)
                wall_guard("bootstrap", sim_cnt + 1)

                if sim_cnt % viewer_update_every == 0:
                    sonic_env.update_viewer()
                    sonic_env.update_reward()

                elapsed = time.monotonic() - step_start
                sleep_time = _env_step_dt - elapsed
                if eval_sonic_mode != "native" and sleep_time > 0:
                    time.sleep(sleep_time)
                sim_cnt += 1

                if sim_cnt == 1 or sim_cnt % 50 == 0:
                    print(
                        f"[SonicEval] bootstrap stabilize step={sim_cnt} "
                        f"pelvis_z={robot.pelvis_z:.3f} "
                        f"pelvis_vz={robot.pelvis_vz:.3f}",
                        flush=True,
                    )

            print(
                f"Bootstrap stabilize complete ({sim_cnt} steps). Starting GR00T prefetch warmup..."
            )

            if "isaac" in sim_mode and not (fast and eval_sonic_mode == "native"):
                fresh_frame = sonic_env.sync_isaac_and_render(reset_accumulation=True)
                observation = {
                    **observation,
                    **fresh_frame,
                    "joint_qpos": fresh_frame["isaac_joint_qpos"],
                }

            if policy_module_name in {"gr00t_n17", "sonic_policy"} and hasattr(
                agent, "_wbc_policy"
            ):
                agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros(
                    (1), dtype=torch.float32
                )

            # ---- Phase 2: GR00T_PREFETCH_WARMUP ----
            warmup_steps = int(
                getattr(
                    agent,
                    "prefetch_warmup_steps",
                    sonic_config.get("GR00T_DECODER_WARMUP_STEPS", 150),
                )
            )
            warmup_cnt = 0
            while warmup_cnt < warmup_steps:
                step_start = time.monotonic()
                if hasattr(agent, "get_prefetch_warmup_action"):
                    action = agent.get_prefetch_warmup_action(
                        observation,
                        info=info,
                        instruction=instruction,
                        prefetch_policy=True,
                    )
                elif hasattr(agent, "get_decoder_warmup_action"):
                    action = agent.get_decoder_warmup_action(
                        observation, info=info, instruction=instruction
                    )
                else:
                    action = agent.get_stabilize_action(
                        observation, info=info, instruction=instruction
                    )
                observation, *_, info = env.step(action)
                wall_guard("prefetch_warmup", warmup_cnt + 1)

                if sim_cnt % viewer_update_every == 0:
                    sonic_env.update_viewer()
                    sonic_env.update_reward()

                elapsed = time.monotonic() - step_start
                sleep_time = _env_step_dt - elapsed
                if eval_sonic_mode != "native" and sleep_time > 0:
                    time.sleep(sleep_time)
                warmup_cnt += 1

                low_cmd_received = agent._low_cmd_received()
                agent_ready = (
                    agent.ready_for_policy() if hasattr(agent, "ready_for_policy") else True
                )
                policy_ready = (
                    agent.policy_action_ready() if hasattr(agent, "policy_action_ready") else True
                )
                if warmup_cnt == 1 or warmup_cnt % 50 == 0:
                    print(
                        f"[SonicEval] prefetch warmup step={warmup_cnt} "
                        f"low_cmd={low_cmd_received} "
                        f"agent_ready={agent_ready} "
                        f"policy_ready={policy_ready}",
                        flush=True,
                    )
                if low_cmd_received and agent_ready and policy_ready:
                    break

            # ---- Safety checks ----
            if policy_module_name not in ("amo", "twist2") and not agent._low_cmd_received():
                raise RuntimeError(
                    "SONIC deploy did not publish LowCmd during prefetch warmup; "
                    "robot cannot be handed over to SONIC control."
                )
            elastic_band = getattr(robot, "elastic_band", None)
            band_enabled = bool(elastic_band is not None and elastic_band.enable)
            if band_enabled:
                raise RuntimeError(
                    "Elastic band is still enabled after prefetch warmup "
                    f"(pelvis_z={robot.pelvis_z:.3f}, pelvis_vz={robot.pelvis_vz:.3f})."
                )
            print(f"Prefetch warmup complete ({warmup_cnt} steps). Engaging GR00T policy now!")

            # ---- Phase 3: Policy Execution ----
            # Metrics are deliberately initialized here so bootstrap stabilization and
            # decoder warmup never contribute to fall, efficiency, or smoothness scores.
            frame_idx = 0
            phase3_start_sim_time = float(robot.mjData.time)
            policy_step_dt = sonic_env.control_decimal * robot.mjModel.opt.timestep
            episode_max_steps = episode_step_limit(max_episode_steps, time_budget, policy_step_dt)
            print(
                f"[SonicEval] Policy budget: {episode_max_steps} steps, "
                f"{episode_max_steps * policy_step_dt:.3f}s simulated time"
            )
            episode_start_time = time.perf_counter()
            fall_detector = FallDetector(sample_dt_s=policy_step_dt)
            phase3_joint_positions: list[np.ndarray] = []
            env_truncated = False
            reached_step_limit = False
            policy_exhausted = False
            policy_wall_collision = False
            episode_over = False
            while not episode_over:
                try:
                    step_start = time.monotonic()
                    action = agent.get_action(observation, info=info, instruction=instruction)
                    observation, reward, terminated, truncated, info = env.step(action)
                    frame_idx += 1
                    wall_guard("policy", frame_idx)
                    episode_over = terminated or truncated
                    env_truncated = bool(truncated)
                    contact_body = core_ground_contact_body(robot.mjModel, robot.mjData)
                    fall_detector, fall_event = fall_detector.update(
                        np.asarray(robot.mjData.xquat[robot.root_body_id], dtype=np.float64),
                        contact_body,
                    )
                    if fall_event is not None:
                        print(
                            "[EvalMetrics][FALL] "
                            f"episode={task_id} reasons={','.join(fall_event.reasons)} "
                            f"pelvis_tilt_deg={fall_event.pelvis_tilt_deg:.2f} "
                            f"hard_tilt_s={fall_event.hard_tilt_seconds:.3f} "
                            f"core_ground_contact_body="
                            f"{fall_event.core_ground_contact_body or 'none'} "
                            f"core_ground_contact_s="
                            f"{fall_event.core_ground_contact_seconds:.3f}",
                            flush=True,
                        )
                    phase3_joint_positions.append(
                        np.asarray(info["proprio"]["body_q"], dtype=np.float64).copy()
                    )
                    if frame_idx >= episode_max_steps:
                        reached_step_limit = True
                        episode_over = True
                    if frame_idx == 1 or frame_idx % step_update_every == 0 or episode_over:
                        report("episode_step", episode=task_id, step=frame_idx)
                    if frame_idx % viewer_update_every == 0:
                        sonic_env.update_viewer()
                        sonic_env.update_reward()
                    sleep_time = _env_step_dt - (time.monotonic() - step_start)
                    if eval_sonic_mode != "native" and sleep_time > 0:
                        time.sleep(sleep_time)
                except StopIteration:
                    policy_exhausted = True
                    episode_over = True
                    print("Episode finished.")
                except EvalSceneWallCollisionError as exc:
                    policy_wall_collision = True
                    episode_over = True
                    print(f"[SonicEval][SAFETY] {exc}", flush=True)

            env_success = bool(raw_env.unwrapped._success)  # type: ignore[attr-defined]
            timed_out = episode_timed_out(
                success=env_success or fall_detector.fall_detected,
                env_truncated=env_truncated,
                reached_step_limit=reached_step_limit,
            )
            is_success, termination_reason = _episode_outcome(
                env_success=env_success,
                fall_detected=fall_detector.fall_detected,
                timed_out=timed_out,
                policy_exhausted=policy_exhausted,
                policy_wall_collision=policy_wall_collision,
            )
            stats[task_id] = is_success
            episode_seconds = time.perf_counter() - episode_start_time
            jerk_score = (
                dimensionless_jerk(np.asarray(phase3_joint_positions), sample_dt_s=policy_step_dt)
                if phase3_joint_positions
                else None
            )
            fall_event = fall_detector.event
            episode_metric = {
                "success": is_success,
                "fall_detected": fall_detector.fall_detected,
                "timed_out": timed_out,
                "termination_reason": termination_reason,
                "fall_reasons": list(fall_event.reasons) if fall_event else [],
                "fall_trigger": (
                    {
                        "pelvis_tilt_deg": fall_event.pelvis_tilt_deg,
                        "hard_tilt_seconds": fall_event.hard_tilt_seconds,
                        "core_ground_contact_seconds": (fall_event.core_ground_contact_seconds),
                        "core_ground_contact_body": (fall_event.core_ground_contact_body),
                    }
                    if fall_event
                    else None
                ),
                "birth_root_z_m": handoff_birth_root_z,
                "startup_simulated_time_s": (sim_cnt + warmup_cnt) * _env_step_dt,
                "phase3_steps": frame_idx,
                "phase3_simulated_time_s": float(robot.mjData.time) - phase3_start_sim_time,
                "completion_time_s": (float(robot.mjData.time) - phase3_start_sim_time)
                if is_success
                else None,
                "env_step_seconds": policy_step_dt,
                "max_episode_steps": episode_max_steps,
                "evaluation_time_budget": time_budget,
                "phase3_wall_time_s": episode_seconds,
                "dimensionless_jerk": jerk_score,
            }
            if controller in ("amo", "twist2"):
                episode_metric.update(
                    wbc_controller=controller,
                    **{
                        ("amo_weights_dir" if controller == "amo" else "twist2_model_path"): str(
                            agent.weights_dir
                        )
                    },
                )
            episode_metrics[task_id] = episode_metric
            _append_eval_metrics_record(eval_dir, {"episode": task_id, **episode_metric})
            _append_eval_stats_line(eval_dir, f"{task_id}: {is_success} \n")
            if hasattr(agent, "stop_stream"):
                agent.stop_stream()
            agent.reset()
            report(
                "episode_end",
                episode=task_id,
                step=frame_idx,
                completed_episodes=len(stats),
                successes=sum(stats.values()),
                episode_seconds=episode_seconds,
                fall_detected=fall_detector.fall_detected,
                success=is_success,
                completion_time_s=episode_metric["completion_time_s"],
                dimensionless_jerk=jerk_score,
                timed_out=timed_out,
                termination_reason=termination_reason,
                steps_per_second=((frame_idx / episode_seconds) if episode_seconds > 0 else 0.0),
            )

            if save_video and isinstance(env, VideoRecorder):
                env.release()

            # ---- Stop per-episode SONIC deploy ----
            if deploy_proc is not None:
                deploy_proc.stop()
                deploy_proc = None
            if eps_idx != episode_indices[-1]:
                time.sleep(2.0)  # Allow DDS discovery to clean up between episodes

        worker_result = (dict(stats), dict(episode_metrics))
        persist_payload("ok", worker_result)
        report("worker_status", status="closing")
        if agent is not None:
            agent.close()
            agent = None
        raw_env.close()
        persist_payload("ok", worker_result)
        report(
            "worker_done",
            completed_episodes=len(stats),
            successes=sum(stats.values()),
        )
        return worker_result
    except BaseException:
        LOGGER.exception("[SonicEval] evaluation failed before cleanup")
        persist_payload("err", traceback.format_exc())
        if raw_env is not None and "isaac" in sim_mode:
            # Kit fast shutdown otherwise exits with 0 before Python reports the exception.
            raw_env.unwrapped.simulation_app.app.post_quit(1)
        raise
    finally:
        if agent is not None:
            try:
                agent.close()
            except Exception:
                pass
            agent = None
        if deploy_proc is not None:
            deploy_proc.stop()
            deploy_proc = None
        if raw_env is not None:
            try:
                raw_env.close()
            except Exception:
                pass


def evaluate_task(
    config: EvalConfig,
    *,
    sonic_config: dict[str, Any] | None = None,
    agent_factory: Callable[..., Any] | None = None,
    show_progress: bool = True,
    env_step_dt: float | None = None,
    sonic_control_dt: float | None = None,
    fast: bool = False,
    hand_type: str = "dex3",
    movement: bool = False,
    seed: int | None = None,
) -> EvalResult:
    if seed is None:
        seed = config.seed
    if config.num_workers <= 0:
        raise ValueError(f"num_workers must be > 0, got {config.num_workers}")
    if config.num_workers != 1:
        raise ValueError(
            "SONIC-token baseline eval requires num_workers=1 because it owns DDS/ZMQ/deploy ports"
        )
    if agent_factory is not None and config.num_workers != 1:
        raise ValueError("custom agent_factory is only supported with num_workers=1")

    if sonic_config is None:
        _hs = get_hand_spec(hand_type)
        sonic_config = _make_sonic_config(hand_spec=_hs)

    eval_dir = config.eval_dir
    env_id = config.env_id
    policy = config.policy
    split = config.split
    host = config.host
    port = config.port
    data_format = config.data_format
    sim_mode = config.sim_mode
    headless = config.headless
    max_episode_steps = config.max_episode_steps
    time_budget = (
        load_task_time_budget(
            config.env_id,
            Path(config.data_dir),
            Path(config.time_budget_file) if config.time_budget_file else None,
        )
        if max_episode_steps is None
        else None
    )
    num_episodes = config.num_episodes
    episode_start = config.episode_start
    data_dir = config.data_dir
    success_criteria = config.success_criteria
    save_video = config.save_video
    num_workers = config.num_workers

    eval_dir_path = Path(config.eval_dir)
    eval_dir_path.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir_path / "eval_metrics.jsonl"
    metrics_path.write_text("")
    log_path = os.environ.get("HUMANO_LOG_FILE") or str(eval_dir_path / "eval_latest.log")
    worker_log_paths = {
        wid: str(eval_dir_path / f"eval_worker_{wid}.log") for wid in range(num_workers)
    }
    for worker_log_path in worker_log_paths.values():
        Path(worker_log_path).write_text("")
    global _PRINT_LOGGER
    configure_logging(log_file=log_path, console=False)
    _PRINT_LOGGER = LOGGER

    terminal_stream = None
    if num_workers == 1 and show_progress:
        terminal_stream = os.fdopen(os.dup(2), "w", buffering=1)
    console = make_console(terminal_stream)
    worker_states = {wid: WorkerProgress() for wid in range(num_workers)}
    _append_eval_stats_line(eval_dir, "================\n")
    _append_eval_stats_line(eval_dir, f"run: {env_id} - {policy}\n")

    async_video = config.async_video
    save_exocentric_video = config.save_exocentric_video
    exocentric_camera = config.exocentric_camera
    exocentric_video_key = config.exocentric_video_key
    sonic_deploy_log_path = (
        None if _get_sonic_mode() != "internal" else str(eval_dir_path / "sonic_deploy.log")
    )

    worker_kwargs = dict(
        wbc_controller=config.wbc_controller,
        env_id=env_id,
        policy=policy,
        split=split,
        host=host,
        port=port,
        data_format=data_format,
        sim_mode=sim_mode,
        headless=headless,
        eval_dir=eval_dir,
        max_episode_steps=max_episode_steps,
        time_budget=time_budget,
        num_episodes=num_episodes,
        episode_start=episode_start,
        data_dir=data_dir,
        success_criteria=success_criteria,
        save_video=save_video,
        sonic_config=sonic_config,
        async_video=async_video,
        save_exocentric_video=save_exocentric_video,
        exocentric_camera=exocentric_camera,
        exocentric_video_key=exocentric_video_key,
        sonic_deploy_log_path=sonic_deploy_log_path,
        env_step_dt=env_step_dt,
        sonic_control_dt=sonic_control_dt,
        fast=fast,
        hand_type=hand_type,
        movement=movement,
        seed=seed,
    )

    console.print(f"Writing eval logs to [bold]{log_path}[/bold]")

    if num_workers == 1:
        stats: dict[str, bool]
        episode_metrics: dict[str, dict[str, Any]]
        if show_progress:
            try:
                with (
                    Live(
                        render_progress(env_id, policy, worker_states, log_path),
                        console=console,
                        refresh_per_second=4,
                    ) as live,
                ):

                    def report(payload: dict[str, Any]) -> None:
                        update_progress(worker_states, 0, payload)
                        live.update(
                            render_progress(env_id, policy, worker_states, log_path),
                            refresh=False,
                        )

                    with route_prints_to_logger(LOGGER), route_stderr_to_logger(LOGGER):
                        try:
                            stats, episode_metrics = _run_eval_worker(
                                **worker_kwargs,
                                worker_id=0,
                                num_workers=1,
                                progress_reporter=report,
                            )
                        except BaseException as exc:
                            LOGGER.exception("[SonicEval] single worker failed: %r", exc)
                            raise
            finally:
                restore_cursor(console)
                if terminal_stream is not None:
                    terminal_stream.close()
        else:
            with route_prints_to_logger(LOGGER), route_stderr_to_logger(LOGGER):
                stats, episode_metrics = _run_eval_worker(
                    **worker_kwargs,
                    worker_id=0,
                    num_workers=1,
                )
    else:
        ctx = mp.get_context("spawn")
        progress_readers: dict[int, Any] = {}
        procs: list[mp.Process] = []
        result_dir = Path(tempfile.mkdtemp(prefix="humano_eval_workers_"))

        try:
            with Live(
                render_progress(env_id, policy, worker_states, log_path),
                console=console,
                refresh_per_second=4,
                auto_refresh=show_progress,
            ) as live:
                for wid in range(num_workers):
                    worker_result_path = str(result_dir / f"worker_{wid}.pkl")
                    recv_conn, send_conn = ctx.Pipe(duplex=False)
                    p = ctx.Process(
                        target=_run_eval_worker_entry,
                        args=(
                            worker_result_path,
                            wid,
                            num_workers,
                            worker_kwargs,
                            worker_log_paths[wid],
                            send_conn,
                        ),
                        name=f"eval-worker-{wid}",
                    )
                    p.start()
                    send_conn.close()
                    progress_readers[wid] = recv_conn
                    procs.append(p)

                while any(p.is_alive() for p in procs) or progress_readers:
                    ready = (
                        wait(list(progress_readers.values()), timeout=0.2)
                        if progress_readers
                        else []
                    )
                    for conn in ready:
                        try:
                            wid, payload = conn.recv()
                        except EOFError:
                            for key, value in list(progress_readers.items()):
                                if value is conn:
                                    value.close()
                                    del progress_readers[key]
                                    break
                            continue
                        update_progress(worker_states, wid, payload)
                        if show_progress:
                            live.update(
                                render_progress(env_id, policy, worker_states, log_path),
                                refresh=False,
                            )
        finally:
            restore_cursor(console)
            if terminal_stream is not None:
                terminal_stream.close()
            for conn in progress_readers.values():
                try:
                    conn.close()
                except Exception:
                    pass

        stats = {}
        episode_metrics = {}
        results_by_worker: dict[int, tuple[dict[str, bool], dict[str, dict[str, Any]]]] = {}
        errors_by_worker: dict[int, str] = {}
        failed_exit: list[tuple[int, int]] = []

        for wid, p in enumerate(procs):
            worker_result_path = result_dir / f"worker_{wid}.pkl"
            p.join()
            exit_code = p.exitcode if p.exitcode is not None else 999
            if exit_code != 0:
                failed_exit.append((wid, exit_code))
            if worker_result_path.exists():
                try:
                    with open(worker_result_path, "rb") as f:
                        kind, got_wid, payload = pickle.load(f)
                    if got_wid != wid:
                        errors_by_worker[wid] = f"mismatched worker payload id={got_wid}"
                    elif kind == "ok":
                        results_by_worker[wid] = payload
                    else:
                        errors_by_worker[wid] = payload
                except Exception:
                    errors_by_worker[wid] = traceback.format_exc()

        for wid in sorted(results_by_worker):
            worker_stats, worker_metrics = results_by_worker[wid]
            overlap = set(worker_stats).intersection(stats)
            if overlap:
                raise RuntimeError(f"Duplicate episode ids across workers: {sorted(overlap)}")
            stats.update(worker_stats)
            episode_metrics.update(worker_metrics)

        if errors_by_worker or failed_exit:
            summary = []
            if errors_by_worker:
                summary.append(f"python_errors={sorted(errors_by_worker)}")
            if failed_exit:
                summary.append(f"nonzero_exit={failed_exit}")
            for wid in sorted(errors_by_worker):
                console.print(
                    f"[red]worker {wid} traceback[/red]\n{errors_by_worker[wid]}\n"
                    f"[dim]log: {worker_log_paths.get(wid, log_path)}[/dim]"
                )
            if failed_exit:
                console.print("[yellow]worker native exits[/yellow]")
                for wid, exit_code in failed_exit:
                    console.print(
                        f"[yellow]worker {wid} exit {exit_code}[/yellow] "
                        f"[dim]log: {worker_log_paths.get(wid, log_path)}[/dim]"
                    )
            console.print(
                "[red]Parallel eval worker failure[/red] "
                + "("
                + ", ".join(summary)
                + f"). See {log_path}"
            )
            raise typer.Exit(code=1)

        missing = sorted(set(range(num_workers)) - set(results_by_worker) - set(errors_by_worker))
        if missing:
            console.print(
                "[red]Parallel eval worker missing result payload[/red] "
                f"(workers={missing}, exits={failed_exit}). See {log_path}"
            )
            raise typer.Exit(code=1)

    # The live/progress console may have been bound to a temporary stream used
    # during worker stdio redirection. Rebind before summary prints so teardown
    # cannot leave Rich writing to a closed file object.
    console = make_console()
    if not stats:
        console.print(f"[red]Eval produced no episode results[/red]. See {log_path}")
        raise typer.Exit(code=1)
    sr = sum(stats.values()) / len(stats)
    metric_summary = summarize_episode_metrics(episode_metrics)
    expected_episode_ids = tuple(
        f"episode_{index}" for index in range(episode_start, episode_start + num_episodes)
    )
    evaluation_health = summarize_evaluation_health(
        expected_episode_ids=expected_episode_ids,
        episode_metrics=episode_metrics,
    )
    evaluation_health_path = eval_dir_path / "evaluation_health.json"
    evaluation_health_path.write_text(
        json.dumps(evaluation_health, indent=2, sort_keys=True) + "\n"
    )
    console.print(f"Success rate {env_id} - {policy}: {sr:.2%}")
    console.print(f"Fall rate {env_id} - {policy}: {metric_summary['fall_rate']:.2%}")
    console.print(f"Timeout rate {env_id} - {policy}: {metric_summary['timeout_rate']:.2%}")
    if metric_summary["median_success_completion_time_s"] is not None:
        console.print(
            "Median successful completion time: "
            f"{metric_summary['median_success_completion_time_s']:.3f}s"
        )
    if metric_summary["median_success_dimensionless_jerk"] is not None:
        console.print(
            "Median successful dimensionless jerk: "
            f"{metric_summary['median_success_dimensionless_jerk']:.6g}"
        )
    console.print(
        "Evaluation health: "
        f"completed={evaluation_health['completed_episodes']}/"
        f"{evaluation_health['requested_episodes']} "
        f"missing={evaluation_health['missing_episodes']} "
        f"complete={evaluation_health['is_complete']}"
    )
    console.print(f"Evaluation health diagnostics: {evaluation_health_path}")
    console.print(f"Eval log: {log_path}")

    _append_eval_stats_line(eval_dir, f"success rate: {sr:.2f} \n")
    _append_eval_stats_line(eval_dir, f"fall rate: {metric_summary['fall_rate']:.6f} \n")
    _append_eval_stats_line(eval_dir, f"timeout rate: {metric_summary['timeout_rate']:.6f} \n")
    _append_eval_stats_line(
        eval_dir,
        "median successful completion time s: "
        f"{metric_summary['median_success_completion_time_s']} \n",
    )
    _append_eval_stats_line(
        eval_dir,
        "median successful dimensionless jerk: "
        f"{metric_summary['median_success_dimensionless_jerk']} \n",
    )
    return EvalResult(
        env_id=env_id,
        policy=policy,
        split=split,
        stats=dict(stats),
        success_rate=sr,
        episode_metrics=dict(episode_metrics),
        fall_rate=metric_summary["fall_rate"],
        timeout_rate=metric_summary["timeout_rate"],
        median_success_completion_time_s=metric_summary["median_success_completion_time_s"],
        median_success_dimensionless_jerk=metric_summary["median_success_dimensionless_jerk"],
        metrics_path=str(metrics_path),
        evaluation_health=dict(evaluation_health),
        evaluation_health_path=str(evaluation_health_path),
        log_path=log_path,
        eval_dir=eval_dir,
    )


def main(
    env_id: Annotated[str, typer.Argument()],
    policy: Annotated[str, typer.Argument()],
    split: Annotated[str, typer.Argument()] = "eval",
    host: Annotated[str, typer.Option()] = "172.17.0.1",
    port: Annotated[int, typer.Option()] = 21000,
    data_format: Annotated[str, typer.Option()] = "lerobot",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = False,
    eval_dir: Annotated[str, typer.Option()] = "data/evals_sonic",
    max_episode_steps: Annotated[int | None, typer.Option(min=1)] = None,
    time_budget_file: Annotated[Path | None, typer.Option()] = None,
    num_episodes: Annotated[int, typer.Option()] = DEFAULT_NUM_EPISODES,
    episode_start: Annotated[int, typer.Option()] = 0,
    data_dir: Annotated[str, typer.Option()] = "data/datagen",
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
    hand_type: Annotated[str, typer.Option()] = "dex3",
    wbc_controller: Annotated[str | None, typer.Option("--wbc-controller")] = None,
    movement: Annotated[bool, typer.Option("--movement/--no-movement")] = False,
    seed: Annotated[int | None, typer.Option()] = None,
):
    evaluate_task(
        EvalConfig(
            env_id=env_id,
            policy=policy,
            wbc_controller=wbc_controller,
            split=split,
            host=host,
            port=port,
            data_format=data_format,
            sim_mode=sim_mode,
            headless=headless,
            eval_dir=eval_dir,
            max_episode_steps=max_episode_steps,
            time_budget_file=str(time_budget_file) if time_budget_file else None,
            num_episodes=num_episodes,
            episode_start=episode_start,
            data_dir=data_dir,
            success_criteria=success_criteria,
            save_video=save_video,
            num_workers=num_workers,
            save_exocentric_video=save_exocentric_video,
            exocentric_camera=exocentric_camera,
            exocentric_video_key=exocentric_video_key,
            async_video=async_video,
        ),
        env_step_dt=env_step_dt,
        sonic_control_dt=sonic_control_dt,
        fast=fast,
        hand_type=hand_type,
        movement=movement,
        seed=seed,
    )


def typer_main():
    configure_logging()
    try:
        typer.run(main)
    except Exception:
        LOGGER.exception("[SonicEval] Fatal error")
        raise


if __name__ == "__main__":
    typer_main()
