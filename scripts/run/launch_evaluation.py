"""
All-in-one tmux launcher for HUMANO SONIC evaluation.

Starts the processes for HUMANO SONIC evaluation in a single tmux window:

    Window 0 - evaluation:
    ┌───────────────────────────┬───────────────────────────┐
    │ Pane 0: HUMANO EvalRunner │ Pane 1: gr00t_serve       │
    └───────────────────────────┴───────────────────────────┘

    When --sonic-mode external:
    ┌───────────────────────────┬───────────────────────────┐
    │ Pane 0: HUMANO EvalRunner │ Pane 1: gr00t_serve       │
    │                           ├───────────────────────────┤
    │                           │ Pane 2: SONIC deploy      │
    └───────────────────────────┴───────────────────────────┘

Prerequisites:
    - tmux installed (sudo apt install tmux)
    - HUMANO environment set up (pip install -e .)
    - Policy checkpoint in --run-dir

Usage (from the repo root):
    python scripts/run/launch_evaluation.py --run-dir /path/to/run --data-dir /path/to/data
    python scripts/run/launch_evaluation.py --run-dir /path/to/run --data-dir /path/to/data --sonic-mode native --sonic-provider gpu
    python scripts/run/launch_evaluation.py --run-dir /path/to/run --data-dir /path/to/data --dry-run

    SONIC mode: native (in-process SonicController); --sonic-provider cpu|gpu selects the device. Legacy internal/external modes are disabled.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from humano.evals.api import DEFAULT_NUM_EPISODES, resolve_wbc_controller

REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_gr00t_root() -> Path:
    configured_root = os.environ.get("HUMANO_GR00T_ROOT")
    if not configured_root:
        return REPO_ROOT / "baselines" / "gr00t_n17"
    root = Path(configured_root).expanduser()
    return root if root.is_absolute() else REPO_ROOT / root


DEFAULT_GR00T_ROOT = _default_gr00t_root()
NPP_LIBRARY_ENV = "HUMANO_NPP_LIBRARY_DIR"
NPP_REQUIRED_LIBRARY = "libnppicc.so.12"

SESSION_NAME = "humano_eval"
GR00T_N17_POLICIES = {
    "gr00t_n17",
    "gr00tn17",
    "gr00t_n1d7",
    "gr00tn1d7",
}
LEROBOT_BASELINE_POLICIES = {
    "act",
    "diffusion",
    "pi05",
    "pi_05",
    "flow_matching",
}
PSI0_POLICIES = {"psi0"}
SONIC_TOKEN_POLICIES = GR00T_N17_POLICIES | LEROBOT_BASELINE_POLICIES | PSI0_POLICIES | {"remote"}


# ---------------------------------------------------------------------------
# RUNTIME BOOTSTRAP
# ---------------------------------------------------------------------------
def _dedupe_existing_dirs(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        expanded = path.expanduser()
        if not expanded.is_dir():
            continue
        key = str(expanded.resolve())
        if key in seen:
            continue
        seen.add(key)
        result.append(expanded)
    return result


def _npp_candidate_dirs() -> list[Path]:
    """Return portable NPP runtime candidates in preference order."""
    candidates: list[Path] = []

    override = os.environ.get(NPP_LIBRARY_ENV, "")
    candidates.extend(Path(part) for part in override.split(os.pathsep) if part)

    # NVIDIA Python wheels install shared libraries below site-packages/nvidia.
    for python_path in sys.path:
        if python_path:
            candidates.append(Path(python_path) / "nvidia" / "npp" / "lib")

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib")

    cuda_roots = [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDAToolkit_ROOT"),
        "/usr/local/cuda",
    ]
    for root_value in cuda_roots:
        if not root_value:
            continue
        root = Path(root_value)
        candidates.extend(
            (root / "lib64", root / "targets" / f"{platform.machine()}-linux" / "lib")
        )

    # Conda package caches are useful when libnpp is present but not installed
    # into the currently active environment.
    conda_roots = [
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
        Path.home() / "miniforge3",
        Path.home() / "mambaforge",
    ]
    for conda_root in conda_roots:
        candidates.extend(sorted((conda_root / "pkgs").glob("libnpp-*/lib"), reverse=True))

    isaac_roots = [
        Path(value)
        for value in (
            os.environ.get("ISAACSIM_PATH"),
            os.environ.get("ISAAC_SIM_PATH"),
        )
        if value
    ]
    isaac_roots.extend(sorted(Path.home().glob("isaacsim*"), reverse=True))
    for isaac_root in isaac_roots:
        candidates.extend(
            sorted(
                (isaac_root / "extscache").glob("omni.sensors.nv.camera-*/bin"),
                reverse=True,
            )
        )

    return _dedupe_existing_dirs(candidates)


def _find_npp_library_dirs() -> list[Path]:
    return [path for path in _npp_candidate_dirs() if (path / NPP_REQUIRED_LIBRARY).exists()]


def _ensure_torchcodec_runtime_libs() -> None:
    """Re-exec with NPP / TorchCodec runtime libs on LD_LIBRARY_PATH if needed."""
    lib_dirs = [str(path) for path in _find_npp_library_dirs()]
    if not lib_dirs:
        return

    current = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    missing = [path for path in lib_dirs if path not in current]
    if not missing:
        return

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(missing + current)
    if env.get("HUMANO_EVAL_TORCHCODEC_LIBS_READY") == "1":
        os.environ.update(env)
        return
    env["HUMANO_EVAL_TORCHCODEC_LIBS_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------
def _quote(value: str | Path | int | float) -> str:
    return shlex.quote(str(value))


def _join_argv(argv: list[str]) -> str:
    return " ".join(_quote(part) for part in argv)


def _append_flag(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _normalized_policy(policy: str) -> str:
    return policy.lower().replace("-", "_").replace(".", "")


def _server_module_for_policy(policy: str) -> str:
    normalized = _normalized_policy(policy)
    if normalized in GR00T_N17_POLICIES:
        return "baselines.gr00t_n17.deploy.gr00t_serve"
    if normalized in PSI0_POLICIES:
        return "baselines.psi0.deploy.psi0_serve"
    if normalized == "pi_05":
        normalized = "pi05"
    modules = {
        "act": "baselines.lerobot_runtime.deploy.act_serve",
        "diffusion": "baselines.lerobot_runtime.deploy.diffusion_serve",
        "pi05": "baselines.lerobot_runtime.deploy.pi05_serve",
        "flow_matching": "baselines.lerobot_runtime.deploy.flow_matching_serve",
    }
    try:
        return modules[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported policy: {policy!r}") from exc


def _pick_free_port(start: int = 22085, end: int = 22999) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found in range {start}-{end}")


def _net_flag(iface: str, name: str) -> int | None:
    try:
        return int((Path("/sys/class/net") / iface / name).read_text().strip(), 0)
    except Exception:
        return None


def _pick_sonic_interface(configured: str = "") -> str:
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


# ---------------------------------------------------------------------------
# ENVIRONMENT BUILDERS
# ---------------------------------------------------------------------------
def _humano_env_prefix(repo_root: Path) -> str:
    """Build PYTHONPATH + LD_LIBRARY_PATH + TORCH_EXTENSIONS_DIR exports."""
    joined = os.pathsep.join((str(repo_root), str(repo_root / "src")))
    torch_ext_dir = repo_root / ".cache" / "torch_extensions"
    library_dirs = [str(path) for path in _find_npp_library_dirs()]
    try:
        import torch as _torch

        torch_lib = str(Path(_torch.__file__).parent / "lib")
        library_dirs.insert(0, torch_lib)
    except ImportError:
        pass
    runtime_ld = ""
    if library_dirs:
        ld_prefix = ":".join(_quote(path) for path in library_dirs)
        runtime_ld = f"LD_LIBRARY_PATH={ld_prefix}:${{LD_LIBRARY_PATH:-}} "
    return (
        f"PYTHONPATH={_quote(joined)}:${{PYTHONPATH:-}} "
        f"{runtime_ld}"
        f"TORCH_EXTENSIONS_DIR=${{TORCH_EXTENSIONS_DIR:-{_quote(str(torch_ext_dir))}}}"
    )


def _sonic_env_prefix(args: argparse.Namespace, interface: str, *, sonic_mode: str) -> str:
    """Build HUMANO SONIC environment variable exports."""
    values = {
        "HUMANO_SONIC_INTERFACE": interface,
        "SONIC_ZMQ_PORT": args.sonic_deploy_zmq_port,
        "SONIC_ZMQ_OUT_PORT": args.sonic_deploy_feedback_port,
        "HUMANO_SONIC_MODE": sonic_mode,
        "HUMANO_SONIC_PROVIDER": args.sonic_provider,
    }
    return " ".join(f"{key}={_quote(value)}" for key, value in values.items())


def _shell_cmd(repo_root: Path, body: str, *, extra_env: str = "") -> str:
    """Wrap a shell command with working directory and environment exports."""
    exports = [_humano_env_prefix(repo_root)]
    instruction_source = os.environ.get("HUMANO_EVAL_INSTRUCTION_SOURCE", "").strip()
    if instruction_source:
        exports.append(f"HUMANO_EVAL_INSTRUCTION_SOURCE={_quote(instruction_source)}")
    for key in ("HUMANO_AMO_WEIGHTS_DIR", "HUMANO_TWIST2_MODEL_PATH"):
        if os.environ.get(key):
            exports.append(f"{key}={_quote(os.environ[key])}")
    if extra_env:
        exports.append(extra_env)
    return f"cd {_quote(repo_root)} && {' '.join(exports)} {body}"


def _isolated_server_shell_cmd(
    repo_root: Path, body: str, *, virtual_env: Path | None = None, extra_env: str = ""
) -> str:
    """Run an external policy interpreter without shared-simulator Torch libraries.

    Isaac evaluation stays in the shared environment, while a policy supplied
    through --server-python owns its Python and CUDA runtime. In particular,
    PI05 uses a Transformers 5 / Torch 2.11 environment that must not load the
    shared Python 3.10 Torch libraries through LD_LIBRARY_PATH.
    """
    exports = [
        f"PYTHONPATH={_quote(os.pathsep.join((str(repo_root), str(repo_root / 'src'))))}:$"
        "{PYTHONPATH:-}",
        f"TORCH_EXTENSIONS_DIR=$"
        "{TORCH_EXTENSIONS_DIR:-"
        f"{_quote(str(repo_root / 'tmp' / 'kyy_baseline_integration' / 'torch_extensions'))}"
        "}",
    ]
    if virtual_env is not None:
        exports.append(f"VIRTUAL_ENV={_quote(virtual_env)}")
    instruction_source = os.environ.get("HUMANO_EVAL_INSTRUCTION_SOURCE", "").strip()
    if instruction_source:
        exports.append(f"HUMANO_EVAL_INSTRUCTION_SOURCE={_quote(instruction_source)}")
    if extra_env:
        exports.append(extra_env)
    return f"cd {_quote(repo_root)} && env -u LD_LIBRARY_PATH {' '.join(exports)} {body}"


def _with_timestamped_log(cmd: str, log_path: Path) -> str:
    """Pipe a command through awk timestamping and tee to a log file."""
    log_path = log_path.expanduser()
    awk_script = r'{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'
    return (
        f"mkdir -p {_quote(log_path.parent)} && "
        f"truncate -s 0 {_quote(log_path)} && "
        f"set -o pipefail && "
        f"stdbuf -oL -eL bash -lc {_quote(cmd)} 2>&1 | "
        f"awk {_quote(awk_script)} | "
        f"tee -a {_quote(log_path)}"
    )


def _with_python_log_file(cmd: str, log_path: Path) -> str:
    log_path = log_path.expanduser()
    return (
        f"mkdir -p {_quote(log_path.parent)} && "
        f"truncate -s 0 {_quote(log_path)} && "
        f"HUMANO_LOG_FILE={_quote(str(log_path))} "
        f"bash -lc {_quote(cmd)}"
    )


# ---------------------------------------------------------------------------
# COMMAND BUILDERS
# ---------------------------------------------------------------------------
def _wait_for_policy_server_health(
    *,
    host: str,
    port: int,
    tries: int,
    sleep_s: float,
) -> None:
    """Wait until the policy server has loaded its checkpoint and serves /health."""
    if tries < 1:
        raise ValueError("wait tries must be at least 1")
    if sleep_s <= 0:
        raise ValueError("wait sleep interval must be positive")

    endpoint = f"http://{host}:{port}/health"
    last_error = "connection not attempted"
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(
                endpoint,
                timeout=min(5.0, max(0.5, sleep_s * 10)),
            ) as response:
                if response.status == 200:
                    print(
                        f"Policy server is healthy after {attempt} probe(s): {endpoint}",
                        flush=True,
                    )
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        if attempt < tries:
            time.sleep(sleep_s)

    raise RuntimeError(
        f"policy server did not become healthy at {endpoint} after {tries} probe(s): {last_error}"
    )


def _parse_config_args(parser, argv=None):
    """Read existing launcher options from JSON; explicit CLI options win."""
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    config, _ = probe.parse_known_args(argv)
    if config.config:
        try:
            values = json.loads(Path(config.config).read_text())
        except (OSError, ValueError) as exc:
            parser.error(f"cannot read config: {exc}")
        if not isinstance(values, dict):
            parser.error("config must be a JSON object")
        actions = {a.dest: a for a in parser._actions}
        for key, value in values.items():
            if key not in actions or key in {"help", "config"}:
                parser.error(f"unknown config option: {key}")
            action = actions[key]
            if value is None:
                parser.error(f"{key} cannot be null; omit it to use the default")
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                if not isinstance(value, bool):
                    parser.error(f"{key} must be boolean")
            elif action.type:
                if isinstance(value, bool) or (action.type is int and isinstance(value, float)):
                    parser.error(f"invalid value for {key}")
                try:
                    values[key] = action.type(value)
                except (TypeError, ValueError):
                    parser.error(f"invalid value for {key}")
            elif not isinstance(value, str):
                parser.error(f"{key} must be a string")
            if action.choices and values[key] not in action.choices:
                parser.error(f"invalid choice for {key}: {value}")
        parser.set_defaults(**values)
    return parser.parse_args(argv)


def _build_server_argv(args: argparse.Namespace, port: int) -> list[str]:
    """Build the policy server command for an integrated baseline."""
    policy_name = _normalized_policy(args.policy)
    is_gr00t_n17 = policy_name in GR00T_N17_POLICIES
    is_psi0 = policy_name in PSI0_POLICIES
    server_module = getattr(args, "server_module", None) or _server_module_for_policy(policy_name)
    server_python = getattr(args, "server_python", None)
    cmd = [
        server_python or sys.executable,
        "-m",
        server_module,
        "--host",
        args.server_host,
        "--port",
        str(port),
        "--policy",
        args.policy,
        "--run-dir",
        str(Path(args.run_dir) / "pretrained_model")
        if policy_name in LEROBOT_BASELINE_POLICIES
        and not (Path(args.run_dir) / "config.json").is_file()
        and (Path(args.run_dir) / "pretrained_model" / "config.json").is_file()
        else args.run_dir,
        "--device",
        args.device,
    ]
    if policy_name in LEROBOT_BASELINE_POLICIES or is_psi0:
        cmd.extend(["--hand-type", getattr(args, "hand_type", "dex3")])
    if policy_name == "pi05" and getattr(args, "tokenizer_path", None):
        cmd.extend(["--tokenizer-path", args.tokenizer_path])
    if args.ckpt_step is not None:
        cmd.extend(["--ckpt-step", str(args.ckpt_step)])
    if is_gr00t_n17:
        cmd.extend(["--embodiment-tag", args.embodiment_tag])
        if args.gr00t_root:
            cmd.extend(["--gr00t-root", args.gr00t_root])
    if is_gr00t_n17 or is_psi0:
        _append_flag(cmd, "--action-exec-horizon", getattr(args, "action_exec_horizon", None))
        if getattr(args, "rtc", False):
            cmd.append("--rtc")
    if is_gr00t_n17 or policy_name == "remote":
        _append_flag(cmd, "--seed", getattr(args, "seed", None))
        if is_gr00t_n17 and getattr(args, "deterministic", False):
            cmd.append("--deterministic")
    if policy_name in LEROBOT_BASELINE_POLICIES and getattr(args, "mock_policy", False):
        cmd.append("--mock-policy")
    return cmd


def _build_eval_cmd(args: argparse.Namespace, port: int) -> str:
    """Build a command that calls the requested HUMANO evaluation module.

    ``eval_task.py`` handles SONIC deploy lifecycle, parallel workers, progress
    UI, and video recording itself.  The eval runs in-process inside the tmux
    pane, connecting to the policy server in the adjacent pane.
    """
    if args.hand_type == "brainco" and args.sim_mode not in {"mujoco", "mujoco_isaac"}:
        raise ValueError("BrainCo requires MuJoCo physics; use --sim-mode mujoco or mujoco_isaac")
    cmd = [sys.executable, "-m"]
    if args.suite_config:
        cmd.extend(
            [
                "humano.cli.eval_suite",
                "--suite-config",
                args.suite_config,
                "--data-root",
                args.data_root,
                "--policy",
                args.policy,
            ]
        )
    else:
        cmd.extend(["humano.cli.eval_task", args.env_id, args.policy])
    cmd.extend(
        [
            "--host",
            args.host,
            "--port",
            str(port),
            "--data-format",
            args.data_format,
            "--sim-mode",
            args.sim_mode,
            "--hand-type",
            args.hand_type,
            "--eval-dir",
            args.eval_dir,
            "--num-episodes",
            str(args.num_episodes),
            "--episode-start",
            str(args.episode_start),
            "--num-workers",
            str(args.num_workers),
            "--success-criteria",
            str(args.success_criteria),
        ]
    )
    _append_flag(cmd, "--wbc-controller", getattr(args, "wbc_controller", None))
    if args.data_dir:
        cmd.extend(["--data-dir", args.data_dir])

    # Boolean flags — pass every value explicitly so we don't rely on CLI defaults
    cmd.append("--headless" if args.headless else "--no-headless")
    cmd.append("--save-video" if args.save_video else "--no-save-video")
    cmd.append(
        "--save-exocentric-video"
        if getattr(args, "save_exocentric_video", False)
        else "--no-save-exocentric-video"
    )
    cmd.append("--async-video" if getattr(args, "async_video", True) else "--no-async-video")
    cmd.append("--fast" if getattr(args, "fast", False) else "--no-fast")

    # Optional flags (only appended when non-None / non-default)
    _append_flag(cmd, "--max-episode-steps", args.max_episode_steps)
    _append_flag(cmd, "--exocentric-camera", getattr(args, "exocentric_camera", None))
    _append_flag(cmd, "--exocentric-video-key", getattr(args, "exocentric_video_key", None))
    _append_flag(cmd, "--env-step-dt", getattr(args, "env_step_dt", None))
    _append_flag(cmd, "--sonic-control-dt", getattr(args, "sonic_control_dt", None))
    _append_flag(cmd, "--seed", getattr(args, "seed", None))
    return _join_argv(cmd)


def _lane_env(args: argparse.Namespace, *, evaluator: bool = False) -> str:
    values = []
    if args.deterministic and args.seed is not None:
        values.extend(["CUBLAS_WORKSPACE_CONFIG=:4096:8", f"PYTHONHASHSEED={_quote(args.seed)}"])
    if args.gpu_id is not None:
        values.extend(["CUDA_DEVICE_ORDER=PCI_BUS_ID", f"CUDA_VISIBLE_DEVICES={args.gpu_id}"])
        if evaluator:
            values.append(f"HUMANO_EVAL_GPU_ID={args.gpu_id}")
            kit_map = os.environ.get("HUMANO_ISAAC_GPU_MAP")
            if kit_map:
                values.append(f"HUMANO_ISAAC_GPU_MAP={_quote(kit_map)}")
    return " ".join(values)


def _server_command(args: argparse.Namespace) -> str:
    wrap = _isolated_server_shell_cmd if args.server_python else _shell_cmd
    return wrap(
        REPO_ROOT, _join_argv(_build_server_argv(args, args.port)), extra_env=_lane_env(args)
    )


def _parallel_commands(args: argparse.Namespace) -> tuple[str, str]:
    interface = _pick_sonic_interface(args.sonic_deploy_interface)
    server = _server_command(args)
    evaluation = _shell_cmd(
        REPO_ROOT,
        _build_eval_cmd(args, args.port),
        extra_env=" ".join(
            (
                _lane_env(args, evaluator=True),
                _sonic_env_prefix(args, interface, sonic_mode=args.sonic_mode),
                f"HUMANO_LOG_FILE={_quote(str(Path(args.eval_dir) / 'eval_latest.log'))}",
            )
        ),
    )
    return server, evaluation


def _build_sonic_deploy_cmd(args: argparse.Namespace, interface: str) -> str:
    """Build the SONIC deploy C++ binary launch command."""
    cmd = [
        "./target/release/g1_deploy_onnx_ref",
        interface,
        f"{args.sonic_deploy_checkpoint}_decoder.onnx",
        args.sonic_deploy_motion_data,
        "--obs-config",
        args.sonic_deploy_obs_config,
        "--encoder-file",
        f"{args.sonic_deploy_checkpoint}_encoder.onnx",
        "--planner-file",
        args.sonic_deploy_planner,
        "--input-type",
        "zmq_manager",
        "--output-type",
        "all",
        "--zmq-host",
        args.sonic_deploy_zmq_host,
        "--zmq-port",
        str(args.sonic_deploy_zmq_port),
        "--zmq-out-port",
        str(args.sonic_deploy_feedback_port),
        "--zmq-verbose",
        "--disable-crc-check",
    ]
    _append_flag(cmd, "--logs-dir", args.sonic_deploy_logs_dir)
    if args.sonic_control_dt is not None:
        cmd.extend(["--control-dt", f"{args.sonic_control_dt:.4f}"])
    return (
        f"cd {_quote(Path(args.sonic_deploy_root).expanduser())} && "
        f"source scripts/setup_env.sh && "
        f"exec {_join_argv(cmd)}"
    )


# ---------------------------------------------------------------------------
# PREREQUISITE CHECKS
# ---------------------------------------------------------------------------
def _resolve_deploy_path(deploy_root: Path, path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return deploy_root / path


def _check_deploy_file(errors: list[str], deploy_root: Path, path_like: str, name: str) -> None:
    path = _resolve_deploy_path(deploy_root, path_like)
    if not path.is_file():
        errors.append(f"{name} not found: {path}")


def _check_deploy_dir(errors: list[str], deploy_root: Path, path_like: str, name: str) -> None:
    path = _resolve_deploy_path(deploy_root, path_like)
    if not path.is_dir():
        errors.append(f"{name} not found: {path}")


def _check_prerequisites(args: argparse.Namespace) -> None:
    """Verify that required tools and files exist before launching."""
    errors = []
    if args.parallel_jobs == 1 and not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")
    if not (REPO_ROOT / "src" / "humano").exists():
        errors.append("humano package not found under src/. Run from the HUMANO repo root.")

    if _normalized_policy(args.policy) in SONIC_TOKEN_POLICIES:
        try:
            import torchcodec  # noqa: F401
        except Exception as exc:
            errors.append(
                "TorchCodec runtime check failed. Install a CUDA-compatible NPP runtime or set "
                f"{NPP_LIBRARY_ENV}=/path/to/npp/lib. Details: {exc}"
            )

    if args.sonic_mode == "external":
        deploy_root = Path(args.sonic_deploy_root).expanduser()
        if not deploy_root.is_dir():
            errors.append(f"deploy_root not found: {deploy_root}")
        else:
            _check_deploy_file(
                errors,
                deploy_root,
                "target/release/g1_deploy_onnx_ref",
                "SONIC deploy binary",
            )
            _check_deploy_file(
                errors,
                deploy_root,
                f"{args.sonic_deploy_checkpoint}_decoder.onnx",
                "SONIC decoder model",
            )
            _check_deploy_file(
                errors,
                deploy_root,
                f"{args.sonic_deploy_checkpoint}_encoder.onnx",
                "SONIC encoder model",
            )
            _check_deploy_file(
                errors,
                deploy_root,
                args.sonic_deploy_obs_config,
                "SONIC observation config",
            )
            _check_deploy_file(
                errors, deploy_root, args.sonic_deploy_planner, "SONIC planner model"
            )
            _check_deploy_dir(
                errors, deploy_root, args.sonic_deploy_motion_data, "SONIC motion data"
            )

    if errors:
        print("ERROR: Prerequisites not met:\n")
        for error in errors:
            print(f"  - {error}")
        print()
        sys.exit(1)


# ---------------------------------------------------------------------------
# TMUX SESSION MANAGEMENT
# ---------------------------------------------------------------------------
def _kill_existing_session() -> None:
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )


def _create_tmux_session(*, sonic_mode: str) -> None:
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION_NAME],
        check=True,
    )
    subprocess.run(["tmux", "set-option", "-t", SESSION_NAME, "-g", "mouse", "on"])
    subprocess.run(["tmux", "bind-key", "-T", "root", "C-\\", "kill-session"])
    subprocess.run(["tmux", "rename-window", "-t", f"{SESSION_NAME}:0", "evaluation"])
    subprocess.run(["tmux", "split-window", "-t", f"{SESSION_NAME}:0", "-h"])
    if sonic_mode == "external":
        subprocess.run(["tmux", "split-window", "-t", f"{SESSION_NAME}:0.1", "-v"])
    time.sleep(0.5)


def _send_to_pane(pane_index: int, cmd: str, wait: float = 1.0) -> None:
    subprocess.run(
        ["tmux", "send-keys", "-t", f"{SESSION_NAME}:0.{pane_index}", cmd, "C-m"],
        check=True,
    )
    time.sleep(wait)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch HUMANO SONIC evaluation in a tmux session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", help="JSON launcher options; CLI overrides config.")
    # Required
    parser.add_argument("--run-dir", help="Policy run/checkpoint directory.")
    parser.add_argument("--data-dir", help="LeRobot dataset root for one task.")
    parser.add_argument("--suite-config", help="JSON suite task manifest.")
    parser.add_argument("--data-root", help="Root containing suite dataset directories.")
    # Policy / server
    parser.add_argument("--wbc-controller", choices=("sonic", "amo", "twist2"))
    parser.add_argument("--policy", default="gr00t_n17")
    parser.add_argument(
        "--external-server",
        action="store_true",
        help="Connect to an existing policy server; requires --port, no checkpoint needed.",
    )
    parser.add_argument("--server-module", default=None)
    parser.add_argument(
        "--server-python",
        default=None,
        help=(
            "Python executable for the policy server, such as the Track B "
            "LeRobot training environment. HUMANO keeps orchestration and protocol ownership."
        ),
    )
    parser.add_argument("--host", default="localhost", help="Host used by HUMANO eval client.")
    parser.add_argument("--port", type=int, default=0, help="Policy server port (0 = auto-pick).")
    parser.add_argument("--server-host", default="0.0.0.0")
    parser.add_argument("--tokenizer-path", help="Local Pi0.5 tokenizer directory.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-id", type=int, help="Host GPU index for the policy and renderer.")
    parser.add_argument("--gpu-ids", help="Comma-separated host GPU indices, one per parallel job.")
    parser.add_argument("--ckpt-step", default=None, help="Checkpoint step or 'latest'.")
    parser.add_argument("--action-exec-horizon", type=int, default=None)
    parser.add_argument(
        "--mock-policy",
        action="store_true",
        help="Use the fixed 40x78 policy backend for Track A smoke tests.",
    )
    parser.add_argument("--embodiment-tag", default="UNITREE_G1_SONIC")
    parser.add_argument(
        "--gr00t-root",
        default=str(DEFAULT_GR00T_ROOT) if DEFAULT_GR00T_ROOT.exists() else "",
    )
    parser.add_argument("--rtc", action="store_true", help="Enable RTC mode in GR00T server.")
    parser.add_argument(
        "--seed", type=int, default=None, help="Base seed for reproducible GR00T eval."
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic PyTorch/CUDA policy inference.",
    )

    # Eval
    parser.add_argument("--env-id", default="humano/G1WholebodyXMovePickTeleop-v0")
    parser.add_argument("--sim-mode", default="mujoco_isaac")
    parser.add_argument("--hand-type", choices=("dex3", "inspire", "brainco"), default="dex3")
    parser.add_argument("--data-format", default="lerobot")
    parser.add_argument("--eval-dir", default="runs/evals")
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        help="Independent policy/eval episode shards (default: one per GPU list entry, otherwise 1).",
    )
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--success-criteria", type=float, default=0.9)

    # Eval display / output
    parser.add_argument("--show-progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="show_progress", action="store_false")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--no-save-video", dest="save_video", action="store_false")
    parser.add_argument("--save-exocentric-video", action="store_true", default=False)
    parser.add_argument("--exocentric-camera", default="franka_camera")
    parser.add_argument("--exocentric-video-key", default="exocentric")
    parser.add_argument("--async-video", action="store_true", default=True)
    parser.add_argument("--no-async-video", dest="async_video", action="store_false")

    # SONIC mode
    parser.add_argument(
        "--sonic-provider",
        choices=["cpu", "gpu"],
        default="cpu",
        help="SONIC decoder inference provider (default: cpu).",
    )
    parser.add_argument(
        "--sonic-mode",
        default="native",
        choices=["native"],
        help="SONIC uses the native backend.",
    )

    # SONIC deploy (external mode)
    parser.add_argument(
        "--sonic-deploy-root",
        default="third_party/gear_sonic_deploy",
    )
    parser.add_argument("--sonic-deploy-interface", default="")
    parser.add_argument("--sonic-deploy-zmq-host", default="localhost")
    parser.add_argument("--sonic-deploy-zmq-port", type=int, default=5556)
    parser.add_argument("--sonic-deploy-feedback-port", type=int, default=5557)
    parser.add_argument("--sonic-deploy-checkpoint", default="policy/release/model")
    parser.add_argument(
        "--sonic-deploy-obs-config", default="policy/release/observation_config.yaml"
    )
    parser.add_argument(
        "--sonic-deploy-planner", default="planner/target_vel/V2/planner_sonic.onnx"
    )
    parser.add_argument("--sonic-deploy-motion-data", default="reference/example/")
    parser.add_argument("--sonic-deploy-logs-dir", default=None)

    # Timing overrides
    parser.add_argument("--env-step-dt", type=float, default=None)
    parser.add_argument("--sonic-control-dt", type=float, default=None)
    parser.add_argument(
        "--fast", action="store_true", help="Skip unnecessary Isaac rendering steps."
    )

    # Connection
    parser.add_argument("--wait-tries", type=int, default=1200)
    parser.add_argument("--wait-sleep-s", type=float, default=0.1)

    # Tmux launcher
    parser.add_argument("--kill-existing", action="store_true", default=True)
    parser.add_argument("--no-kill-existing", dest="kill_existing", action="store_false")
    parser.add_argument("--attach", action="store_true", default=True)
    parser.add_argument("--no-attach", dest="attach", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching.")

    args = _parse_config_args(parser)
    if args.gpu_ids is not None:
        if args.gpu_id is not None:
            parser.error("choose either --gpu-id or --gpu-ids")
        try:
            args.gpu_ids = [int(value) for value in args.gpu_ids.split(",")]
        except ValueError:
            parser.error("--gpu-ids requires comma-separated integers")
        if args.parallel_jobs is None:
            args.parallel_jobs = len(args.gpu_ids)
        if len(args.gpu_ids) != args.parallel_jobs or any(gpu < 0 for gpu in args.gpu_ids):
            parser.error("--gpu-ids requires one nonnegative GPU index per parallel job")
        if args.parallel_jobs == 1:
            args.gpu_id = args.gpu_ids[0]
    if args.parallel_jobs is None:
        args.parallel_jobs = 1
    if args.gpu_id is not None and args.gpu_id < 0:
        parser.error("--gpu-id must be nonnegative")
    if args.gpu_id is not None or args.gpu_ids is not None:
        if args.device not in ("cuda:0", "cuda", "cpu"):
            parser.error("with explicit GPU IDs, --device must be cuda:0, cuda, or cpu")
    if args.parallel_jobs == 1:
        signal.signal(signal.SIGINT, _signal_handler)

    if args.external_server:
        if args.port <= 0:
            parser.error("--external-server requires an explicit positive --port")
    elif not args.run_dir:
        parser.error("--run-dir is required unless --external-server is set")
    if _normalized_policy(args.policy) == "remote" and not (
        args.external_server or args.server_module
    ):
        parser.error("--policy remote requires --external-server or --server-module")

    args.wbc_controller = resolve_wbc_controller(args.policy, args.wbc_controller)
    if args.wbc_controller in ("amo", "twist2") and args.fast:
        parser.error(
            f"{args.wbc_controller.upper()} evaluation requires regular camera updates; omit --fast"
        )
    if args.deterministic and args.seed is None:
        parser.error("--deterministic requires --seed")
    if bool(args.data_dir) == bool(args.suite_config):
        parser.error("provide exactly one of --data-dir or --suite-config")
    if args.suite_config and not args.data_root:
        parser.error("--suite-config requires --data-root")

    if args.parallel_jobs < 1:
        parser.error("--parallel-jobs must be positive")
    if args.parallel_jobs > 1:
        from humano.evals.parallel import run_parallel, validate_parallel

        validate_parallel(args)
        if not args.dry_run:
            _check_prerequisites(args)
        run_parallel(args, _parallel_commands)
        return

    if args.port == 0:
        args.port = _pick_free_port()

    repo_root = REPO_ROOT

    # Override sim_mode for gr00t_n17
    # policy_name = _normalized_policy(args.policy)
    # if policy_name in GR00T_N17_POLICIES and args.sim_mode == "mujoco":
    #     args.sim_mode = "mujoco_isaac"
    #     print(
    #         "[launch_evaluation] gr00t_n17 uses rendered Isaac observations; "
    #         "overriding --sim-mode mujoco -> mujoco_isaac"
    #     )

    sonic_interface = (
        _pick_sonic_interface(args.sonic_deploy_interface)
        if args.sonic_mode in ("internal", "external", "native")
        else ""
    )
    sonic_env = (
        _sonic_env_prefix(args, sonic_interface, sonic_mode=args.sonic_mode)
        if args.sonic_mode != "none"
        else ""
    )

    eval_dir = Path(args.eval_dir).expanduser()
    eval_log = eval_dir / "eval_latest.log"
    policy_server_log = eval_dir / "policy_server.log"
    sonic_deploy_log = eval_dir / "sonic_deploy.log"

    server_cmd = "" if args.external_server else _server_command(args)
    eval_cmd = _shell_cmd(
        repo_root,
        _build_eval_cmd(args, args.port),
        extra_env=" ".join(filter(None, (sonic_env, _lane_env(args, evaluator=True)))),
    )
    sonic_deploy_cmd = (
        _build_sonic_deploy_cmd(args, sonic_interface) if args.sonic_mode == "external" else ""
    )

    if server_cmd:
        server_cmd = _with_python_log_file(server_cmd, policy_server_log)
    if args.sonic_mode == "internal":
        eval_cmd = (
            f"mkdir -p {_quote(sonic_deploy_log.parent)} && "
            f"truncate -s 0 {_quote(sonic_deploy_log)} && "
            f"{eval_cmd}"
        )
    eval_cmd = _with_python_log_file(eval_cmd, eval_log)
    if args.sonic_mode == "external":
        sonic_deploy_cmd = _with_timestamped_log(sonic_deploy_cmd, sonic_deploy_log)

    # Summary
    print("=" * 60)
    print("  HUMANO SONIC Evaluation Launcher")
    print("=" * 60)
    print(f"  Policy server:  {args.server_host}:{args.port}")
    print(f"  Eval client:    {args.host}:{args.port}")
    print(f"  Policy:         {args.policy}")
    if args.suite_config:
        print(f"  Suite:          {Path(args.suite_config).stem}")
    else:
        print(f"  Env:            {args.env_id}")
    print(f"  Episodes:       {args.num_episodes}")
    if args.suite_config:
        print(f"  Suite config:   {args.suite_config}")
        print(f"  Data root:      {args.data_root}")
    else:
        print(f"  Data dir:       {args.data_dir}")
    print(f"  Eval dir:       {args.eval_dir}")
    print(f"  Eval log:       {eval_log}")
    if not args.external_server:
        print(f"  Server log:     {policy_server_log}")
    if args.seed is not None:
        print(f"  Seed:           {args.seed}")
    if args.deterministic:
        print("  Deterministic:  enabled")
    if args.env_step_dt is not None or args.sonic_control_dt is not None:
        env_label = (
            f"{args.env_step_dt * 1000:.0f}ms" if args.env_step_dt is not None else "default"
        )
        sonic_label = (
            f"{args.sonic_control_dt * 1000:.0f}ms"
            if args.sonic_control_dt is not None
            else "default"
        )
        print(f"  Timing:         env.step={env_label} SONIC_CONTROL={sonic_label}")
    if args.fast:
        print("  Fast mode:      enabled (skips Isaac rendering)")
    if args.sonic_mode in ("external", "internal"):
        print(f"  Deploy log:     {sonic_deploy_log}")
    print(f"  SONIC mode:     {args.sonic_mode}")
    print(f"  SONIC provider: {args.sonic_provider}")
    if args.sonic_mode != "none":
        print(f"    Interface:    {sonic_interface}")
        print(f"    ZMQ:          {args.sonic_deploy_zmq_host}:{args.sonic_deploy_zmq_port}")
        print(f"    Feedback:     {args.sonic_deploy_feedback_port}")
    print("=" * 60)
    print("server_cmd:")
    print(f"  {server_cmd}")
    if args.sonic_mode == "external":
        print("sonic_deploy_cmd:")
        print(f"  {sonic_deploy_cmd}")
    print("eval_cmd:")
    print(f"  {eval_cmd}")
    print("=" * 60)

    if args.dry_run:
        print("[dry-run] Done. No tmux session created.")
        return

    _check_prerequisites(args)
    if args.kill_existing:
        _kill_existing_session()
    _create_tmux_session(sonic_mode=args.sonic_mode)

    print(f"Created tmux session: {SESSION_NAME}")
    if server_cmd:
        print("Starting policy server (pane 1, right/top)...")
        _send_to_pane(1, server_cmd, wait=0.0)
        if _normalized_policy(args.policy) in {"psi0", "pi05", "remote"}:
            _wait_for_policy_server_health(
                host=args.host, port=args.port, tries=args.wait_tries, sleep_s=args.wait_sleep_s
            )
    if args.sonic_mode == "external":
        print("Starting SONIC deploy (pane 2, right/bottom)...")
        _send_to_pane(2, sonic_deploy_cmd, wait=3.0)
    print("Starting HUMANO eval (pane 0, left)...")
    _send_to_pane(0, eval_cmd, wait=1.0)
    subprocess.run(["tmux", "select-pane", "-t", f"{SESSION_NAME}:0.0"])

    _SONIC_PANE_LABELS: dict[str, str] = {
        "external": "    Pane 2 (right/bottom): SONIC deploy",
        "internal": "    SONIC deploy:          internal child process of eval_task",
        "native": "    SONIC deploy:          in-process SonicController (no ZMQ/DDS)",
    }
    print()
    print("=" * 60)
    print("  Evaluation launched.")
    print()
    print(f"  tmux session: {SESSION_NAME}")
    print("  Window 'evaluation':")
    print("    Pane 0 (left):         HUMANO EvalRunner  <-- you are here")
    if server_cmd:
        print("    Pane 1 (right/top):    policy server")
    else:
        print("    Policy server:        externally managed")
    if args.sonic_mode in _SONIC_PANE_LABELS:
        print(_SONIC_PANE_LABELS[args.sonic_mode])
    print()
    print("  Navigation:")
    print("    Ctrl+b, arrow keys  - Switch between panes")
    print("    Ctrl+b, d           - Detach from session")
    print("    Ctrl+\\              - Kill entire session")
    print("=" * 60)

    if args.attach:
        try:
            subprocess.run(["tmux", "attach", "-t", SESSION_NAME])
        except KeyboardInterrupt:
            pass

    result = subprocess.run(
        ["tmux", "has-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"\nSession '{SESSION_NAME}' is still running.")
        print(f"  Reattach:  tmux attach -t {SESSION_NAME}")
        print(f"  Kill:      tmux kill-session -t {SESSION_NAME}")


def _signal_handler(_sig, _frame) -> None:
    print("\nShutdown requested...")
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    _ensure_torchcodec_runtime_libs()
    main()
