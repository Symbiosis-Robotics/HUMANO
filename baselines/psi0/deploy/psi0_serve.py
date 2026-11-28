from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import numpy as np
from PIL import Image

from humano.policy.client import RequestMessage, ResponseMessage

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = REPO_ROOT / "baselines" / "psi0"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

STATE_DIM = 46
ACTION_DIM = 78
ACTION_HORIZON = 40
IMAGE_SHAPE = (360, 640, 3)
STATE_PARTS = (
    ("left_leg", 6),
    ("right_leg", 6),
    ("waist", 3),
    ("left_arm", 7),
    ("left_hand", 7),
    ("right_arm", 7),
    ("right_hand", 7),
    ("projected_gravity", 3),
)


def _state_parts(hand_type: str):
    if hand_type not in {"dex3", "brainco"}:
        raise ValueError(f"unsupported hand type: {hand_type}")
    return tuple((name, 6 if hand_type == "brainco" and name.endswith("hand") else size) for name, size in STATE_PARTS)


class Runtime(Protocol):
    def reset(self) -> None: ...

    def predict(
        self, image: np.ndarray, state: np.ndarray, instruction: str
    ) -> np.ndarray: ...


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _vector(values: Any, name: str, size: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return result


def _request_arrays(request: RequestMessage, hand_type: str = "dex3") -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(request.image.get("ego_view"))
    if image.shape != IMAGE_SHAPE or not np.isfinite(image).all():
        raise ValueError(f"ego_view must be finite with shape {IMAGE_SHAPE}")
    parts = [_vector(request.state.get(name), name, size) for name, size in _state_parts(hand_type)]
    state = np.concatenate(parts).astype(np.float32, copy=False)[None, :]
    return image, state


def _action_chunk(values: Any, action_dim: int = ACTION_DIM) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (ACTION_HORIZON, action_dim):
        raise ValueError(
            f"action must have shape ({ACTION_HORIZON}, {action_dim}), got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError("action must contain only finite values")
    return result


def _validate_checkpoint_config(config: Any, hand_type: str = "dex3") -> None:
    state_dim = sum(size for _, size in _state_parts(hand_type))
    action_dim = state_dim + 32
    model = config.model
    if getattr(model, "use_dit", False):
        raise ValueError("HUMANO's psi0 runtime supports only use_dit=false checkpoints")
    actual = (
        model.odim,
        model.action_dim,
        model.action_chunk_size,
        model.action_exec_horizon,
    )
    expected = (state_dim, action_dim, ACTION_HORIZON, ACTION_HORIZON)
    if actual != expected:
        raise ValueError(f"psi0 checkpoint dimensions must be {expected}, got {actual}")
    field = config.data.transform.field
    # Some training checkpoints leave these optional padding fields unset while
    # declaring the fixed model dimensions above. The runtime input/output shapes
    # are validated against state_dim and action_dim, so only reject explicit
    # conflicting values here.
    if field.pad_state_dim not in (None, state_dim) or field.pad_action_dim not in (None, action_dim):
        raise ValueError("psi0 normalization dimensions, when set, must be (46, 78)")


def _resolve_checkpoint_step(run_dir: Path, ckpt_step: str) -> str:
    if ckpt_step != "latest":
        return ckpt_step
    steps = []
    for parent in (run_dir / "checkpoints", run_dir):
        for path in parent.glob("ckpt_*"):
            suffix = path.name.removeprefix("ckpt_")
            if path.is_dir() and suffix.isdigit():
                steps.append(int(suffix))
    if not steps:
        raise FileNotFoundError(f"no numeric Psi0 checkpoints found under {run_dir}")
    return str(max(steps))


class Psi0Server:
    def __init__(self, runtime: Runtime, hand_type: str = "dex3"):
        self.hand_type = hand_type
        self.action_dim = sum(size for _, size in _state_parts(hand_type)) + 32
        self.runtime = runtime
        self._lock = threading.Lock()
        self._session_generation = 0

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "baseline": "psi0",
            "policy_type": "psi0",
            "backend": "checkpoint",
            "action_shape": [ACTION_HORIZON, self.action_dim],
        }

    def reset(self) -> dict[str, str]:
        with self._lock:
            self.runtime.reset()
        return {"status": "ok", "baseline": "psi0"}

    def predict_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = RequestMessage.deserialize(payload)
        image, state = _request_arrays(request, self.hand_type)
        generation = request.history.get("session_generation", 0)
        reset_requested = request.history.get("reset") is True
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("session_generation must be a non-negative integer")
        with self._lock:
            if generation < self._session_generation:
                raise ValueError("request belongs to an older Psi0 session")
            if generation > self._session_generation and not reset_requested:
                raise ValueError("a new session_generation requires reset=true")
            self._session_generation = generation
            if reset_requested:
                self.runtime.reset()
            if "inference_seed" in payload:
                self.runtime.seed(int(payload["inference_seed"]))
            action = _action_chunk(self.runtime.predict(image, state, request.instruction), self.action_dim)
        return ResponseMessage(action=action, err=0.0).serialize()


class Psi0CheckpointRuntime:
    def __init__(self, run_dir: Path, ckpt_step: str, device: str, hand_type: str = "dex3"):
        self.state_dim = sum(size for _, size in _state_parts(hand_type))
        self.action_dim = self.state_dim + 32
        import torch
        from psi.models.psi0 import Psi0Model
        from torchvision.transforms import v2

        config = _namespace(json.loads((run_dir / "run_config.json").read_text()))
        _validate_checkpoint_config(config, hand_type)
        self._seed(config.seed or 42, torch)
        resolved_step = _resolve_checkpoint_step(run_dir, ckpt_step)
        self.model = Psi0Model.from_pretrained(run_dir, resolved_step, config, device=device)
        self.model.to(device).eval()
        self.device = torch.device(device)
        self.field = config.data.transform.field
        image_config = config.data.transform.model
        self.image_transform = v2.Compose(
            [v2.Resize(image_config.resize.size), v2.CenterCrop(image_config.center_crop.size)]
        )

    def seed(self, seed: int) -> None:
        import torch

        self._seed(seed, torch)

    @staticmethod
    def _seed(seed: int, torch: Any) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def reset(self) -> None:
        return None

    @staticmethod
    def _bounds(values: Any, name: str, size: int) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (size,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must contain {size} finite values")
        return result

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        if not self.field.normalize_state:
            return state
        low = self._bounds(self.field.state_min, "state_min", self.state_dim)
        high = self._bounds(self.field.state_max, "state_max", self.state_dim)
        ill = np.abs(high - low) < 1e-4 * (np.abs(high) + np.abs(low) + 1e-8)
        normalized = np.where(ill, 0.0, (state - low) / (high - low) * 2.0 - 1.0)
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.field.action_norm_type not in {"bounds", "bounds_q99"}:
            raise ValueError(f"unsupported action normalization: {self.field.action_norm_type}")
        low = self._bounds(self.field.action_min, "action_min", self.action_dim)
        high = self._bounds(self.field.action_max, "action_max", self.action_dim)
        if self.field.use_norm_mask:
            mask = np.asarray(self.field.action_norm_masks, dtype=bool)
        else:
            mask = np.ones(self.action_dim, dtype=bool)
        return np.where(mask, 0.5 * (action + 1.0) * (high - low) + low, action)

    def predict(
        self, image: np.ndarray, state: np.ndarray, instruction: str
    ) -> np.ndarray:
        import torch

        state_tensor = torch.from_numpy(self._normalize_state(state)).to(self.device)
        action = self.model.predict_action(
            observations=[[self.image_transform(Image.fromarray(image))]],
            states=state_tensor.unsqueeze(0),
            instructions=[instruction],
            num_inference_steps=10,
            traj2ds=None,
        )
        normalized = action.reshape(ACTION_HORIZON, self.action_dim).cpu().numpy()
        return np.asarray(self._denormalize_action(normalized), dtype=np.float32)


class _RequestHandler(BaseHTTPRequestHandler):
    server_version = "HumanoPsi0/1.0"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send(404, {"error": "not found"})
            return
        self._send(200, self.server.policy_server.health())

    def do_POST(self) -> None:
        if self.path == "/reset":
            self._send(200, self.server.policy_server.reset())
            return
        if self.path != "/act":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            response = self.server.policy_server.predict_action(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
            return
        except (OSError, RuntimeError) as exc:
            self._send(500, {"error": str(exc)})
            return
        self._send(200, response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the HUMANO Psi0 baseline.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21000)
    parser.add_argument("--policy", default="psi0")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ckpt-step", default="latest")
    parser.add_argument("--action-exec-horizon", type=int, default=ACTION_HORIZON)
    parser.add_argument("--hand-type", choices=("dex3", "brainco"), default="dex3")
    parser.add_argument("--rtc", action="store_true")
    args = parser.parse_args()
    if args.policy != "psi0" or args.action_exec_horizon != ACTION_HORIZON:
        raise ValueError("psi0 requires policy=psi0 and action horizon 40")
    if args.rtc:
        raise ValueError("HUMANO's psi0 runtime does not support RTC yet")
    runtime = Psi0CheckpointRuntime(Path(args.run_dir), args.ckpt_step, args.device, args.hand_type)
    httpd = ThreadingHTTPServer((args.host, args.port), _RequestHandler)
    httpd.policy_server = Psi0Server(runtime, args.hand_type)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
