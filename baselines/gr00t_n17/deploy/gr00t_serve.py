from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from humano.policy.server import (
    RequestHandler as Handler,
)
from humano.policy.server import (
    RequestMessage,
    ResponseMessage,
    StaleSessionRequest,
    serve_server,
)
from humano.utils.config import configure_logging

REPO_ROOT = Path(__file__).resolve().parents[3]
LOGGER = logging.getLogger(__name__)


def _default_gr00t_root() -> Path:
    configured_root = os.environ.get("HUMANO_GR00T_ROOT")
    if not configured_root:
        return REPO_ROOT / "baselines" / "gr00t_n17"
    root = Path(configured_root).expanduser()
    return root if root.is_absolute() else REPO_ROOT / root


DEFAULT_GR00T_ROOT = _default_gr00t_root()


def _configure_reproducibility(seed: int, deterministic: bool) -> Any:
    """Seed the server before model construction and configure deterministic CUDA kernels."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    return torch


def _normalize_policy_name(policy: str) -> str:
    return policy.split(":", 1)[0].lower().replace("-", "_").replace(".", "")


def _is_gr00t_n17_policy(policy: str) -> bool:
    return _normalize_policy_name(policy) in {
        "gr00t_n17",
        "gr00tn17",
        "gr00t_n1d7",
        "gr00tn1d7",
    }


class Gr00tN17Server:
    def __init__(
        self,
        *,
        policy: str,
        run_dir: str,
        device: str,
        gr00t_root: str,
        embodiment_tag: str | None,
        action_exec_horizon: int | None,
        fill_missing_state_zero: bool,
        broadcast_single_image: bool,
        seed: int | None,
        deterministic: bool,
    ):
        if not _is_gr00t_n17_policy(policy):
            raise ValueError(f"Unsupported policy for HUMANO GR00T server: {policy}")

        root = Path(gr00t_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"GR00T root does not exist: {root}")
        self._torch = _configure_reproducibility(seed or 0, deterministic)
        self._inference_lock = threading.Lock()
        self._session_generation = 0
        self._cuda_devices = []
        if self._torch.cuda.is_available() and device.startswith("cuda"):
            self._cuda_devices = [self._torch.device(device).index or 0]
        sys.path.insert(0, str(root))

        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.model.gr00t_n1d7 import gr00t_n1d7
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        if ":" in policy and embodiment_tag is None:
            embodiment_tag = policy.split(":", 1)[1]
        embodiment_tag = embodiment_tag or os.environ.get(
            "GR00T_EMBODIMENT_TAG", "UNITREE_G1_SONIC"
        )

        model_dir = Path(run_dir).expanduser().resolve()
        self.action_exec_horizon = action_exec_horizon
        self.fill_missing_state_zero = fill_missing_state_zero
        self.broadcast_single_image = broadcast_single_image
        self.embodiment_tag = EmbodimentTag.resolve(embodiment_tag)
        LOGGER.info("[humano.gr00t_serve] GR00T root: %s", root)
        LOGGER.info("[humano.gr00t_serve] Model dir: %s", model_dir)
        LOGGER.info("[humano.gr00t_serve] Embodiment tag: %s", self.embodiment_tag)
        LOGGER.info("[humano.gr00t_serve] Device: %s", device)
        self._patch_local_cosmos_backbone(gr00t_n1d7)
        self.policy = Gr00tPolicy(
            embodiment_tag=self.embodiment_tag,
            model_path=str(model_dir),
            device=device,
            strict=True,
        )
        self.modality_configs = self.policy.get_modality_config()

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

    def reset(self) -> dict[str, Any]:
        with self._inference_lock:
            self.policy.reset()
            self._session_generation = 0
        return {"status": "ok"}

    @staticmethod
    def _patch_local_cosmos_backbone(gr00t_n1d7_module: Any) -> None:
        original = gr00t_n1d7_module.get_backbone_cls

        def get_backbone_cls(config: Any) -> Any:
            model_name = str(config.model_name)
            if (
                "Cosmos-Reason2" in model_name
                or "Qwen3-VL" in model_name
                or "qwen3-vl" in model_name.lower()
            ):
                from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

                return Qwen3Backbone
            return original(config)

        gr00t_n1d7_module.get_backbone_cls = get_backbone_cls

    @staticmethod
    def _first_present(mapping: dict[str, Any], keys: list[str]) -> Any | None:
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None

    @staticmethod
    def _as_btd(value: Any, horizon: int, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, None, :]
        elif arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim != 3:
            raise ValueError(f"{name} must be (D,), (T,D), or (B,T,D), got {arr.shape}")
        if arr.shape[1] == horizon:
            return arr
        if arr.shape[1] == 1:
            return np.repeat(arr, horizon, axis=1)
        if arr.shape[1] > horizon:
            return arr[:, -horizon:, :]
        pad = np.repeat(arr[:, -1:, :], horizon - arr.shape[1], axis=1)
        return np.concatenate([arr, pad], axis=1)

    @staticmethod
    def _as_video(value: Any, horizon: int, name: str) -> np.ndarray:
        arr = np.asarray(value)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.size and arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        if arr.ndim == 3:
            arr = arr[None, None, ...]
        elif arr.ndim == 4:
            arr = arr[None, ...]
        elif arr.ndim != 5:
            raise ValueError(
                f"{name} must be (H,W,C), (T,H,W,C), or (B,T,H,W,C), got {arr.shape}"
            )
        if arr.shape[-1] != 3:
            raise ValueError(f"{name} must be RGB with C=3, got {arr.shape}")
        if arr.shape[1] == horizon:
            return arr
        if arr.shape[1] == 1:
            return np.repeat(arr, horizon, axis=1)
        if arr.shape[1] > horizon:
            return arr[:, -horizon:, ...]
        pad = np.repeat(arr[:, -1:, ...], horizon - arr.shape[1], axis=1)
        return np.concatenate([arr, pad], axis=1)

    @staticmethod
    def _action_btd(value: Any, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1:
            return arr[None, None, :]
        if arr.ndim == 2:
            return arr[None, :, :]
        if arr.ndim == 3:
            return arr
        raise ValueError(f"{name} must be (D,), (T,D), or (B,T,D), got {arr.shape}")

    def _build_observation(self, request: RequestMessage) -> dict[str, dict[str, Any]]:
        image_dict = request.image
        state_dict = request.state
        video_horizon = len(self.modality_configs["video"].delta_indices)
        state_horizon = len(self.modality_configs["state"].delta_indices)
        language_horizon = len(self.modality_configs["language"].delta_indices)
        observation: dict[str, dict[str, Any]] = {
            "video": {},
            "state": {},
            "language": {},
        }

        image_values = list(image_dict.values())
        for key in self.modality_configs["video"].modality_keys:
            raw = self._first_present(
                image_dict, [key, f"video.{key}", f"observation.images.{key}"]
            )
            if raw is None and self.broadcast_single_image and len(image_values) == 1:
                raw = image_values[0]
            if raw is None:
                raise KeyError(
                    f"Missing video key {key!r}; got {sorted(image_dict.keys())}"
                )
            observation["video"][key] = self._as_video(
                raw, video_horizon, f"video.{key}"
            )

        batch_size = next(iter(observation["video"].values())).shape[0]
        for key in self.modality_configs["state"].modality_keys:
            raw = self._first_present(
                state_dict, [key, f"state.{key}", f"observation.state.{key}"]
            )
            if raw is None:
                if not self.fill_missing_state_zero:
                    raise KeyError(
                        f"Missing state key {key!r}; got {sorted(state_dict.keys())}"
                    )
                raw = np.zeros((batch_size, state_horizon, 1), dtype=np.float32)
            observation["state"][key] = self._as_btd(raw, state_horizon, f"state.{key}")

        for key in self.modality_configs["language"].modality_keys:
            observation["language"][key] = [
                [request.instruction] * language_horizon for _ in range(batch_size)
            ]
        return observation

    def predict_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = RequestMessage.deserialize(payload)
        observation = self._build_observation(request)
        generation = request.history.get("session_generation", 0)
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise ValueError("session_generation must be a non-negative integer")
        if generation < 0:
            raise ValueError("session_generation must be a non-negative integer")
        reset_requested = request.history.get("reset") is True
        inference_seed = payload.get("inference_seed")
        with self._inference_lock:
            if generation < self._session_generation:
                raise StaleSessionRequest(
                    f"request belongs to older session {generation}; "
                    f"current session is {self._session_generation}"
                )
            if generation > self._session_generation and not reset_requested:
                raise ValueError("a new session_generation requires reset=true")
            if generation > self._session_generation:
                self._session_generation = generation
            if inference_seed is None:
                action, _info = self.policy.get_action(observation)
            else:
                with self._torch.random.fork_rng(devices=self._cuda_devices):
                    self._torch.manual_seed(int(inference_seed))
                    if self._cuda_devices:
                        self._torch.cuda.manual_seed_all(int(inference_seed))
                    action, _info = self.policy.get_action(observation)
        chunks = []
        for key in self.modality_configs["action"].modality_keys:
            if key not in action:
                raise KeyError(f"GR00T output missing action key {key!r}")
            arr = self._action_btd(action[key], f"action.{key}")
            if self.action_exec_horizon is not None:
                arr = arr[:, : self.action_exec_horizon, :]
            chunks.append(arr)
        pred_actions = np.concatenate(chunks, axis=-1)
        if pred_actions.shape[0] == 1:
            pred_actions = pred_actions[0]
        return ResponseMessage(pred_actions.astype(np.float32), 0.0).serialize()


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Serve a GR00T N1.7 checkpoint for HUMANO eval.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=22085)
    parser.add_argument("--policy", default="gr00t_n17")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embodiment-tag")
    parser.add_argument("--gr00t-root", default=str(DEFAULT_GR00T_ROOT))
    parser.add_argument("--action-exec-horizon", type=int)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fill-missing-state-zero", action="store_true")
    parser.add_argument(
        "--no-broadcast-single-image",
        dest="broadcast_single_image",
        action="store_false",
    )
    parser.set_defaults(broadcast_single_image=True)
    args = parser.parse_args()

    policy_server = Gr00tN17Server(
        policy=args.policy,
        run_dir=args.run_dir,
        device=args.device,
        gr00t_root=args.gr00t_root,
        embodiment_tag=args.embodiment_tag,
        action_exec_horizon=args.action_exec_horizon,
        fill_missing_state_zero=args.fill_missing_state_zero,
        broadcast_single_image=args.broadcast_single_image,
        seed=args.seed,
        deterministic=args.deterministic,
    )
    serve_server(policy_server, host=args.host, port=args.port)
    return 0


def _main() -> int:
    try:
        return main()
    except Exception:
        LOGGER.exception("[humano.gr00t_serve] Fatal error")
        raise


if __name__ == "__main__":
    raise SystemExit(_main())
