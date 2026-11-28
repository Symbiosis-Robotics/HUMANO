"""Mock and checkpoint-backed inference runtimes for HUMANO."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from humano.policy.server import action_dim

from .contract import ACTION_HORIZON, BaselineSpec, validate_action_chunk


class FixedPolicyRuntime:
    def __init__(self, spec: BaselineSpec, hand_type: str = "dex3"):
        self.spec = spec
        self.hand_type = hand_type

    def reset(self) -> None:
        return None

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        if self.spec.requires_task and not observation.get("task", "").strip():
            raise ValueError(f"{self.spec.name} requires a non-empty task")
        return np.zeros((ACTION_HORIZON, action_dim(self.hand_type)), dtype=np.float32)


class LeRobotPolicyRuntime:
    def __init__(
        self,
        *,
        spec: BaselineSpec,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        device: Any,
        hand_type: str = "dex3",
    ):
        self.spec = spec
        self.hand_type = hand_type
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def _batch(self, observation: dict[str, Any]) -> dict[str, Any]:
        import torch

        state = np.concatenate(
            (
                observation["observation.state"],
                observation["observation.projected_gravity"],
            )
        ).astype(np.float32, copy=False)
        # (46,) mixed joint rad and unitless gravity, GR00T/body-frame order
        image = np.asarray(observation["observation.images.ego_view"])
        # (H, W, 3) uint8 or normalized RGB, ego-camera frame
        if not image.flags.writeable:
            image = image.copy()
            # (H, W, 3) writable RGB buffer for torch.from_numpy
        image_tensor = torch.from_numpy(image)
        # (H, W, 3) image values, ego-camera frame
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.to(torch.float32) / 255.0
            # (H, W, 3) normalized RGB, ego-camera frame
        else:
            image_tensor = image_tensor.to(torch.float32)
            # (H, W, 3) source-scale RGB, ego-camera frame
        return {
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(self.device),
            "observation.images.ego_view": (
                image_tensor.permute(2, 0, 1).contiguous().unsqueeze(0).to(self.device)
            ),
            "task": observation.get("task", ""),
            "robot_type": "g1_sonic",
        }

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        import torch

        task = observation.get("task", "")
        if self.spec.requires_task and (not isinstance(task, str) or not task.strip()):
            raise ValueError(f"{self.spec.name} requires a non-empty task")

        with torch.no_grad():
            batch = self.preprocessor(self._batch(observation))
            if self.spec.policy_type == "diffusion":
                batch = {
                    **batch,
                    "observation.state": batch["observation.state"].unsqueeze(1),
                }
            if self.spec.policy_type == "multi_task_dit":
                action = torch.stack(
                    [self.policy.select_action(batch) for _ in range(ACTION_HORIZON)],
                    dim=1,
                )
                # (B, 40, 78) mixed SONIC tokens and hand targets, robot action frame
            else:
                action = self.policy.predict_action_chunk(batch)
                # (B, 40, 78) mixed SONIC tokens and hand targets, robot action frame
            action = self.postprocessor(action)
            # (B, 40, 78) denormalized mixed action units, robot action frame

        if not isinstance(action, torch.Tensor):
            raise TypeError(
                f"{self.spec.name} postprocessor must return torch.Tensor, "
                f"got {type(action).__name__}"
            )
        result = action.detach().to(device="cpu", dtype=torch.float32).numpy()
        # (1, 40, 78) denormalized mixed action units, robot action frame
        if result.shape != (1, ACTION_HORIZON, action_dim(self.hand_type)):
            raise ValueError(
                f"{self.spec.name} checkpoint must output shape "
                f"(1, {ACTION_HORIZON}, {action_dim(self.hand_type)}), got {result.shape}"
            )
        return validate_action_chunk(result[0], self.hand_type)


def validate_checkpoint_config(spec: BaselineSpec, config: Any) -> None:
    if config.type != spec.policy_type:
        raise ValueError(
            f"{spec.name} requires policy type {spec.policy_type!r}, got {config.type!r}"
        )
    if spec.objective is not None and getattr(config, "objective", None) != spec.objective:
        raise ValueError(
            f"{spec.name} requires objective {spec.objective!r}, "
            f"got {getattr(config, 'objective', None)!r}"
        )


def _get_policy_class(policy_type: str) -> type:
    policy_modules = {
        "act": ("lerobot.policies.act.modeling_act", "ACTPolicy"),
        "diffusion": (
            "lerobot.policies.diffusion.modeling_diffusion",
            "DiffusionPolicy",
        ),
        "multi_task_dit": (
            "lerobot.policies.multi_task_dit.modeling_multi_task_dit",
            "MultiTaskDiTPolicy",
        ),
        "pi05": ("lerobot.policies.pi05.modeling_pi05", "PI05Policy"),
    }
    try:
        module_name, class_name = policy_modules[policy_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported Track B policy type: {policy_type!r}") from exc
    return getattr(importlib.import_module(module_name), class_name)


def _load_pre_post_processors(spec: BaselineSpec, checkpoint_path: Path, device: str) -> tuple[Any, Any]:
    from lerobot.processor import (
        PolicyProcessorPipeline,
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )

    if spec.name == "pi05":
        from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep  # noqa: F401

    overrides = {"device_processor": {"device": device}}
    tokenizer_path = os.environ.get("HUMANO_PI05_TOKENIZER_PATH")
    if spec.name == "pi05" and tokenizer_path:
        overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer_path}

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(checkpoint_path),
        config_filename="policy_preprocessor.json",
        overrides=overrides,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(checkpoint_path),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def load_runtime(
    spec: BaselineSpec,
    checkpoint_dir: str | Path,
    device: str,
    *,
    mock_policy: bool = False,
    hand_type: str = "dex3",
) -> FixedPolicyRuntime | LeRobotPolicyRuntime:
    if mock_policy:
        return FixedPolicyRuntime(spec, hand_type)

    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_path}")

    import torch

    import lerobot.policies  # noqa: F401 - registers the supported config classes
    if spec.name == "pi05":
        import lerobot.policies.pi05.processor_pi05  # noqa: F401 - registers PI05 processor steps
    from lerobot.configs import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    validate_checkpoint_config(spec, config)
    expected_state = 44 if hand_type == "brainco" else 46
    if tuple(config.input_features["observation.state"].shape) != (expected_state,) or tuple(config.output_features["action"].shape) != (action_dim(hand_type),):
        raise ValueError(f"Checkpoint state/action dimensions do not match {hand_type}")
    config.device = device
    policy = _get_policy_class(config.type).from_pretrained(checkpoint_path, config=config)
    policy = policy.to(torch.device(device)).eval()
    preprocessor, postprocessor = _load_pre_post_processors(spec, checkpoint_path, device)
    return LeRobotPolicyRuntime(
        spec=spec,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=torch.device(device),
        hand_type=hand_type,
    )
