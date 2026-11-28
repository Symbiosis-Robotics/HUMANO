from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

DEFAULT_MAX_EPISODE_STEPS = 3000
DEFAULT_NUM_EPISODES = 10


def resolve_wbc_controller(policy: str, explicit: str | None = None) -> str:
    selected = explicit or os.environ.get("HUMANO_WBC_CONTROLLER")
    if selected is None:
        return policy.lower() if policy.lower() in ("amo", "twist2") else "sonic"
    if selected not in ("amo", "twist2", "sonic") or (
        policy.lower() in ("amo", "twist2") and selected != policy.lower()
    ):
        raise ValueError(f"Incompatible motion controller {selected!r} for policy {policy!r}")
    return selected


@dataclass(frozen=True)
class EvalConfig:
    env_id: str
    policy: str
    split: str = "eval"
    host: str = "172.17.0.1"
    port: int = 21000
    data_format: str = "lerobot"
    sim_mode: str = "mujoco_isaac"
    headless: bool = False
    eval_dir: str = "data/evals"
    max_episode_steps: int | None = None
    time_budget_file: str | None = None
    num_episodes: int = DEFAULT_NUM_EPISODES
    episode_start: int = 0
    data_dir: str = "data/datagen"
    success_criteria: float = 0.9
    save_video: bool = True
    num_workers: int = 1
    save_exocentric_video: bool = False
    exocentric_camera: str = "franka_camera"
    exocentric_video_key: str = "exocentric"
    async_video: bool = True
    seed: int | None = None
    wbc_controller: str | None = None


@dataclass
class EvalResult:
    env_id: str
    policy: str
    split: str
    stats: dict[str, bool]
    success_rate: float
    log_path: str
    eval_dir: str
    episode_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    fall_rate: float | None = None
    timeout_rate: float | None = None
    median_success_completion_time_s: float | None = None
    median_success_dimensionless_jerk: float | None = None
    metrics_path: str = ""
    evaluation_health: dict[str, Any] = field(default_factory=dict)
    evaluation_health_path: str = ""


class EvalRunner:
    def __init__(
        self,
        config: EvalConfig,
        *,
        agent_factory: Callable[..., Any] | None = None,
        show_progress: bool = True,
        env_step_dt: float | None = None,
        sonic_control_dt: float | None = None,
        fast: bool = False,
    ):
        self.config = config
        self.agent_factory = agent_factory
        self.show_progress = show_progress
        self.env_step_dt = env_step_dt
        self.sonic_control_dt = sonic_control_dt
        self.fast = fast

    def run(self, policy: Any | None = None) -> EvalResult:
        config = self.config
        agent_factory = self.agent_factory

        if policy is not None:
            if isinstance(policy, str):
                config = replace(config, policy=policy)
                agent_factory = None
            elif callable(policy):
                agent_factory = policy
            elif hasattr(policy, "make_agent"):
                agent_factory = policy.make_agent
            else:
                raise TypeError(
                    "policy must be a baseline name, an agent factory, or an object "
                    "with make_agent(robot, host, port)."
                )

        if resolve_wbc_controller(config.policy, config.wbc_controller) in (
            "amo",
            "twist2",
        ) or config.policy.lower().replace("-", "_").replace(".", "") in {
            "gr00t_n17",
            "gr00tn17",
            "gr00t_n1d7",
            "gr00tn1d7",
            "amo",
            "act",
            "diffusion",
            "pi05",
            "pi_05",
            "flow_matching",
        }:
            from humano.cli.eval_task import evaluate_task
        else:
            raise ValueError(
                f"Unsupported policy: {config.policy}. HUMANO currently routes "
                "simulation eval through AMO, TWIST2 and SONIC-token baselines."
            )

        return evaluate_task(
            config,
            agent_factory=agent_factory,
            show_progress=self.show_progress,
            env_step_dt=self.env_step_dt,
            sonic_control_dt=self.sonic_control_dt,
            fast=self.fast,
        )
