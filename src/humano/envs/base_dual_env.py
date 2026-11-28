"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from humano.core.task import Task
    from humano.engines.isaacsim import IsaacSimSimulator
    from humano.engines.mujoco import MujocoSimulator

import gymnasium as gym

from humano.tasks.registry import TaskRegistry
from humano.utils import env_flag

_ISAAC_LOADED = False
_SIMULATION_APP = None


def _preload_native_runtime(*, wbc: bool = False) -> None:
    if wbc:
        # Bind WBC's native dependencies before Isaac loads incompatible copies.
        import pinocchio  # noqa: F401

    # Import torch/curobo before SimulationApp startup to avoid late native loads
    # that can trip glibc TLS assertions during HUMANO eval.
    import torch  # noqa: F401
    from curobo.types.state import JointState  # noqa: F401


class BaseDualSim(gym.Env):
    """
    Base class for environments that can run in both Mujoco and Isaac Sim.
    """

    task: Task
    sim_mode: str

    mujoco: MujocoSimulator
    isaac: IsaacSimSimulator | None

    def __init__(
        self,
        task: str | Task,
        sim_mode="mujoco_isaac",  # =SIM_MODE.MUJOCO_ISAAC
        headless=True,
        webrtc=False,
        *args,
        **kwargs,
    ) -> None:
        if kwargs.get("hand_type") == "brainco" and sim_mode not in {"mujoco", "mujoco_isaac"}:
            raise ValueError(
                "BrainCo requires MuJoCo physics; use sim_mode='mujoco' or 'mujoco_isaac'"
            )
        self.headless = headless
        self.webrtc = webrtc
        if "isaac" in sim_mode:
            if not _ISAAC_LOADED:
                self._init_isaac(headless, webrtc)

        if isinstance(task, str):
            # FIXME dynamic import task
            self.task = TaskRegistry.make(task, *args, **kwargs)
        else:
            self.task = task

        self.sim_mode = sim_mode

        if "isaac" in self.sim_mode:
            from humano.engines.isaacsim import IsaacSimSimulator

            self.isaac = IsaacSimSimulator(self.task, headless=headless)
        else:
            self.isaac = None

        from humano.engines import MujocoSimulator

        self.mujoco = MujocoSimulator(self.task, headless=("isaac" in self.sim_mode) or headless)

        self.action_space = self.task.action_space
        self.observation_space = self.task.observation_space

    def _init_isaac(self, headless: bool, webrtc: bool = False):
        global _ISAAC_LOADED, _SIMULATION_APP
        assert not _ISAAC_LOADED, "Isaac already loaded"
        _preload_native_runtime()
        import isaacsim
        from omni.isaac.kit import SimulationApp  # type: ignore

        from humano.engines.isaac_app import create_simulation_app
        from humano.engines.render_quality import get_render_quality_preset

        render_quality = get_render_quality_preset()

        # Step 1: Create SimulationApp
        _SIMULATION_APP = create_simulation_app(
            SimulationApp,
            headless=headless,
            renderer=render_quality.app_renderer,
            anti_aliasing=render_quality.app_anti_aliasing,
            hide_ui=False,
        )

        # Step 2: Enable WebRTC streaming if requested
        if webrtc:
            from omni.isaac.core.utils.extensions import enable_extension

            # Determine Isaac Sim version and setup accordingly
            try:
                from isaacsim import util  # This exists in Isaac Sim 4.5.0+

                # Isaac Sim 4.5.0+ behavior
                _SIMULATION_APP.set_setting("/app/window/drawMouse", True)
                enable_extension("omni.kit.livestream.webrtc")
            except ImportError:
                # Isaac Sim 4.2.0 behavior
                _SIMULATION_APP.set_setting("/app/window/drawMouse", True)
                _SIMULATION_APP.set_setting("/app/livestream/proto", "ws")
                _SIMULATION_APP.set_setting("/app/livestream/websocket/framerate_limit", 60)
                _SIMULATION_APP.set_setting("/ngx/enabled", False)
                enable_extension("omni.services.streamclient.webrtc")
        _ISAAC_LOADED = True

    @property
    def simulation_app(self):
        return _SIMULATION_APP

    def spin(self, timeout=100):  # spin isaac for a while, easy for debugging
        start = time.monotonic()
        while (
            "isaac" in self.sim_mode and not self.headless and self.simulation_app.is_running()  # type: ignore
        ):
            self.simulation_app.update()  # type: ignore
            end = time.monotonic()
            if end - start > timeout:
                print(f"Spin timeout after {timeout} seconds.")
                break

    def close(self):
        self.mujoco.close()
        if _ISAAC_LOADED:
            assert self.isaac is not None
            while (
                "isaac" in self.sim_mode and not self.headless and self.simulation_app.is_running()  # type: ignore
            ):
                self.isaac.update_visuals()
                self.simulation_app.update()  # type: ignore
            print("Closing IsaacSim simulator...")
            _SIMULATION_APP.close()
        super().close()
