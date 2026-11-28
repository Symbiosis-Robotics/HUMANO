"""
Shared Isaac SimulationApp creation helpers.
"""

from __future__ import annotations

import os
from pathlib import Path

from humano.utils import env_flag

HYDRA_WAIT_IDLE = "/app/hydraEngine/waitIdle"
HYDRA_RENDER_COMPLETE = "/app/updateOrder/checkForHydraRenderComplete"
THROTTLING_ENABLE_ASYNC = "/exts/isaacsim.core.throttling/enable_async"


def _compact_dict(values: dict) -> dict:
    return {key: value for key, value in values.items() if value is not None and value != ""}


def create_simulation_app(
    SimulationApp,
    *,
    headless: bool,
    renderer: str = "RayTracedLighting",
    width: int | None = None,
    height: int | None = None,
    anti_aliasing: int | None = None,
    hide_ui: bool | None = None,
    multi_gpu: bool = False,
):
    experience = os.getenv("HUMANO_ISAAC_EXPERIENCE", "").strip()
    zero_delay = env_flag("HUMANO_ISAAC_ZERO_DELAY", default=True)
    disable_throttling_async = env_flag("HUMANO_ISAAC_DISABLE_THROTTLING_ASYNC", default=True)

    settings: list[tuple[str, object, object]] = []
    if zero_delay:
        settings.append((HYDRA_WAIT_IDLE, 1, True))
        settings.append((HYDRA_RENDER_COMPLETE, 1000, 1000))
    if disable_throttling_async:
        settings.append((THROTTLING_ENABLE_ASYNC, "false", False))
    extra_args = [f"--{key}={arg_value}" for key, arg_value, _ in settings]
    if experience:
        for folder in ("extscache", "extsPhysics", "extsDeprecated"):
            extra_args.extend(["--ext-folder", str(Path(os.environ["ISAAC_PATH"]) / folder)])

    sim_cfg = _compact_dict(
        {
            "headless": headless,
            "renderer": renderer,
            "multi_gpu": multi_gpu,
            "anti_aliasing": anti_aliasing,
            "hide_ui": hide_ui,
            "width": width,
            "height": height,
        }
    )
    gpu_id = os.getenv("HUMANO_EVAL_GPU_ID")
    if gpu_id is not None:
        if not gpu_id.isascii() or not gpu_id.isdecimal():
            raise ValueError("HUMANO_EVAL_GPU_ID must be a non-negative integer")
        active_gpu = int(gpu_id)
        # Kit's Vulkan GPU table may be ordered differently from CUDA devices.
        gpu_map = os.getenv("HUMANO_ISAAC_GPU_MAP")
        if gpu_map is not None:
            slots = [slot.strip() for slot in gpu_map.split(",")]
            if active_gpu >= len(slots) or not all(
                slot.isascii() and slot.isdecimal() for slot in slots
            ):
                raise ValueError(
                    f"HUMANO_ISAAC_GPU_MAP={gpu_map!r} requires Kit GPU indices including slot {gpu_id}"
                )
            active_gpu = int(slots[active_gpu])
        # The evaluator sees one CUDA device, remapped to logical index 0.
        sim_cfg.update(active_gpu=active_gpu, physics_gpu=0, multi_gpu=False)
    if extra_args:
        sim_cfg["extra_args"] = extra_args

    app = SimulationApp(sim_cfg, experience=experience)

    for key, _, runtime_value in settings:
        app.set_setting(key, runtime_value)

    return app
