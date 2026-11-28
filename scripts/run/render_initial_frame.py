from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import humano.envs  # noqa: F401
from humano.wbc.sonic.config import load_sonic_config
from humano.core.actor import ObjectActor
from humano.robots.hands import get_hand_spec


def _env_id(task: str) -> str:
    if task.startswith("humano/"):
        return task
    if task.endswith("-v0"):
        return f"humano/{task}"
    raise ValueError("Pass a Gym env id like G1WholebodyTableApplePickTeleop-v0")


def _task_debug_name(task: str) -> str:
    name = task.split("/", 1)[-1]
    return name[:-3] if name.endswith("-v0") else name


def _task_debug_category(task: Any) -> str | None:
    module_parts = type(task).__module__.split(".")
    for category in ("basic", "advanced", "condition", "goal", "long"):
        if category in module_parts:
            return category
    return None


def _serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def _actor_summary(env) -> dict[str, Any]:
    summary = {}
    for name, actor in env.unwrapped.task.layout.actors.items():
        asset = getattr(actor, "asset", actor)
        entry = {
            "type": type(actor).__name__,
            "uid": getattr(actor, "uid", None),
            "instance_id": getattr(actor, "instance_id", None),
            "simulation_name": getattr(actor, "simulation_name", None),
            "asset_uid": getattr(asset, "uid", None),
            "asset_name": getattr(asset, "name", None),
            "asset_label": getattr(asset, "label", None),
            "position": list(getattr(actor.pose, "position", [])),
            "quaternion": list(getattr(actor.pose, "quaternion", [])),
        }
        if name == "robot":
            entry["robot_uid"] = getattr(getattr(actor, "robot", None), "uid", None)
        if isinstance(actor, ObjectActor):
            stable_pose = getattr(actor.asset, "stable_poses", None)
            if stable_pose is not None and len(stable_pose) > 0:
                entry["stable_z"] = _serializable(stable_pose[0][2])
        if hasattr(actor, "size"):
            entry["size"] = list(actor.size)
        summary[name] = entry
    return summary


def _first_image(obs: dict[str, Any]) -> tuple[str, np.ndarray]:
    for key in ("head_stereo_left", "head_stereo_right"):
        value = obs.get(key)
        if isinstance(value, np.ndarray) and value.ndim == 3:
            return key, value
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] == 3:
            return key, value
    raise RuntimeError(f"No RGB image found in observation keys: {sorted(obs)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Defaults to data/debug/<category>/<task>/ for basic and advanced "
            "tasks, or data/debug/<task>/ for other task categories."
        ),
    )
    parser.add_argument("--hand-type", default="dex3")
    parser.add_argument("--render-hz", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-condition", default=None)
    parser.add_argument("--initial-position", default=None)
    parser.add_argument("--goal", default=None)
    args = parser.parse_args()

    hand_spec = get_hand_spec(args.hand_type)
    sonic_config = load_sonic_config(hand_spec)

    make_kwargs = dict(
        sim_mode="mujoco",
        render_hz=args.render_hz,
        physics_dt=sonic_config["SIMULATE_DT"],
        headless=True,
        sonic_config=sonic_config,
        hand_type=hand_spec.name,
        target=None,
    )
    if args.initial_condition is not None:
        make_kwargs["initial_condition"] = args.initial_condition
    if args.initial_position is not None:
        make_kwargs["initial_position"] = args.initial_position
    if args.goal is not None:
        make_kwargs["goal"] = args.goal
    env = gym.make(_env_id(args.task), **make_kwargs)
    try:
        obs, info = env.reset(seed=args.seed)
        image_key, image = _first_image(obs)
        if args.output_dir is not None:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path("data/debug")
            category = _task_debug_category(env.unwrapped.task)
            if category is not None:
                output_dir /= category
            output_dir /= _task_debug_name(args.task)
        selected_condition = getattr(env.unwrapped.task, "selected_initial_condition", None)
        selected_goal = getattr(env.unwrapped.task, "selected_goal", None)
        if selected_condition is not None:
            output_dir = output_dir / str(selected_condition)
        if selected_goal is not None:
            output_dir = output_dir / str(selected_goal)
        if args.initial_position is not None and args.output_dir is None:
            output_dir = output_dir / f"initial_position_{args.initial_position}"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"first_frame_{image_key}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        scene_path = output_dir / "initial_scene.json"
        scene_summary = {
            "initial_condition": selected_condition,
            "selected_goal": selected_goal,
            "initial_position": args.initial_position,
            "actors": _actor_summary(env),
        }
        scene_path.write_text(
            json.dumps(
                _serializable(scene_summary),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"image={image_path}")
        print(f"scene={scene_path}")
        print(
            "actors="
            + json.dumps(
                _serializable(_actor_summary(env)),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        print("info_keys=" + json.dumps(sorted(map(str, info.keys()))))
    finally:
        env.close()


if __name__ == "__main__":
    main()
