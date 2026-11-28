from __future__ import annotations

import copy
from typing import Any

import mujoco
import numpy as np

from humano.dr.material import resolve_render_table_material, with_task_table_material
from humano.utils.geometry import local_aabbs_to_world


_INDEXED_FURNITURE_CLEARANCE_M = 0.5
_TABLE_FALLBACK_MATERIAL = {
    "type": "usd_preview_surface",
    "name": "HumanoReplayTableWalnut",
    "diffuse_color": [0.24, 0.13, 0.07],
    "roughness": 0.82,
    "metallic": 0.0,
    "render_source": "indexed_walnut",
}
_OOD_TABLE_MATERIAL_SOURCE = "ood_background_table"


def _render_table_material(
    recorded_material: dict[str, Any] | None,
    *,
    distribution: str = "train",
    enforce_whitelist: bool = False,
    train_materials: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if recorded_material:
        return resolve_render_table_material(
            recorded_material,
            distribution=distribution,
            enforce_whitelist=enforce_whitelist,
            allowed_materials=train_materials if distribution == "train" else None,
        )
    if enforce_whitelist:
        return resolve_render_table_material(
            {},
            distribution=distribution,
            enforce_whitelist=True,
            allowed_materials=train_materials if distribution == "train" else None,
        )
    return copy.deepcopy(_TABLE_FALLBACK_MATERIAL)


def _prepare_indexed_environment_config(
    env_conf: dict[str, Any],
    *,
    enforce_whitelist: bool = False,
    train_materials: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prepared = copy.deepcopy(env_conf)
    dr_state = prepared.get("dr_state_dict", {})
    scene_state = dr_state.get("scene")
    material_state = dr_state.get("material")
    if isinstance(material_state, dict):
        material_state["ground_material"] = None
    if scene_state is None:
        return prepared
    if not isinstance(scene_state, dict):
        raise ValueError("Indexed environment has an invalid saved scene state")
    table_state = scene_state.get("table")
    if table_state is None:
        return prepared
    if not isinstance(table_state, dict):
        raise ValueError("Indexed environment has an invalid saved task table")
    table2_state = scene_state.get("table2")
    if table2_state is not None and not isinstance(table2_state, dict):
        raise ValueError("Indexed environment has an invalid saved second task table")

    return _prepare_table_material_config(
        prepared,
        enforce_whitelist=enforce_whitelist,
        train_materials=train_materials,
    )


def _prepare_table_material_config(
    env_conf: dict[str, Any],
    *,
    enforce_whitelist: bool = False,
    train_materials: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prepared = copy.deepcopy(env_conf)
    dr_state = prepared.get("dr_state_dict", {})
    scene_state = dr_state.get("scene")
    if not isinstance(scene_state, dict):
        return prepared
    material_state = dr_state.get("material")
    table_state = scene_state.get("table")
    table2_state = scene_state.get("table2")

    ood = prepared.get("ood", {})
    background_material_ood = ood.get("dimension") == "background" and (
        "material_distribution" in ood.get("params", {})
    )
    distribution = ood["params"]["material_distribution"] if background_material_ood else "train"
    enforce_whitelist = (
        enforce_whitelist or bool(prepared.get("eval_scene")) or background_material_ood
    )
    for table_name, state in (("table", table_state), ("table2", table2_state)):
        if not isinstance(state, dict):
            continue
        recorded_material = state.get("material")
        if isinstance(material_state, dict):
            recorded_material = material_state.get(f"{table_name}_material", recorded_material)
        if background_material_ood:
            if not isinstance(recorded_material, dict):
                raise ValueError(f"Background material OOD has no recorded {table_name} material")
            table_material = _render_table_material(
                recorded_material,
                distribution=distribution,
                enforce_whitelist=enforce_whitelist,
                train_materials=train_materials,
            )
            table_material["render_source"] = _OOD_TABLE_MATERIAL_SOURCE
        else:
            table_material = _render_table_material(
                recorded_material,
                distribution=distribution,
                enforce_whitelist=enforce_whitelist,
                train_materials=train_materials,
            )
        if table_material.get("fallback_from") is not None:
            print(
                f"[SonicRender] Replaced {table_name} material "
                f"{table_material['fallback_from']} with {table_material['name']}"
            )
        prepared = with_task_table_material(prepared, table_material, table_name=table_name)
    return prepared


def _install_indexed_scene_randomizer(sonic_env, template_scene) -> None:
    from humano.assets import AssetManager
    from humano.dr.scene import TabletopSceneDR, TabletopSceneDRCfg
    from humano.scenes.indexed import configure_indexed_furniture

    class EvalIndexedSceneDR(TabletopSceneDR):
        def __init__(self) -> None:
            self.cfg = TabletopSceneDRCfg()
            self._inner_state = None

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            table_cfg = state_dict.get("table")
            if not isinstance(table_cfg, dict):
                raise ValueError("Saved scene state has no task table to preserve")
            table2_cfg = state_dict.get("table2")
            if table2_cfg is not None and not isinstance(table2_cfg, dict):
                raise ValueError("Saved scene state has an invalid second task table")
            scene = template_scene.copy()
            configure_indexed_furniture(
                scene,
                table_cfg,
                clearance_m=_INDEXED_FURNITURE_CLEARANCE_M,
                additional_table_cfgs=[table2_cfg] if table2_cfg is not None else None,
            )
            table = AssetManager.create(
                "primitive:box",
                size=table_cfg["size"],
                position=table_cfg["pose"]["position"],
                quaternion=table_cfg["pose"]["quaternion"],
            )
            table.set_material(_render_table_material(table_cfg.get("material")))
            scene.table = table
            if table2_cfg is not None:
                table2 = AssetManager.create(
                    "primitive:box",
                    size=table2_cfg["size"],
                    position=table2_cfg["pose"]["position"],
                    quaternion=table2_cfg["pose"]["quaternion"],
                )
                table2.set_material(_render_table_material(table2_cfg.get("material")))
                scene.table2 = table2
            self._inner_state = scene

        def __call__(self, split: str, *args, **kwargs):
            if self._inner_state is None:
                raise RuntimeError("Indexed scene state was not loaded before task reset")
            return self._inner_state

    sonic_env.task.dr.randomizers["scene"] = EvalIndexedSceneDR()


def _task_geom_ids(mujoco_sim) -> tuple[np.ndarray, np.ndarray]:
    """Partition MuJoCo geoms except the ground plane; floor objects remain included."""
    from humano.assets.primitive import Primitive

    fixed_actors = {
        name
        for name, actor in mujoco_sim.task.layout.actors.items()
        if isinstance(actor, Primitive)
    }
    fixed_geom_names = {f"{name}_geom" for name in fixed_actors}
    moving_ids: list[int] = []
    fixed_ids: list[int] = []
    model = mujoco_sim.mjModel
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name == "ground":
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        target = (
            fixed_ids
            if body_name == "world" or body_name in fixed_actors or geom_name in fixed_geom_names
            else moving_ids
        )
        target.append(geom_id)
    return np.asarray(moving_ids, dtype=np.int32), np.asarray(fixed_ids, dtype=np.int32)


def _moving_geom_ids(mujoco_sim) -> np.ndarray:
    """Select non-ground robot and moving task-object geoms."""
    return _task_geom_ids(mujoco_sim)[0]


def _collect_current_aabbs(mujoco_sim, geom_ids: np.ndarray | None = None) -> dict[str, Any]:
    """Collect current robot and task-object AABBs for eval wall checks."""
    selected_geom_ids = _moving_geom_ids(mujoco_sim) if geom_ids is None else geom_ids
    model = mujoco_sim.mjModel
    data = mujoco_sim.mjData
    if selected_geom_ids.size == 0:
        bounds_min = np.empty((0, 3), dtype=np.float64)
        bounds_max = np.empty((0, 3), dtype=np.float64)
        finite_geom_ids = selected_geom_ids
    else:
        geom_aabbs = np.asarray(model.geom_aabb[selected_geom_ids], dtype=np.float64)
        rotations = np.asarray(data.geom_xmat[selected_geom_ids], dtype=np.float64).reshape(
            -1, 3, 3
        )
        bounds_min, bounds_max = local_aabbs_to_world(
            geom_aabbs[:, :3],
            geom_aabbs[:, 3:],
            np.asarray(data.geom_xpos[selected_geom_ids], dtype=np.float64),
            rotations,
        )
        finite = np.all(np.isfinite(bounds_min), axis=1) & np.all(np.isfinite(bounds_max), axis=1)
        bounds_min = bounds_min[finite]
        bounds_max = bounds_max[finite]
        finite_geom_ids = selected_geom_ids[finite]
    geom_names = {
        int(geom_id): (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
            or f"geom_{int(geom_id)}"
        )
        for geom_id in finite_geom_ids
    }
    return {
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "frame_indices": np.zeros(finite_geom_ids.size, dtype=np.int32),
        "geom_ids": finite_geom_ids,
        "geom_names": geom_names,
    }


def _validate_indexed_trajectory_bounds(scene, trajectory_aabbs: dict[str, Any]) -> dict[str, Any]:
    bounds_min = np.asarray(trajectory_aabbs["bounds_min"], dtype=np.float64)
    bounds_max = np.asarray(trajectory_aabbs["bounds_max"], dtype=np.float64)
    bounds = scene.bounds
    safe_min = np.asarray(bounds.min_xy, dtype=np.float64) + bounds.margin_m
    safe_max = np.asarray(bounds.max_xy, dtype=np.float64) - bounds.margin_m
    valid = np.all(bounds_min[:, :2] >= safe_min, axis=1) & np.all(
        bounds_max[:, :2] <= safe_max, axis=1
    )
    if not np.all(valid):
        index = int(np.flatnonzero(~valid)[0])
        geom_id = int(trajectory_aabbs["geom_ids"][index])
        geom_name = trajectory_aabbs["geom_names"].get(geom_id, f"geom_{geom_id}")
        frame_index = int(trajectory_aabbs["frame_indices"][index])
        raise ValueError(
            "Indexed scene bounds violation before render: "
            f"scene={scene.uid} frame={frame_index} geom={geom_name} "
            f"aabb_min={bounds_min[index].tolist()} aabb_max={bounds_max[index].tolist()} "
            f"safe_xy=[{safe_min.tolist()}, {safe_max.tolist()}]"
        )

    obstacle_clearance = float(bounds.margin_m)
    for obstacle in scene.immutable_obstacles:
        obstacle_min = np.asarray(obstacle.min_xyz, dtype=np.float64) - obstacle_clearance
        obstacle_max = np.asarray(obstacle.max_xyz, dtype=np.float64) + obstacle_clearance
        overlaps = np.all(bounds_max >= obstacle_min, axis=1) & np.all(
            bounds_min <= obstacle_max, axis=1
        )
        if not np.any(overlaps):
            continue
        index = int(np.flatnonzero(overlaps)[0])
        geom_id = int(trajectory_aabbs["geom_ids"][index])
        geom_name = trajectory_aabbs["geom_names"].get(geom_id, f"geom_{geom_id}")
        frame_index = int(trajectory_aabbs["frame_indices"][index])
        raise ValueError(
            "Indexed scene immutable obstacle collision before render: "
            f"scene={scene.uid} obstacle={obstacle.prim_path} "
            f"category={obstacle.category} frame={frame_index} geom={geom_name} "
            f"aabb_min={bounds_min[index].tolist()} aabb_max={bounds_max[index].tolist()} "
            f"obstacle_min={obstacle_min.tolist()} obstacle_max={obstacle_max.tolist()}"
        )
    return {
        "policy": bounds.policy,
        "validated": bounds.validated,
        "source": bounds.source,
        "safe_min_xy": safe_min.tolist(),
        "safe_max_xy": safe_max.tolist(),
        "checked_aabb_count": int(bounds_min.shape[0]),
        "immutable_obstacle_count": len(scene.immutable_obstacles),
        "immutable_obstacle_clearance_m": obstacle_clearance,
        "immutable_obstacle_collision_free": True,
    }


def _validate_current_indexed_scene(
    scene, mujoco_sim, geom_ids: np.ndarray | None = None
) -> dict[str, Any]:
    return _validate_indexed_trajectory_bounds(
        scene,
        _collect_current_aabbs(mujoco_sim, geom_ids),
    )
