"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

import copy
import json
import os
from typing import Dict, Tuple

import carb
import carb.settings
import numpy as np
import omni.isaac.core.utils.prims as isaacsim_prims
import omni.isaac.core.utils.stage as isaacsim_stage
import omni.kit.commands
import omni.kit.primitive.mesh
import omni.physx
import omni.replicator.core as rep
import omni.usd
import transforms3d as t3d
from curobo.types.state import JointState
from isaacsim.core.prims import SingleArticulation
from isaacsim.util.debug_draw import _debug_draw
from omni.isaac.core.objects import GroundPlane, cuboid, sphere
from omni.isaac.core.prims import GeometryPrim, RigidPrim, XFormPrim
from omni.isaac.core.robots.robot import Robot as IsaacRobot
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.world.world import World

# from omni.isaac.core.materials import VisualMaterial
from omni.isaac.sensor import Camera as IssacCamera
from omni.kit.material.library import create_mdl_material
from pxr import Gf, Sdf, Semantics, Tf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation

import humano
import humano.scenes

# from humano.core import sensors
import humano.sensors
from humano.core.actor import ArticulatedObjectActor, ObjectActor, VisualFrame, VisualGrasp
from humano.core.robot import Robot
from humano.core.simulator import Simulator
from humano.core.task import Task
from humano.engines.render_quality import get_render_quality_preset
from humano.scenes.hssd import HssdSceneManager, HssdSuite, table_top_center_and_yaw
from humano.scenes.indexed import IndexedBackgroundScene, IndexedVisualItem, detached_cabinet_fronts
from humano.utils import env_flag, resolve_data_path
from humano.utils.articulated_joint_state import resolve_articulated_joint_positions
from humano.utils.geometry import (
    compose_pose_with_offset,
    overlapping_aabb_indices,
    polygon_intersects_aabb_xy,
)

# import isaacsim

# from isaacsim.core.api.world.world import World
# from isaacsim.core.api.robots.robot import Robot
# from isaacsim.core.prims import RigidPrim, GeometryPrim, XFormPrim
# from isaacsim.sensors.camera import Camera
# import omni.replicator.core as rep
# import carb.settings
# import omni.usd
# import omni.kit.commands

# import isaacsim.core.utils.prims as isaacsim_prims
# import isaacsim.core.utils.stage as isaacsim_stage


# import importlib.resources as res
# from importlib.resources import as_file


def euler_angles_to_quats(
    euler_angles: np.ndarray, degrees: bool = False, extrinsic: bool = True, device=None
) -> np.ndarray:
    if extrinsic:
        order = "xyz"
    else:
        order = "XYZ"
    rot = Rotation.from_euler(order, euler_angles, degrees=degrees)
    result = rot.as_quat()
    if len(result.shape) == 1:
        result = result[[3, 0, 1, 2]]
    else:
        result = result[:, [3, 0, 1, 2]]
    return result


def compute_obb(
    bbox_cache: UsdGeom.BBoxCache, prim_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the Oriented Bounding Box (OBB) of a prim

    .. note::

        * The OBB does not guarantee the smallest possible bounding box, it rotates and scales the default AABB.
        * The rotation matrix incorporates any scale factors applied to the object.
        * The `half_extent` values do not include these scaling effects.

    Args:
        bbox_cache (UsdGeom.BBoxCache): USD Bounding Box Cache object to use for computation
        prim_path (str): Prim path to compute OBB for

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following OBB information:
            - The centroid of the OBB as a NumPy array.
            - The axes of the OBB as a 2D NumPy array, where each row represents a different axis.
            - The half extent of the OBB as a NumPy array.

    Example:

    .. code-block:: python

        >>> import omni.isaac.core.utils.bounds as bounds_utils
        >>>
        >>> # 1 stage unit length cube centered at (0.0, 0.0, 0.0)
        >>> cache = bounds_utils.create_bbox_cache()
        >>> centroid, axes, half_extent = bounds_utils.compute_obb(cache, prim_path="/World/Cube")
        >>> centroid
        [0. 0. 0.]
        >>> axes
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
        >>> half_extent
        [0.5 0.5 0.5]
        >>>
        >>> # the same cube rotated 45 degrees around the z-axis
        >>> cache = bounds_utils.create_bbox_cache()
        >>> centroid, axes, half_extent = bounds_utils.compute_obb(cache, prim_path="/World/Cube")
        >>> centroid
        [0. 0. 0.]
        >>> axes
        [[ 0.70710678  0.70710678  0.        ]
         [-0.70710678  0.70710678  0.        ]
         [ 0.          0.          1.        ]]
        >>> half_extent
        [0.5 0.5 0.5]
    """
    # Compute the BBox3d for the prim
    prim = get_prim_at_path(prim_path)
    bound = bbox_cache.ComputeWorldBound(prim)

    # Compute the translated centroid of the world bound
    centroid = bound.ComputeCentroid()

    # Compute the axis vectors of the OBB
    # NOTE: The rotation matrix incorporates the scale factors applied to the object
    rotation_matrix = bound.GetMatrix().ExtractRotationMatrix()
    x_axis = rotation_matrix.GetRow(0)
    y_axis = rotation_matrix.GetRow(1)
    z_axis = rotation_matrix.GetRow(2)

    # Compute the half-lengths of the OBB along each axis
    # NOTE the size/extent values do not include any scaling effects
    half_extent = bound.GetRange().GetSize() * 0.5

    return np.array([*centroid]), np.array([[*x_axis], [*y_axis], [*z_axis]]), np.array(half_extent)


def get_obb_corners(centroid: np.ndarray, axes: np.ndarray, half_extent: np.ndarray) -> np.ndarray:
    """Computes the corners of the Oriented Bounding Box (OBB) from the given OBB information

    Args:
        centroid (np.ndarray): The centroid of the OBB as a NumPy array.
        axes (np.ndarray): The axes of the OBB as a 2D NumPy array, where each row represents a different axis.
        half_extent (np.ndarray): The half extent of the OBB as a NumPy array.

    Returns:
        np.ndarray: NumPy array of shape (8, 3) containing each corner location of the OBB

        :math:`c_0 = (x_{min}, y_{min}, z_{min})`
        |br| :math:`c_1 = (x_{min}, y_{min}, z_{max})`
        |br| :math:`c_2 = (x_{min}, y_{max}, z_{min})`
        |br| :math:`c_3 = (x_{min}, y_{max}, z_{max})`
        |br| :math:`c_4 = (x_{max}, y_{min}, z_{min})`
        |br| :math:`c_5 = (x_{max}, y_{min}, z_{max})`
        |br| :math:`c_6 = (x_{max}, y_{max}, z_{min})`
        |br| :math:`c_7 = (x_{max}, y_{max}, z_{max})`

    Example:

    .. code-block:: python

        >>> import omni.isaac.core.utils.bounds as bounds_utils
        >>>
        >>> cache = bounds_utils.create_bbox_cache()
        >>> centroid, axes, half_extent = bounds_utils.compute_obb(cache, prim_path="/World/Cube")
        >>> bounds_utils.get_obb_corners(centroid, axes, half_extent)
        [[-0.5 -0.5 -0.5]
        [-0.5 -0.5  0.5]
        [-0.5  0.5 -0.5]
        [-0.5  0.5  0.5]
        [ 0.5 -0.5 -0.5]
        [ 0.5 -0.5  0.5]
        [ 0.5  0.5 -0.5]
        [ 0.5  0.5  0.5]]
    """
    corners = [
        centroid - axes[0] * half_extent[0] - axes[1] * half_extent[1] - axes[2] * half_extent[2],
        centroid - axes[0] * half_extent[0] - axes[1] * half_extent[1] + axes[2] * half_extent[2],
        centroid - axes[0] * half_extent[0] + axes[1] * half_extent[1] - axes[2] * half_extent[2],
        centroid - axes[0] * half_extent[0] + axes[1] * half_extent[1] + axes[2] * half_extent[2],
        centroid + axes[0] * half_extent[0] - axes[1] * half_extent[1] - axes[2] * half_extent[2],
        centroid + axes[0] * half_extent[0] - axes[1] * half_extent[1] + axes[2] * half_extent[2],
        centroid + axes[0] * half_extent[0] + axes[1] * half_extent[1] - axes[2] * half_extent[2],
        centroid + axes[0] * half_extent[0] + axes[1] * half_extent[1] + axes[2] * half_extent[2],
    ]
    return np.array(corners)


class IsaacSimSimulator(Simulator):
    SCENE_PRIM_PATH = "/World/scene"

    def __init__(self, task: Task, render_hz: int = 30, headless: bool = False) -> None:
        self.render_quality = get_render_quality_preset()
        self._config_isaac()
        # self.render_hz = render_hz # @deprecated

        # self.layout = layout
        self.task = task
        self.render_hz = (
            task.metadata["render_hz"] if "render_hz" in self.task.metadata else render_hz
        )

        self.step_id = 0
        self.is_scene_create = False
        self.is_isaac_reset = False
        self.headless = headless

        self.visulize_spheres = self.task.robot.visulize_spheres

        self.need_gravity = self.task.metadata.get("need_gravity", False)

    def reset(self) -> None:
        self._config_isaac()
        self.step_id = 0
        if hasattr(self, "cameras"):
            for camera in self.cameras.values():
                camera.initialize()

        self.is_isaac_reset = False

    def reset_render_accumulation(self) -> None:
        omni.usd.get_context().reset_renderer_accumulation()

    def _config_isaac(self):
        success, result = omni.kit.commands.execute(
            "ChangeSetting", path="/rtx/directLighting/sampledLighting/autoEnable", value=False
        )
        if not success:
            print("warning: set autoEnable sampleed lighting falied", result)
        zero_delay = env_flag("HUMANO_ISAAC_ZERO_DELAY", default=True)
        disable_throttling_async = env_flag("HUMANO_ISAAC_DISABLE_THROTTLING_ASYNC", default=True)

        # Disable capture on play. Delay/throttling settings honor env flags so
        # sync diagnostics can intentionally test async behavior.
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
        carb.settings.get_settings().set("/omni/replicator/asyncRendering", False)
        carb.settings.get_settings().set("/app/asyncRendering", False)
        carb.settings.get_settings().set("/app/hydraEngine/waitIdle", bool(zero_delay))
        carb.settings.get_settings().set(
            "/app/updateOrder/checkForHydraRenderComplete",
            1000 if zero_delay else 0,
        )
        carb.settings.get_settings().set(
            "/exts/isaacsim.core.throttling/enable_async",
            not disable_throttling_async,
        )
        if self.render_quality.replicator_mode == "pathtraced":
            rep.settings.set_render_pathtraced(
                samples_per_pixel=self.render_quality.path_tracing_spp
            )
        else:
            rep.settings.set_render_rtx_realtime(
                antialiasing=self.render_quality.anti_aliasing
            )
        # rep.orchestrator.set_capture_on_play(False) # Data will be captured manually using step
        # carb.settings.get_settings().set_bool("/omni/replicator/captureMotionBlur", 0)
        # carb.settings.get_settings().set_bool("/rtx/post/motionblur/enabled", 0)

    def update_layout(self) -> None:
        if not self.is_scene_create:
            self._setup_scene()
            self.is_scene_create = True

        self.__update_scene()
        self.__update_robot()
        self.__update_cameras()
        self.__update_lights()
        self.update_visuals()

        # 5. add objects
        self.__reset_objects()
        self.current_visible_objects = []
        # object_shader_params = copy.deepcopy(self.task.layout.material_info["object_shader_params"])
        for obj_name, obj_info in self.task.layout.actors.items():
            if not isinstance(obj_info, ObjectActor) and not isinstance(
                obj_info, ArticulatedObjectActor
            ):
                continue

            if isinstance(obj_info, ArticulatedObjectActor):
                self.add_articulated_object(obj_name, obj_info)

            else:
                simulation_name = obj_info.simulation_name
                if simulation_name not in self.objects:
                    self.__create_object(obj_name, obj_info)

                self.__update_object(obj_name, obj_info)
                """ if obj_info["bTarget"]:
                    self.target_obj_id = obj_info["id"]
                    self.target_obj = self.objects[obj_name] """
                self.current_visible_objects.append(self.objects[simulation_name])

        self.step_id = 0
        """ since isaac is already reset, we reset init robot qpos here
        """
        if self.is_isaac_reset and self.is_scene_create:
            joint_state = self.task.robot.init_joint_states
            joint_indices = []
            qpos = []
            # print("syncing isaac to mujoco qpos:", joint_state)
            for jname, jpos in joint_state.items():
                isaac_jname = self.task.robot.jname_mujoco_to_isaac(jname)
                joint_indices.append(self.robot.get_dof_index(isaac_jname))
                qpos.append(jpos)
            self.robot.set_joint_positions(qpos, joint_indices=np.arange(len(joint_state)))

    def _setup_scene(self, move_surface_to_origin=True):
        assert not self.is_scene_create
        # NOTE: set correct physics_dt, rendering_dt to only affects isaacsim rendering quality

        self.world = World(
            physics_dt=1 / (self.render_hz * 10), rendering_dt=1 / (self.render_hz), backend="numpy"
        )
        self.workspace_prim_path = f"/World/workspace"
        self.workspace = XFormPrim(prim_path=self.workspace_prim_path)

        self.objects = {}
        self.current_visible_objects = []
        self.active_tableground_materials = []
        self.active_tableground_shaders = []
        self.scenes = {}
        self.cameras = {}
        self.lights = []
        self.articulated_objects = {}
        self.articulated_object_roots = {}
        self.articulated_visual_objects = {}
        self.articulated_visual_bodies = {}
        self.articulated_root_offsets = {}
        self.articulated_joint_names_by_link = {}
        self._hssd_collision_hidden: dict[str, set[str]] = {}
        self._indexed_collision_hidden: dict[str, set[str]] = {}

        self.add_robot()
        self.add_cameras()
        self.add_lights()

        self.__pre_add_objects()
        self.spheres = None

    def __update_scene(self, move_surface_to_origin=True):
        scene = getattr(self.task.layout, "scene", None)
        if scene is None:
            return
        scene_uid = getattr(scene, "cache_key", scene.uid)

        for uid, (_scene, _) in self.scenes.items():
            if scene_uid != uid:
                _scene.GetAttribute("visibility").Set("invisible")
                # scene.GetAttribute("xformOp:translate").Set((0, 0, 0))

        if isinstance(scene, IndexedBackgroundScene):
            from humano.assets.primitive import Primitive
            self._update_indexed_background(scene)
            table_box = self.task.layout.actors.get("table")
            if isinstance(table_box, Primitive):
                self._setup_table(f"{self.workspace_prim_path}/table_cuboid", table_box)
            return

        assert isinstance(scene, HssdSuite), "not supported scene type yet"

        scene_uid = scene.uid

        scene_prim_path = f"{self.SCENE_PRIM_PATH}/s_{scene.uid.replace(':', '_')}"
        surface_prim_path = scene.conf["surface"]["prim_path"].replace("/World", scene_prim_path)

        has_surface2 = "surface2" in scene.conf and scene.conf["surface2"] is not None
        surface2_prim_path = None
        if has_surface2:
            surface2_prim_path = scene.conf["surface2"]["prim_path"].replace(
                "/World", scene_prim_path
            )

        if scene_uid not in self.scenes:
            # self.add_scene(move_surface_to_origin)
            if isinstance(scene, humano.scenes.ShowHouse):
                raise NotImplementedError("ShowHouse scene is not implemented yet.")

            #### START adding HSSD scenes ####
            from pxr import UsdGeom, UsdPhysics

            # data_dir = resolve_data_path
            try:
                data_dir = resolve_data_path(
                    scene.data_dir, auto_download=True
                )  # f"scenes/hssd/{scene_name}"
            except FileNotFoundError:
                # put download logic into SceneManager
                from humano.scenes import SceneManager

                SceneManager.get(scene.uid.split(":")[0]).load(scene.uid)
                data_dir = resolve_data_path(scene.data_dir)

            env_url = os.path.abspath(f"{data_dir}/{scene.name}.usd")
            isaacsim_stage.add_reference_to_stage(usd_path=env_url, prim_path=scene_prim_path)
            scene_prim = self.world.stage.GetPrimAtPath(scene_prim_path)

            surface_prim = self.world.stage.GetPrimAtPath(surface_prim_path)
            surface_prim.GetAttribute("visibility").Set("invisible")

            if has_surface2:
                surface2_prim = self.world.stage.GetPrimAtPath(surface2_prim_path)
                surface2_prim.GetAttribute("visibility").Set("invisible")

            if not scene_prim.GetAttribute("xformOp:translate"):
                UsdGeom.Xformable(scene_prim).AddTranslateOp()  # type: ignore
            if not scene_prim.GetAttribute("xformOp:rotateXYZ"):
                UsdGeom.Xformable(scene_prim).AddRotateXYZOp()  # type: ignore
            if not scene_prim.GetAttribute("xformOp:scale"):
                UsdGeom.Xformable(scene_prim).AddScaleOp()  # type: ignore

            # scene: HssdSuite = self.task.layout.scene
            scene_prim.GetAttribute("xformOp:rotateXYZ").Set(tuple(scene.center_orientation))

            scale = scene.conf["scale"]
            scene_prim.GetAttribute("xformOp:scale").Set((scale, scale, scale))

            surface_obb = self.calc_surface_center(surface_prim)
            self.scenes[scene.uid] = [scene_prim, surface_obb]  # hssd_env # store it
        else:
            scene_prim = self.world.stage.GetPrimAtPath(scene_prim_path)
            surface_prim = self.world.stage.GetPrimAtPath(surface_prim_path)
            surface_prim.GetAttribute("visibility").Set("invisible")
            self.scenes[scene.uid][0] = scene_prim  # hssd_env # store it
            surface_obb = self.scenes[scene.uid][1]

            scale = scene.conf["scale"]
            scene_prim.GetAttribute("xformOp:scale").Set((scale, scale, scale))
            scene_prim.GetAttribute("xformOp:translate").Set((0, 0, 0))
            scene_prim.GetAttribute("xformOp:rotateXYZ").Set(tuple(scene.center_orientation))
            surface_obb = self.calc_surface_center(surface_prim)

            if has_surface2:
                surface2_prim = self.world.stage.GetPrimAtPath(surface2_prim_path)
                surface2_prim.GetAttribute("visibility").Set("invisible")

        ceiling = scene_prim.GetPrimAtPath(f"{scene_prim_path}/ceilings")
        ceiling.GetAttribute("visibility").Set("visible")  # hide ceiling for better visualization

        if move_surface_to_origin:
            surface_center_position = -surface_obb["position"] + np.array(
                scene.center_offset, dtype=np.float32
            )  # self._config.hssd.center_offset
            if self.task.robot.uid == "g1_sonic":
                # FIXME backward compatibility: robot touches the ground
                surface_center_position[2] = 0.0
            scene_prim.GetAttribute("xformOp:translate").Set(tuple(surface_center_position))

        scene_prim.GetAttribute("visibility").Set("visible")

        table_box = self.task.layout.actors.get("table")
        if table_box is not None:
            self._setup_table(f"{self.workspace_prim_path}/table_cuboid", table_box)

        table2_box = self.task.layout.actors.get("table2")
        if table2_box is not None:
            self._setup_table(f"{self.workspace_prim_path}/table2_cuboid", table2_box)

    @staticmethod
    def _indexed_scene_prim_path(scene: IndexedBackgroundScene) -> str:
        scene_key = scene.cache_key.replace(":", "_").replace("/", "_").replace("@", "_").replace("-", "_")
        return f"{IsaacSimSimulator.SCENE_PRIM_PATH}/s_{scene_key}"

    def _apply_indexed_scene_lighting(self, scene_prim):
        """Execute LightingDR's policy and restore scene lights on a cached ID reset."""
        light_dr = self.task.dr.get_randomizer("lighting")
        disabled = light_dr is not None and light_dr.disable_authored_scene_lights
        originals = dict(getattr(self, "_authored_scene_light_intensities", {}))
        disabled_paths = []
        for prim in Usd.PrimRange(scene_prim):
            if not prim.HasAPI(UsdLux.LightAPI):
                continue
            path = str(prim.GetPath())
            intensity = UsdLux.LightAPI(prim).GetIntensityAttr()
            if disabled:
                if path not in originals:
                    originals[path] = intensity.Get()
                intensity.Set(0.0)
                disabled_paths.append(path)
            elif path in originals:
                intensity.Set(originals.pop(path))
        self._authored_scene_light_intensities = originals
        return disabled_paths

    def _update_indexed_background(self, scene: IndexedBackgroundScene) -> None:
        components = scene.metadata.get("visual_components", [])
        if components and (
            not np.allclose(components[0]["position"], [0, 0, 0])
            or not np.allclose(components[0]["quaternion"], [1, 0, 0, 0])
            or os.path.abspath(self.resolve_data_path(components[0]["usd_path"]))
            != os.path.abspath(scene.usd_path)
        ):
            raise ValueError("Primary visual component must match the indexed background origin")
        scene_prim_path = self._indexed_scene_prim_path(scene)
        cache_key = scene.cache_key
        if cache_key not in self.scenes:
            isaacsim_stage.add_reference_to_stage(
                usd_path=os.path.abspath(scene.usd_path),
                prim_path=scene_prim_path,
            )
            for index, item in enumerate(scene.metadata.get("visual_components", [])[1:], 1):
                component_path = f"{scene_prim_path}/component_{index}"
                isaacsim_stage.add_reference_to_stage(
                    usd_path=self.resolve_data_path(item["usd_path"]), prim_path=component_path
                )
                component = self.world.stage.GetPrimAtPath(component_path)
                transform = UsdGeom.Xformable(component)
                transform.ClearXformOpOrder()
                transform.AddTranslateOp().Set(Gf.Vec3d(*item["position"]))
                q = item["quaternion"]
                transform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                    Gf.Quatd(q[0], Gf.Vec3d(*q[1:]))
                )
            scene_prim = self.world.stage.GetPrimAtPath(scene_prim_path)
            if not scene_prim.IsValid():
                raise RuntimeError(
                    f"Failed to reference indexed scene {scene.uid} from {scene.usd_path}"
                )
            self.scenes[cache_key] = [scene_prim, None]
        else:
            scene_prim = self.world.stage.GetPrimAtPath(scene_prim_path)
            self.scenes[cache_key][0] = scene_prim

        xformable = UsdGeom.Xformable(scene_prim)
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp()
        orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
        scale_op = xformable.AddScaleOp()
        transform = scene.transform
        translate_op.Set(Gf.Vec3d(*transform.translation))
        orient_op.Set(Gf.Quatd(*transform.rotation_wxyz))
        scale_op.Set(Gf.Vec3d(*transform.scale))

        root_path = str(scene_prim.GetPath())
        profile = scene.visual_profile
        # Older iTHOR indexes omit some drawer fascias authored as independent doors.
        # Read their bounds from the current stage so that frozen cabinet masks also hide them.
        hidden = set(profile.hidden_prim_paths)
        missing_fronts = []
        if any(
            item.category == "cabinet" and item.prim_path in hidden
            for item in profile.background_items
        ):
            indexed = {item.prim_path for item in profile.background_items}
            bounds = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                False,
                True,
            )
            for path in profile.visible_prim_paths:
                name = path.rsplit("/", 1)[-1]
                if path in indexed or not name.startswith("door_"):
                    continue
                prim = self.world.stage.GetPrimAtPath(path.replace("/World", root_path, 1))
                if not prim.IsValid():
                    continue
                box = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
                if not box.IsEmpty():
                    low, high = box.GetMin(), box.GetMax()
                    missing_fronts.append(
                        IndexedVisualItem(
                            path, "door", list(low)[:2], list(high)[:2], low[2], high[2]
                        )
                    )
        extra_hidden = detached_cabinet_fronts([*profile.background_items, *missing_fronts], hidden)
        for source_path in hidden | extra_hidden:
            prim = self.world.stage.GetPrimAtPath(source_path.replace("/World", root_path, 1))
            if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeInvisible()
        for source_path in profile.visible_prim_paths:
            if source_path in extra_hidden:
                continue
            prim = self.world.stage.GetPrimAtPath(source_path.replace("/World", root_path, 1))
            if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeVisible()
        for prim in Usd.PrimRange(scene_prim):
            lowered = prim.GetName().lower()
            if prim.IsA(UsdGeom.Imageable) and (
                "_collision_" in lowered
                or lowered.endswith("_collision")
                or (profile.hide_ceiling and ("ceiling" in lowered or "roof" in lowered))
            ):
                UsdGeom.Imageable(prim).MakeInvisible()
        if profile.disable_authored_physics:
            self._disable_background_physics(scene_prim)
        UsdGeom.Imageable(scene_prim).MakeVisible()
        fill = scene.metadata.get("render_fill_lights")
        if fill:
            for index, position in enumerate(fill["positions"]):
                light = UsdLux.RectLight.Define(self.world.stage, f"{root_path}/RenderFill{index}")
                light.CreateWidthAttr(fill["width"])
                light.CreateHeightAttr(fill["height"])
                light.CreateIntensityAttr(fill["intensity"])
                light.CreateNormalizeAttr(False)
                light_xform = UsdGeom.Xformable(light.GetPrim())
                light_xform.ClearXformOpOrder()
                light_xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        self._apply_indexed_scene_lighting(scene_prim)

    def replace_indexed_background(self, scene: IndexedBackgroundScene) -> None:
        """Replace only the Isaac visual background, preserving the task layout."""
        if self.task.layout.actors.get("table") is None:
            table_prim_path = f"{self.workspace_prim_path}/table_cuboid"
            table_prim = self.world.stage.GetPrimAtPath(table_prim_path)
            if table_prim.IsValid():
                self.world.stage.RemovePrim(table_prim_path)
        self._update_indexed_background(scene)
        active_path = self._indexed_scene_prim_path(scene)
        for child in self.world.stage.GetPrimAtPath(self.SCENE_PRIM_PATH).GetChildren():
            if str(child.GetPath()) != active_path:
                UsdGeom.Imageable(child).MakeInvisible()

    def filter_indexed_background_for_trajectory(
        self,
        scene: IndexedBackgroundScene,
        *,
        trajectory_aabbs: dict[str, object],
        dataset_trajectory_aabbs: dict[str, object] | None = None,
        clearance_m: float,
        restore_hidden_assets: bool = False,
    ) -> dict[str, object]:
        from omni.isaac.core.utils.bounds import create_bbox_cache

        if clearance_m < 0:
            raise ValueError("Indexed background clearance must be non-negative")
        root_path = self._indexed_scene_prim_path(scene)
        for prim_path in self._indexed_collision_hidden.get(scene.cache_key, set()):
            prim = self.world.stage.GetPrimAtPath(prim_path)
            if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeVisible()
        self._indexed_collision_hidden[scene.cache_key] = set()

        indexed_items = {item.prim_path: item for item in scene.visual_profile.background_items}
        item_categories = {path: item.category for path, item in indexed_items.items()}
        task_alignment = scene.metadata.get("task_alignment") or {}
        anchor_path = task_alignment.get("hidden_anchor_prim_path")
        protected_hidden_paths = {str(anchor_path)} if anchor_path else set()
        hidden_profile_paths = set(scene.visual_profile.hidden_prim_paths)
        visible_paths = sorted(
            (
                set(scene.visual_profile.visible_prim_paths)
                | (hidden_profile_paths if restore_hidden_assets else set())
            )
            - protected_hidden_paths
        )
        restored_runtime_paths = []
        if restore_hidden_assets:
            for source_path in visible_paths:
                if source_path not in hidden_profile_paths:
                    continue
                prim = self.world.stage.GetPrimAtPath(source_path.replace("/World", root_path, 1))
                if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(prim).MakeVisible()
                    restored_runtime_paths.append(source_path)
        bbox_cache = create_bbox_cache()
        assets: dict[str, dict[str, object]] = {}
        skipped: list[dict[str, str]] = []
        envelope_prefixes = (
            "ceiling_",
            "doorframe_",
            "floor_",
            "room_",
            "roof_",
            "wall_",
            "windowframe_",
            "windowpane_",
        )
        room_area = (scene.bounds.max_xy[0] - scene.bounds.min_xy[0]) * (
            scene.bounds.max_xy[1] - scene.bounds.min_xy[1]
        )
        for source_path in visible_paths:
            source_name = source_path.rsplit("/", 1)[-1].lower()
            if source_name.startswith(envelope_prefixes):
                skipped.append({"prim_path": source_path, "reason": "room_envelope"})
                continue
            item = indexed_items.get(source_path)
            if item is not None and item.category == "mesh" and item.max_z is not None:
                span_x = item.max_xy[0] - item.min_xy[0]
                span_y = item.max_xy[1] - item.min_xy[1]
                item_area = span_x * span_y
                floor_patch = (
                    item.min_z is not None
                    and item.max_z - item.min_z <= 0.25
                    and min(span_x, span_y) >= 3.0
                )
                if abs(item.max_z - scene.bounds.floor_z) <= 0.05 and (
                    item_area >= 0.5 * room_area or floor_patch
                ):
                    skipped.append({"prim_path": source_path, "reason": "floor_support_layer"})
                    continue
            prim_path = source_path.replace("/World", root_path, 1)
            prim = self.world.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
                skipped.append({"prim_path": source_path, "reason": "missing_prim"})
                continue
            geometry_bounds: list[dict[str, object]] = []
            for candidate in Usd.PrimRange(prim):
                if not candidate.IsA(UsdGeom.Boundable):
                    continue
                lowered = candidate.GetName().lower()
                if "_collision_" in lowered or lowered.endswith("_collision"):
                    continue
                imageable = UsdGeom.Imageable(candidate)
                if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                    continue
                if candidate.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(candidate)
                    points = mesh.GetPointsAttr().Get() or []
                    counts = mesh.GetFaceVertexCountsAttr().Get() or []
                    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
                    if points and counts and indices:
                        transform = UsdGeom.Xformable(candidate).ComputeLocalToWorldTransform(
                            Usd.TimeCode.Default()
                        )
                        world_points = np.asarray(
                            [
                                transform.Transform(
                                    Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                                )
                                for point in points
                            ],
                            dtype=np.float64,
                        )
                        offset = 0
                        for count in counts:
                            count = int(count)
                            face_indices = np.asarray(indices[offset : offset + count], dtype=np.int64)
                            offset += count
                            if count < 3:
                                continue
                            face_points = world_points[face_indices]
                            geometry_bounds.append(
                                {
                                    "prim_path": str(candidate.GetPath()),
                                    "min": face_points.min(axis=0),
                                    "max": face_points.max(axis=0),
                                    "kind": "mesh_face",
                                    "xy_polygon": face_points[:, :2],
                                }
                            )
                        continue
                try:
                    candidate_min, candidate_max = self._world_aligned_aabb(
                        bbox_cache, candidate
                    )
                except ValueError:
                    continue
                geometry_bounds.append(
                    {
                        "prim_path": str(candidate.GetPath()),
                        "min": candidate_min,
                        "max": candidate_max,
                        "kind": "boundable_aabb",
                    }
                )
            if not geometry_bounds:
                skipped.append({"prim_path": source_path, "reason": "empty_bound"})
                continue
            combined_min = np.min(
                np.stack([np.asarray(value["min"]) for value in geometry_bounds]), axis=0
            )
            combined_max = np.max(
                np.stack([np.asarray(value["max"]) for value in geometry_bounds]), axis=0
            )
            assets[source_path] = {
                "prim_path": prim_path,
                "category": item_categories.get(source_path, source_name.split("_", 1)[0]),
                "geometry": geometry_bounds,
                "min": combined_min,
                "max": combined_max,
            }

        trajectory_sources = [("current_trajectory", trajectory_aabbs)]
        if dataset_trajectory_aabbs is not None:
            trajectory_sources.append(("training_dataset", dataset_trajectory_aabbs))
        overlaps_by_path: dict[str, list[dict[str, object]]] = {}
        for source_path, asset in assets.items():
            source_overlaps: list[dict[str, object]] = []
            for source_name, source in trajectory_sources:
                bounds_min = np.asarray(source["bounds_min"], dtype=np.float64)
                bounds_max = np.asarray(source["bounds_max"], dtype=np.float64)
                broad = overlapping_aabb_indices(
                    np.asarray(asset["min"]),
                    np.asarray(asset["max"]),
                    bounds_min,
                    bounds_max,
                    clearance=clearance_m,
                )
                if broad.size == 0:
                    continue
                first_geometry = None
                first_index = None
                for geometry in asset["geometry"]:
                    indices = overlapping_aabb_indices(
                        np.asarray(geometry["min"]),
                        np.asarray(geometry["max"]),
                        bounds_min,
                        bounds_max,
                        clearance=clearance_m,
                    )
                    if not indices.size:
                        continue
                    polygon = geometry.get("xy_polygon")
                    first_index = next(
                        (
                            int(index)
                            for index in indices
                            if polygon is None
                            or polygon_intersects_aabb_xy(
                                polygon,
                                bounds_min[index, :2] - clearance_m,
                                bounds_max[index, :2] + clearance_m,
                            )
                        ),
                        None,
                    )
                    if first_index is not None:
                        first_geometry = geometry
                        break
                if first_geometry is None or first_index is None:
                    continue
                geom_ids = np.asarray(source["geom_ids"], dtype=np.int32)
                frame_indices = np.asarray(source["frame_indices"], dtype=np.int32)
                geom_id = int(geom_ids[first_index])
                geom_names = source["geom_names"]
                assert isinstance(geom_names, dict)
                entry: dict[str, object] = {
                    "source": source_name,
                    "geometry_prim_path": str(first_geometry["prim_path"]),
                    "geometry_kind": str(first_geometry["kind"]),
                    "first_overlap_frame": int(frame_indices[first_index]),
                    "first_overlap_geom_id": geom_id,
                    "first_overlap_geom_name": str(
                        geom_names.get(geom_id, f"geom_{geom_id}")
                    ),
                }
                episode_indices = source.get("episode_indices")
                if episode_indices is not None:
                    entry["first_overlap_episode"] = int(
                        np.asarray(episode_indices, dtype=np.int32)[first_index]
                    )
                source_overlaps.append(entry)
            if source_overlaps:
                overlaps_by_path[source_path] = source_overlaps

        def family(path: str) -> str:
            parts = path.rsplit("_", 3)
            return parts[0] if len(parts) == 4 and all(v.isdigit() for v in parts[1:]) else path

        structural = {
            "cabinet",
            "counter",
            "countertop",
            "cube",
            "dishwasher",
            "drawer",
            "faucet",
            "fixture",
            "kitchen_counter",
            "knob",
            "oven",
            "refrigerator",
            "sink",
            "tap",
        }
        support_surfaces = {
            "basket",
            "bench",
            "bin",
            "bowl",
            "box",
            "cabinet",
            "cart",
            "coffeetable",
            "counter",
            "countertop",
            "crate",
            "desk",
            "diningtable",
            "island",
            "kitchen_counter",
            "nightstand",
            "plate",
            "shelf",
            "sidetable",
            "stand",
            "table",
            "tray",
        }
        container_supports = {
            "basket",
            "bin",
            "bowl",
            "box",
            "crate",
            "plate",
            "tray",
        }
        non_support_surfaces = {
            "ceiling",
            "cube",
            "floor",
            "mesh",
            "room",
            "wall",
            "window",
        }
        asset_paths = sorted(assets)
        family_members: dict[str, set[str]] = {}
        for path in asset_paths:
            family_members.setdefault(family(path), set()).add(path)

        def normalized_asset_terms(path: str, asset: dict[str, object]) -> set[str]:
            category = str(asset["category"]).strip().lower().replace("-", "_").replace(" ", "_")
            name = path.rsplit("/", 1)[-1].lower().replace("-", "_").replace(" ", "_")
            return {
                value
                for source in (category, name)
                for value in [source, *source.split("_")]
                if value
            }

        def can_support_placed_assets(path: str, asset: dict[str, object]) -> bool:
            terms = normalized_asset_terms(path, asset)
            return not terms.intersection(non_support_surfaces) and bool(
                terms.intersection(support_surfaces)
                or any(term.endswith(("counter", "shelf", "stand", "table")) for term in terms)
            )

        def supports_asset(
            supporter_path: str,
            supporter: dict[str, object],
            candidate: dict[str, object],
        ) -> bool:
            supporter_min = np.asarray(supporter["min"], dtype=np.float64)
            supporter_max = np.asarray(supporter["max"], dtype=np.float64)
            candidate_min = np.asarray(candidate["min"], dtype=np.float64)
            candidate_max = np.asarray(candidate["max"], dtype=np.float64)
            candidate_size = candidate_max - candidate_min
            if np.any(candidate_size[:2] <= 0.0):
                return False
            overlap_xy = np.maximum(
                0.0,
                np.minimum(supporter_max[:2], candidate_max[:2])
                - np.maximum(supporter_min[:2], candidate_min[:2]),
            )
            overlap_fraction = float(np.prod(overlap_xy) / np.prod(candidate_size[:2]))
            if overlap_fraction < 0.15:
                return False
            bottom_gap = float(candidate_min[2] - supporter_max[2])
            terms = normalized_asset_terms(supporter_path, supporter)
            if terms.intersection(container_supports):
                return (
                    candidate_min[2] >= supporter_min[2] - 0.03
                    and candidate_min[2] <= supporter_max[2] + 0.12
                )
            return -0.03 <= bottom_gap <= 0.12

        def geometries_touch(first: dict[str, object], second: dict[str, object]) -> bool:
            first_geometry = list(first["geometry"])
            second_geometry = list(second["geometry"])
            if len(first_geometry) > len(second_geometry):
                first_geometry, second_geometry = second_geometry, first_geometry
            second_min = np.stack([np.asarray(value["min"]) for value in second_geometry])
            second_max = np.stack([np.asarray(value["max"]) for value in second_geometry])
            return any(
                overlapping_aabb_indices(
                    np.asarray(value["min"]),
                    np.asarray(value["max"]),
                    second_min,
                    second_max,
                    clearance=0.03,
                ).size
                for value in first_geometry
            )

        hidden_paths: set[str] = set()
        hidden_reasons: dict[str, str] = {}
        component_records: list[dict[str, object]] = []
        for seed_path in sorted(overlaps_by_path):
            seed = assets[seed_path]
            paths = set(family_members[family(seed_path)])
            if str(seed["category"]).strip().lower() in structural:
                for candidate_path, candidate in assets.items():
                    if str(candidate["category"]).strip().lower() not in structural:
                        continue
                    if geometries_touch(seed, candidate):
                        paths.update(family_members[family(candidate_path)])
            hidden_paths.update(paths)
            for path in paths:
                hidden_reasons[path] = (
                    "trajectory_overlap" if path in overlaps_by_path else "linked_asset_component"
                )
            component_records.append(
                {"seed_prim_paths": [seed_path], "hidden_prim_paths": sorted(paths)}
            )

        supported_by_hidden_asset: dict[str, list[str]] = {}
        support_dependencies: list[dict[str, str]] = []
        pending_supporters = sorted(hidden_paths)
        while pending_supporters:
            supporter_path = pending_supporters.pop(0)
            supporter = assets[supporter_path]
            if not can_support_placed_assets(supporter_path, supporter):
                continue
            for candidate_path in asset_paths:
                if candidate_path in hidden_paths or candidate_path == supporter_path:
                    continue
                candidate = assets[candidate_path]
                if not supports_asset(supporter_path, supporter, candidate):
                    continue
                hidden_paths.add(candidate_path)
                hidden_reasons[candidate_path] = "supported_by_hidden_asset"
                supported_by_hidden_asset.setdefault(candidate_path, []).append(supporter_path)
                support_dependencies.append(
                    {
                        "supporter_prim_path": supporter_path,
                        "supported_prim_path": candidate_path,
                    }
                )
                pending_supporters.append(candidate_path)

        hidden: list[dict[str, object]] = []
        for source_path in sorted(hidden_paths):
            asset = assets[source_path]
            prim_path = str(asset["prim_path"])
            prim = self.world.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            UsdGeom.Imageable(prim).MakeInvisible()
            self._indexed_collision_hidden[scene.cache_key].add(prim_path)
            hidden.append(
                {
                    "prim_path": source_path,
                    "category": asset["category"],
                    "world_aabb_min": np.asarray(asset["min"]).tolist(),
                    "world_aabb_max": np.asarray(asset["max"]).tolist(),
                    "reason": hidden_reasons[source_path],
                    "overlaps": overlaps_by_path.get(source_path, []),
                    "supported_by_hidden_asset": supported_by_hidden_asset.get(source_path, []),
                }
            )
        return {
            "mode": "all_index_assets_mesh_surface_overlap",
            "restore_hidden_assets": bool(restore_hidden_assets),
            "clearance_m": float(clearance_m),
            "candidate_count": len(assets),
            "geometry_candidate_count": sum(len(value["geometry"]) for value in assets.values()),
            "hidden_prim_paths": [entry["prim_path"] for entry in hidden],
            "hidden_items": hidden,
            "direct_overlaps": [
                {
                    "prim_path": path,
                    "overlaps": overlaps,
                }
                for path, overlaps in sorted(overlaps_by_path.items())
            ],
            "linked_components": component_records,
            "support_dependencies": support_dependencies,
            "restored_index_hidden_prim_paths": scene.metadata.get(
                "restored_index_hidden_prim_paths", []
            ),
            "restored_runtime_hidden_prim_paths": restored_runtime_paths,
            "protected_hidden_prim_paths": sorted(protected_hidden_paths),
            "trajectory_sources": {
                name: {
                    "episode_count": int(source.get("episode_count", 1)),
                    "trajectory_frame_count": int(source["trajectory_frame_count"]),
                    "trajectory_aabb_count": int(source["trajectory_aabb_count"]),
                    "uncompressed_trajectory_aabb_count": int(
                        source.get(
                            "uncompressed_trajectory_aabb_count",
                            source["trajectory_aabb_count"],
                        )
                    ),
                    "quantization_cell_m": source.get("quantization_cell_m"),
                    "exact_state": bool(source["exact_state"]),
                }
                for name, source in trajectory_sources
            },
            "skipped_items": skipped,
        }

    @staticmethod
    def _set_background_transform(scene_prim, orientation: np.ndarray, scale: float) -> None:
        xform = UsdGeom.Xformable(scene_prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        orient_op = xform.AddOrientOp()
        scale_op = xform.AddScaleOp()
        quaternion = t3d.quaternions.mat2quat(orientation)
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        orient_op.Set(Gf.Quatf(*[float(value) for value in quaternion]))
        scale_op.Set(Gf.Vec3d(scale, scale, scale))

    @staticmethod
    def _disable_background_physics(scene_prim) -> None:
        for prim in Usd.PrimRange(scene_prim):
            collision = UsdPhysics.CollisionAPI(prim)
            if collision:
                collision.CreateCollisionEnabledAttr(False)
            rigid_body = UsdPhysics.RigidBodyAPI(prim)
            if rigid_body:
                rigid_body.CreateRigidBodyEnabledAttr(False)
            if prim.GetTypeName() == "PhysicsScene":
                prim.SetActive(False)

    @staticmethod
    def _surface_top_center(surface_prim) -> np.ndarray:
        from omni.isaac.core.utils.bounds import compute_obb, create_bbox_cache

        centroid, axes, half_extent = compute_obb(create_bbox_cache(), surface_prim.GetPrimPath())
        axes = np.asarray(axes, dtype=np.float64)
        axis_norms = np.linalg.norm(axes, axis=1)
        unit_axes = axes / axis_norms[:, None]
        vertical_axis = int(np.argmax(np.abs(unit_axes @ np.array([0.0, 0.0, 1.0]))))
        direction = np.sign(unit_axes[vertical_axis, 2]) or 1.0
        return np.asarray(centroid, dtype=np.float64) + (
            direction * axes[vertical_axis] * float(half_extent[vertical_axis])
        )

    @staticmethod
    def _top_level_furniture_ancestor(surface_prim, furniture_prim):
        current = surface_prim
        while current.IsValid() and current.GetParent() != furniture_prim:
            current = current.GetParent()
        if not current.IsValid() or current.GetParent() != furniture_prim:
            raise ValueError(
                f"HSSD surface {surface_prim.GetPath()} is not under {furniture_prim.GetPath()}"
            )
        return current

    @staticmethod
    def _world_aligned_aabb(bbox_cache: UsdGeom.BBoxCache, prim) -> tuple[np.ndarray, np.ndarray]:
        aligned_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if aligned_range.IsEmpty():
            raise ValueError(f"HSSD furniture prim has an empty bound: {prim.GetPath()}")
        return (
            np.asarray(aligned_range.GetMin(), dtype=np.float64),
            np.asarray(aligned_range.GetMax(), dtype=np.float64),
        )

    def _hide_overlapping_hssd_furniture(
        self,
        *,
        scene_uid: str,
        scene_prim_path: str,
        surface_prim,
        trajectory_aabbs: dict[str, object],
        clearance_m: float,
    ) -> dict[str, object]:
        from omni.isaac.core.utils.bounds import create_bbox_cache

        stage = self.world.stage
        for prim_path in self._hssd_collision_hidden.get(scene_uid, set()):
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeVisible()
        self._hssd_collision_hidden[scene_uid] = set()

        furniture_prim = stage.GetPrimAtPath(f"{scene_prim_path}/furniture")
        if not furniture_prim.IsValid():
            raise ValueError(f"HSSD furniture prim does not exist: {furniture_prim.GetPath()}")
        protected_prim = self._top_level_furniture_ancestor(surface_prim, furniture_prim)

        bounds_min = np.asarray(trajectory_aabbs["bounds_min"], dtype=np.float64)
        bounds_max = np.asarray(trajectory_aabbs["bounds_max"], dtype=np.float64)
        frame_indices = np.asarray(trajectory_aabbs["frame_indices"], dtype=np.int32)
        geom_ids = np.asarray(trajectory_aabbs["geom_ids"], dtype=np.int32)
        geom_names = trajectory_aabbs["geom_names"]
        assert isinstance(geom_names, dict)

        hidden: list[dict[str, object]] = []
        skipped: list[dict[str, str]] = []
        candidates = list(furniture_prim.GetChildren())
        bbox_cache = create_bbox_cache()
        for candidate in candidates:
            if candidate == protected_prim:
                continue
            imageable = UsdGeom.Imageable(candidate)
            if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                skipped.append(
                    {"prim_path": str(candidate.GetPath()), "reason": "already_invisible"}
                )
                continue
            try:
                candidate_min, candidate_max = self._world_aligned_aabb(bbox_cache, candidate)
            except ValueError:
                skipped.append({"prim_path": str(candidate.GetPath()), "reason": "empty_bound"})
                continue
            overlap_indices = overlapping_aabb_indices(
                candidate_min,
                candidate_max,
                bounds_min,
                bounds_max,
                clearance=clearance_m,
            )
            if overlap_indices.size == 0:
                continue

            first_index = int(overlap_indices[0])
            geom_id = int(geom_ids[first_index])
            prim_path = str(candidate.GetPath())
            imageable.MakeInvisible()
            self._hssd_collision_hidden[scene_uid].add(prim_path)
            hidden.append(
                {
                    "prim_path": prim_path,
                    "world_aabb_min": candidate_min.tolist(),
                    "world_aabb_max": candidate_max.tolist(),
                    "first_overlap_frame": int(frame_indices[first_index]),
                    "first_overlap_geom_id": geom_id,
                    "first_overlap_geom_name": str(geom_names.get(geom_id, f"geom_{geom_id}")),
                }
            )

        return {
            "mode": "whole_episode_if_any_frame_overlaps",
            "clearance_m": float(clearance_m),
            "exact_state": bool(trajectory_aabbs["exact_state"]),
            "trajectory_frame_count": int(trajectory_aabbs["trajectory_frame_count"]),
            "moving_geom_count": int(trajectory_aabbs["moving_geom_count"]),
            "trajectory_aabb_count": int(trajectory_aabbs["trajectory_aabb_count"]),
            "candidate_furniture_count": len(candidates),
            "protected_furniture_prim": str(protected_prim.GetPath()),
            "hidden_furniture_prim_paths": [entry["prim_path"] for entry in hidden],
            "hidden_furniture": hidden,
            "skipped_furniture": skipped,
        }

    def replace_hssd_background(
        self,
        scene_uid: str,
        *,
        trajectory_aabbs: dict[str, object] | None = None,
        clearance_m: float = 0.05,
    ) -> dict[str, object]:
        """Replace the visual room while keeping the current MuJoCo task layout."""
        if clearance_m < 0:
            raise ValueError("HSSD collision clearance must be non-negative")
        table = self.task.layout.actors.get("table")
        if table is None:
            raise ValueError("HSSD background replacement requires layout actor 'table'")
        if self.task.layout.actors.get("table2") is not None:
            raise ValueError("HSSD background replacement currently supports one table")

        scene = HssdSceneManager().load(scene_uid)
        scene_prim_path = f"{self.SCENE_PRIM_PATH}/render_{scene.uid.replace(':', '_')}"
        stage = self.world.stage
        for child in stage.GetPrimAtPath(self.SCENE_PRIM_PATH).GetChildren():
            UsdGeom.Imageable(child).MakeInvisible()

        scene_prim = stage.GetPrimAtPath(scene_prim_path)
        if not scene_prim.IsValid():
            data_dir = resolve_data_path(scene.data_dir, auto_download=True)
            usd_path = os.path.abspath(f"{data_dir}/{scene.name}.usd")
            isaacsim_stage.add_reference_to_stage(usd_path=usd_path, prim_path=scene_prim_path)
            scene_prim = stage.GetPrimAtPath(scene_prim_path)

        target_top, table_yaw = table_top_center_and_yaw(
            table.pose.position, table.pose.quaternion, table.size
        )
        base_rotation = Rotation.from_euler(
            "xyz", scene.center_orientation, degrees=True
        ).as_matrix()
        yaw_rotation = Rotation.from_euler("z", table_yaw).as_matrix()
        self._set_background_transform(
            scene_prim, yaw_rotation @ base_rotation, scene.conf["scale"]
        )

        surface_path = scene.conf["surface"]["prim_path"].replace("/World", scene_prim_path)
        surface_prim = stage.GetPrimAtPath(surface_path)
        if not surface_prim.IsValid():
            raise ValueError(f"HSSD surface prim does not exist: {surface_path}")
        source_top = self._surface_top_center(surface_prim)
        translation = target_top - source_top
        scene_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*translation.tolist()))
        UsdGeom.Imageable(scene_prim).MakeVisible()
        UsdGeom.Imageable(surface_prim).MakeVisible()
        self._disable_background_physics(scene_prim)

        table_prim = stage.GetPrimAtPath(f"{self.workspace_prim_path}/table_cuboid")
        if table_prim.IsValid():
            UsdGeom.Imageable(table_prim).MakeInvisible()
        aligned_top = self._surface_top_center(surface_prim)
        result: dict[str, object] = {
            "scene_uid": scene.uid,
            "mujoco_table_top": target_top.tolist(),
            "hssd_table_top": aligned_top.tolist(),
            "alignment_error_m": float(np.linalg.norm(aligned_top - target_top)),
        }
        if trajectory_aabbs is not None:
            result["collision_filter"] = self._hide_overlapping_hssd_furniture(
                scene_uid=scene.uid,
                scene_prim_path=scene_prim_path,
                surface_prim=surface_prim,
                trajectory_aabbs=trajectory_aabbs,
                clearance_m=clearance_m,
            )
        return result

    def __update_cameras(self):
        for cname, cameraEntity in self.task.layout.cameras.items():
            p, q = cameraEntity.pose.position, cameraEntity.pose.quaternion
            isaacsim_camera = self.cameras[cname]
            # isaacsim_camera.set_local_pose(p, [q[-1], q[0], q[1], q[2]]) # wxyz
            isaacsim_camera.set_local_pose(p, q)  # xyzw
            isaacsim_camera.set_clipping_range(0.01, 10.0)  # FIXME hardcoded
            isaacsim_camera.set_focal_length(cameraEntity.focal_length)

            fx, fy = cameraEntity.fx, cameraEntity.fy
            horizontal_aperture = (
                cameraEntity.focal_length * cameraEntity.resolution[0] / fx
            )
            vertical_aperture = (
                cameraEntity.focal_length * cameraEntity.resolution[1] / fy
            )
            isaacsim_camera.set_horizontal_aperture(horizontal_aperture)
            isaacsim_camera.set_vertical_aperture(vertical_aperture)

    def __update_robot(self):
        robot_prim_path = self.robot.prim_path
        for shader_path in isaacsim_prims.find_matching_prim_paths(
            f"{robot_prim_path}/Looks/material_*/Shader"
        ):  # FIXME
            shader = UsdShade.Shader(isaacsim_prims.get_prim_at_path(shader_path))
            for key in ["reflection_roughness_constant", "metallic_constant", "specular_level"]:
                shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(
                    self.task.layout.robot.shaders[key]
                )  # type: ignore

    def __update_lights(self):
        assert len(self.lights) == len(self.task.layout.lights)

        for i, light_prim in enumerate(self.lights):  # self.lights:
            # light_prim = light[0]
            light_info = self.task.layout.lights[i]

            if i == 0:
                self.light_center.set_local_pose(
                    light_info.center_light_postion, light_info.center_light_orientation
                )

            light_prim.GetAttribute("xformOp:translate").Set(tuple(light_info.pose.position))
            light_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
            light_prim.GetAttribute("inputs:radius").Set(light_info.light_radius)
            light_prim.GetAttribute("inputs:length").Set(light_info.light_length)
            light_prim.GetAttribute("inputs:intensity").Set(light_info.light_intensity)
            light_prim.GetAttribute("inputs:color").Set(tuple(light_info.light_color))
            # Match USD's default: size OOD compensates intensity analytically.
            light_prim.GetAttribute("inputs:normalize").Set(False)
            light_prim.GetAttribute("inputs:enableColorTemperature").Set(
                light_info.light_enable_color_temperature
            )
            light_prim.GetAttribute("inputs:colorTemperature").Set(
                light_info.light_color_temperature
            )

    def __reset_objects(self):
        # from omni.isaac.core.utils.prims import delete_prim
        for obj_name in self.objects:
            obj = self.objects[obj_name]
            obj_xform = XFormPrim(prim_path=obj["object_prim_path"])
            obj_xform.set_visibility(False)  # obj["geom"]
            obj_xform.set_local_pose([0.0, 0.0, -1.0])  # obj["xform"]
            obj["bActive"] = False

    def __pre_add_objects(self):
        for obj in self.task.preload_objects():
            if obj.simulation_name not in self.objects:
                self.__create_object(obj.simulation_name, obj)

    def __create_object(self, obj_key: str, object_info: ObjectActor):
        obj_id = object_info.uid  # object_info["id"]
        simulation_name = object_info.simulation_name
        object_prim_path = f"{self.workspace_prim_path}/{simulation_name}"

        # if obj_id == 100:
        #     cube = self.__create_cuboid(object_prim_path,
        #                                 object_info["position"],
        #                                 object_info["dims"],
        #                                 object_info["orientation"],
        #                                 color=np.array([1., 0., 0.]))
        #     obj_xform = XFormPrim(prim_path=object_prim_path)
        #     obj_geom = cube
        #     obj_rigid = RigidPrim(prim_path=object_prim_path)
        # else:

        object_usd_path = os.path.abspath(
            resolve_data_path(object_info.asset.usd_path, auto_download=True)
        )

        isaacsim_stage.add_reference_to_stage(usd_path=object_usd_path, prim_path=object_prim_path)

        # XFormPrim normalizes named USD scale ops when it prepares the pose for replay.
        prim = get_prim_at_path(object_prim_path)
        authored_transform = UsdGeom.Xformable(prim).GetLocalTransformation()
        authored_scale = np.asarray(Gf.Transform(authored_transform).GetScale())
        obj_xform = XFormPrim(prim_path=object_prim_path)
        obj_xform.set_local_scale(authored_scale * getattr(object_info.asset, "scale", 1.0))
        geom_prim_path = f"{object_prim_path}/Meshes"
        if get_prim_at_path(geom_prim_path).IsValid():
            obj_rigid = RigidPrim(prim_path=geom_prim_path)
            obj_rigid.disable_rigid_body_physics()
            collision_prim_path = f"{geom_prim_path}/collision"
            if get_prim_at_path(collision_prim_path).IsValid():
                obj_collision_geom = GeometryPrim(collision_prim_path)
                obj_collision_geom.set_collision_enabled(False)
        else:
            carb.log_warn(
                f"Skipping physics setup for {object_prim_path}: "
                "referenced asset has no /Meshes prim"
            )

        usd_prim = isaacsim_prims.get_prim_at_path(object_prim_path)
        for prim in Usd.PrimRange(usd_prim):
            if prim.GetTypeName() != "Shader":
                continue
            shader = UsdShade.Shader(prim)
            if shader.GetIdAttr().Get() != "UsdUVTexture":
                continue
            texture_input = shader.GetInput("file")
            texture = texture_input.Get() if texture_input else None
            if not isinstance(texture, Sdf.AssetPath) or texture.resolvedPath:
                continue
            relative_path = texture.path.replace("\\", "/").partition("/output_usd/")[2]
            if not relative_path:
                continue
            local_path = os.path.join(os.path.dirname(object_usd_path), relative_path)
            if os.path.isfile(local_path):
                texture_input.Set(Sdf.AssetPath(local_path))

        semantics = [("prim", f"{obj_id}")]
        for semantic_type, semantic_value in semantics:
            instance_name = Tf.MakeValidIdentifier(f"{semantic_type}_{semantic_value}")
            sem = Semantics.SemanticsAPI.Apply(usd_prim, instance_name)
            sem.CreateSemanticTypeAttr()
            sem.CreateSemanticDataAttr()
            sem.GetSemanticTypeAttr().Set(semantic_type)
            sem.GetSemanticDataAttr().Set(semantic_value)

        obj = {
            "id": obj_id,
            "object_prim_path": object_prim_path,
            "bActive": False,
            "bTarget": obj_key == "target",  # object_info["bTarget"],
        }

        self.objects[simulation_name] = obj

    def __update_object(self, obj_name, obj_info: ObjectActor):
        obj = self.objects[obj_info.simulation_name]
        # obj_geom = GeometryPrim(prim_path=obj["object_prim_path"])
        # obj_geom.set_visibility(True)

        # # enable physics for now
        # geom_prim_path = f'{obj["object_prim_path"]}/Meshes'
        # obj_rigid = RigidPrim(prim_path=geom_prim_path)
        # # obj_rigid.set_kinematic_enabled(True)
        # obj_rigid.set_linear_velocity((0, 0, 0))
        # obj_rigid.set_angular_velocity((0, 0, 0))

        # obj_rigid.enable_rigid_body_physics()
        # obj_collision_geom = GeometryPrim(f"{geom_prim_path}/collision")
        # obj_collision_geom.set_collision_enabled(True)

        # num_subframes = 1
        # self.world.step(render=False)
        # for _ in range(2):# BUG: extra render to avoid empty frame
        #     rep.orchestrator.step(rt_subframes=num_subframes, pause_timeline=False)

        # change object pose
        obj_xform = XFormPrim(prim_path=obj["object_prim_path"])
        obj_xform.set_local_pose(obj_info.pose.position, obj_info.pose.quaternion)
        obj_xform.set_visibility(True)
        # geom_prim_path = f'{obj["object_prim_path"]}/Meshes'
        # obj_geom = GeometryPrim(prim_path=geom_prim_path)
        # obj_geom.set_local_pose([0,0,0], [1,0,0,0]) # reset mesh local pose
        obj["bActive"] = True

        # # disable physics for now
        # obj_rigid = RigidPrim(prim_path=geom_prim_path)
        # obj_rigid.disable_rigid_body_physics()
        # obj_collision_geom = GeometryPrim(f"{geom_prim_path}/collision")
        # obj_collision_geom.set_collision_enabled(False)

        # object_prim_path = obj["xform"].prim_path
        # object_shader_param = object_shader_params.pop()
        for shader_path in isaacsim_prims.find_matching_prim_paths(
            f"{obj['object_prim_path']}/Looks/material_*/material_*"
        ):
            shader = UsdShade.Shader(isaacsim_prims.get_prim_at_path(shader_path))
            # shader.SetSourceAsset('OmniPBR.mdl')
            for key in ["reflection_roughness_constant", "metallic_constant", "specular_level"]:
                shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(obj_info.material[key])  # type:ignore object_shader_param[key]

    def step(self, mujoco_env=None, *, advance_physics=True):
        if not self.is_isaac_reset:
            self.world.reset()
            # self.__update_object()
            self.robot.initialize()

            # update articulated objects if articulated objects is not empty
            if self.articulated_objects:
                for obj_name in self.articulated_objects.keys():
                    self.articulated_objects[obj_name].initialize()
                    self.articulated_objects[obj_name].disable_gravity()

            # TODO
            if not self.need_gravity:
                self.robot.disable_gravity()
            else:
                self.robot.disable_gravity()

            # set robot to initial pose
            robo_actor = self.task.layout.actors["robot"]
            robo_position = robo_actor.pose.position.copy()
            # robo_position[2] += 0.055 # HACK fix for veag 1 base height
            self.robot.set_world_pose(robo_position, robo_actor.pose.quaternion)
            # self.world.physics_sim_view.flush()

            # set articulated objects to initial pose if articulated objects is not empty
            if self.articulated_objects:
                for obj_name in self.articulated_objects.keys():
                    articulated = self.task.layout.actors[obj_name]
                    self.articulated_object_roots[obj_name].set_world_pose(
                        articulated.pose.position,
                        articulated.pose.quaternion,
                    )
                    position, orientation = self._articulated_root_pose(
                        obj_name,
                        articulated.pose.position,
                        articulated.pose.quaternion,
                    )
                    self.articulated_objects[obj_name].set_world_pose(
                        position, orientation
                    )
                    # self.world.physics_sim_view.flush()

            if self.articulated_visual_objects:
                for obj_name, visual_object in self.articulated_visual_objects.items():
                    articulated = self.task.layout.actors[obj_name]
                    visual_object.set_world_pose(
                        articulated.pose.position,
                        articulated.pose.quaternion,
                    )

            # Disable motors
            num_dofs = self.robot._articulation_view.num_dof
            zeros = np.zeros(num_dofs, dtype=np.float32)
            ones = np.ones(num_dofs, dtype=np.float32)  # fix joint oscillation
            self.robot._articulation_view.set_gains(kps=ones * 800, kds=ones * 20)

            # disable motors if articulated objects is not empty
            if self.articulated_objects:
                for obj_name in self.articulated_objects.keys():
                    num_dofs_articulated = self.articulated_objects[
                        obj_name
                    ]._articulation_view.num_dof
                    zero = np.zeros(num_dofs_articulated, dtype=np.float32)
                    one = np.ones(num_dofs_articulated, dtype=np.float32)  # fix joint oscillation
                    self.articulated_objects[obj_name]._articulation_view.set_gains(
                        kps=zero, kds=zero
                    )

            joint_state = (
                self.task.robot.init_joint_states
            )  # self.layout.robot_info["init_robot_qpos"]
            joint_indices = []
            qpos = []
            # print("syncing isaac to mujoco qpos:", joint_state)
            for jname, jpos in joint_state.items():
                isaac_jname = self.task.robot.jname_mujoco_to_isaac(jname)
                joint_indices.append(self.robot.get_dof_index(isaac_jname))
                qpos.append(jpos)
            self.robot.set_joint_positions(qpos, joint_indices=joint_indices)
            # self.world.physics_sim_view.flush()

            # set articulated joint state if articulated objects is not empty
            if self.articulated_objects:
                for obj_name in self.articulated_objects.keys():
                    articulate_joint_qpos = self.task.layout.actors[
                        obj_name
                    ].asset.articulate_init_joint_qpos
                    qpos = resolve_articulated_joint_positions(
                        articulate_joint_qpos,
                        num_dofs=self.articulated_objects[
                            obj_name
                        ]._articulation_view.num_dof,
                    )
                    self.articulated_objects[obj_name].set_joint_positions(qpos)

            # Stop all motion
            self.robot._articulation_view.set_joint_velocities(zeros)
            self.is_isaac_reset = True
            self._update_collision_spheres()

            # stop articulated objects joint motion if articulated objects is not empty
            if self.articulated_objects:
                for obj_name in self.articulated_objects.keys():
                    self._stop_articulated_object_motion(
                        self.articulated_objects[obj_name]
                    )

        if mujoco_env is not None:
            self.sync_states(mujoco_env)

        self.world.step(render=False)
        if not advance_physics and mujoco_env is not None:
            # Isaac 4.5 has no functional flush-only path for tensor state writes.
            # Publish the transforms, then discard the solver result by restoring
            # the authoritative MuJoCo state before the zero-time capture below.
            self.sync_states(mujoco_env)
        hand_spec = getattr(self.task.robot, "hand_spec", None)
        if not advance_physics and hand_spec is not None and hand_spec.name == "brainco":
            self.world.physics_sim_view.update_articulations_kinematic()
            omni.physx.get_physx_interface().update_transformations(False, True)
        self.update_visuals()
        rep.orchestrator.step(
            rt_subframes=self.render_quality.rt_subframes,
            pause_timeline=False,
            delta_time=None if advance_physics else 0.0,
        )
        self._update_collision_spheres()
        self.step_id += 1

    def _update_collision_spheres(self):
        """Update collision spheres visualization for the robot."""
        if not self.visulize_spheres:
            return

        from curobo.types.base import TensorDeviceType

        tensor_args = TensorDeviceType()

        sim_js = self.robot.get_joints_state()
        sim_js_names = self.robot.dof_names
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities),
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(
            [self.task.robot.jname_mujoco_to_isaac(j) for j in self.task.robot.joint_names]
        )
        sph_list = self.task.robot.kin_model.get_robot_as_spheres(cu_js.position)

        T_world_robot = np.eye(4)
        T_world_robot[:3, 3] = self.robot.get_world_pose()[0]
        T_world_robot[:3, :3] = t3d.quaternions.quat2mat(self.robot.get_world_pose()[1])

        if self.spheres is None:
            self.spheres = []
            for si, s in enumerate(sph_list[0]):
                T_robot_sphere = np.eye(4)
                T_robot_sphere[:3, 3] = s.pose[:3]
                T_robot_sphere[:3, :3] = t3d.quaternions.quat2mat(s.pose[3:])

                T_world_sphere = T_world_robot @ T_robot_sphere
                s.position = T_world_sphere[:3, 3]
                sp = sphere.VisualSphere(
                    prim_path="/curobo/robot_sphere_" + str(si),
                    position=np.ravel(s.position),
                    radius=float(s.radius),
                    color=np.array([0, 0.8, 0.2]),
                )
                self.spheres.append(sp)
        else:
            for si, s in enumerate(sph_list[0]):
                T_robot_sphere = np.eye(4)
                T_robot_sphere[:3, 3] = s.pose[:3]
                T_robot_sphere[:3, :3] = t3d.quaternions.quat2mat(s.pose[3:])

                T_world_sphere = T_world_robot @ T_robot_sphere
                s.position = T_world_sphere[:3, 3]
                self.spheres[si].set_world_pose(position=np.ravel(s.position))
                self.spheres[si].set_radius(float(s.radius))

    def sync_states(self, mujoco_env):
        states = mujoco_env.get_states()
        step_id, joint_state, obj_names, obj_positions, obj_orientations, robot_position = (
            states[:6]
        )

        for obj_name, obj_pos, obj_ori in zip(obj_names, obj_positions, obj_orientations):
            obj = self.objects[obj_name]
            obj_xfrom = XFormPrim(prim_path=obj["object_prim_path"])
            obj_xfrom.set_local_pose(translation=obj_pos, orientation=obj_ori)  # obj["xform"]

        joint_indices = []
        qpos = []
        # print("syncing isaac to mujoco qpos:", joint_state)
        for jname, jpos in joint_state.items():
            isaac_jname = self.task.robot.jname_mujoco_to_isaac(jname)
            joint_indices.append(self.robot.get_dof_index(isaac_jname))
            qpos.append(round(jpos, 6))
        self.robot.set_joint_positions(qpos, joint_indices=joint_indices)

        if self.articulated_objects:
            for obj_name in self.articulated_objects.keys():
                articulated_object = self.articulated_objects[obj_name]
                articulate_joint_indices = []
                articulate_joint_pos = []
                prefix, joint_names, _ = mujoco_env.articulated_actor_info[obj_name]
                for articulate_joint in joint_names:
                    articulate_jpos = float(mujoco_env.mjData.joint(articulate_joint).qpos[0])
                    isaac_joint_name = articulate_joint.removeprefix(prefix)
                    try:
                        joint_index = articulated_object.get_dof_index(isaac_joint_name)
                    except KeyError:
                        body_id = int(mujoco_env.mjModel.joint(articulate_joint).bodyid[0])
                        link_name = mujoco_env.mjModel.body(body_id).name.removeprefix(prefix)
                        isaac_joint = self.articulated_joint_names_by_link[obj_name].get(
                            link_name
                        )
                        if isaac_joint is None:
                            raise KeyError(
                                f"No Isaac joint found for MuJoCo joint {articulate_joint!r} "
                                f"on link {link_name!r}"
                            ) from None
                        joint_index = articulated_object.get_dof_index(isaac_joint)
                    articulate_joint_indices.append(joint_index)
                    articulate_joint_pos.append(round(articulate_jpos, 6))
                if articulate_joint_indices:
                    articulated_object.set_joint_positions(
                        articulate_joint_pos, joint_indices=articulate_joint_indices
                    )

        if self.need_gravity:
            self.robot.set_world_pose(position=robot_position[:3], orientation=robot_position[3:])
            if self.articulated_objects:
                for obj_name, articulated_object in self.articulated_objects.items():
                    root_body_name = mujoco_env.articulated_actor_info[obj_name][2]
                    body = mujoco_env.mjData.body(root_body_name)
                    self.articulated_object_roots[obj_name].set_world_pose(
                        position=body.xpos,
                        orientation=body.xquat,
                    )
                    position, orientation = self._articulated_root_pose(
                        obj_name,
                        body.xpos,
                        body.xquat,
                    )
                    articulated_object.set_world_pose(
                        position=position,
                        orientation=orientation,
                    )
            for obj_name, visual_object in self.articulated_visual_objects.items():
                root_body_name = mujoco_env.articulated_actor_info[obj_name][2]
                body = mujoco_env.mjData.body(root_body_name)
                visual_object.set_world_pose(position=body.xpos, orientation=body.xquat)
        self._sync_articulated_visual_bodies(mujoco_env)
        # self.world.physics_sim_view.flush()
        # Stop all motion
        num_dofs = self.robot._articulation_view.num_dof
        zeros = np.zeros(num_dofs, dtype=np.float32)
        self.robot._articulation_view.set_joint_velocities(zeros)

        if self.articulated_objects:
            for articulated_object in self.articulated_objects.values():
                self._stop_articulated_object_motion(articulated_object)
        return step_id

    def _add_articulated_visual_bodies(self, obj_name, obj_info, object_prim_path, asset_usd_path):
        mapping_path = getattr(obj_info.asset, "usd_body_mapping", None)
        if mapping_path is None:
            return
        with open(self.resolve_data_path(mapping_path)) as stream:
            mapping = json.load(stream)
        source_stage = Usd.Stage.Open(asset_usd_path)
        source_root = source_stage.GetDefaultPrim().GetPath()
        stage = omni.usd.get_context().get_stage()
        bodies = {}
        # Parents must be positioned before children because the mapped USD
        # links retain their hierarchy while receiving absolute MuJoCo poses.
        for body_name, source_path in sorted(mapping.items(), key=lambda item: item[1].count("/")):
            source_path = Sdf.Path(source_path)
            if not source_path.HasPrefix(source_root):
                raise ValueError(f"Body mapping path {source_path} is outside {source_root}")
            prim_path = source_path.ReplacePrefix(source_root, Sdf.Path(object_prim_path))
            if not stage.GetPrimAtPath(prim_path):
                raise ValueError(f"Body mapping prim does not exist: {prim_path}")
            bodies[body_name] = XFormPrim(prim_path=str(prim_path))
        self.articulated_visual_bodies[obj_name] = bodies

    def _sync_articulated_visual_bodies(self, mujoco_env):
        for obj_name, visual_bodies in self.articulated_visual_bodies.items():
            prefix = mujoco_env.articulated_actor_info[obj_name][0]
            for body_name, visual_body in visual_bodies.items():
                body = mujoco_env.mjData.body(prefix + body_name)
                visual_body.set_world_pose(position=body.xpos, orientation=body.xquat)

    @staticmethod
    def _stop_articulated_object_motion(articulated_object):
        num_dofs = articulated_object._articulation_view.num_dof
        articulated_object._articulation_view.set_joint_velocities(
            np.zeros(num_dofs, dtype=np.float32)
        )
        articulated_object.set_linear_velocity(np.zeros(3, dtype=np.float32))
        articulated_object.set_angular_velocity(np.zeros(3, dtype=np.float32))

    def _articulated_root_pose(self, obj_name, position, orientation):
        offset_position, offset_orientation = self.articulated_root_offsets[obj_name]
        return compose_pose_with_offset(
            position,
            orientation,
            offset_position,
            offset_orientation,
        )

    def get_states(self) -> dict[str, float]:
        raise NotImplementedError

    def set_states(self, states: dict[str, float]) -> None:
        raise NotImplementedError

    def add_articulated_object(self, obj_name: str, obj_info: ObjectActor):
        asset_name = obj_info.asset.name
        uid = obj_info.asset.uid
        object_prim_path = f"{self.workspace_prim_path}/articulated_objects/{asset_name}"
        asset_usd_path = self.resolve_data_path(obj_info.asset.usd_path)

        isaacsim_stage.add_reference_to_stage(
            usd_path=asset_usd_path,
            prim_path=object_prim_path,
        )

        stage = omni.usd.get_context().get_stage()
        object_prim = stage.GetPrimAtPath(object_prim_path)
        articulation_roots = [
            prim
            for prim in Usd.PrimRange(object_prim)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        if len(articulation_roots) > 1:
            root_paths = [str(prim.GetPath()) for prim in articulation_roots]
            raise RuntimeError(
                f"Expected at most one articulation root under {object_prim_path}, found {root_paths}"
            )

        if articulation_roots:
            articulation_root = articulation_roots[0]
            self.articulated_object_roots[obj_name] = XFormPrim(
                prim_path=object_prim_path,
                position=obj_info.pose.position,
                orientation=obj_info.pose.quaternion,
            )
            xform_cache = UsdGeom.XformCache()
            object_world = xform_cache.GetLocalToWorldTransform(object_prim)
            articulation_world = xform_cache.GetLocalToWorldTransform(articulation_root)
            object_to_articulation = articulation_world * object_world.GetInverse()
            offset_translation = object_to_articulation.ExtractTranslation()
            offset_quaternion = object_to_articulation.ExtractRotation().GetQuat()
            self.articulated_root_offsets[obj_name] = (
                np.asarray(offset_translation, dtype=np.float64),
                np.asarray(
                    [offset_quaternion.GetReal(), *offset_quaternion.GetImaginary()],
                    dtype=np.float64,
                ),
            )
        else:
            # Some legacy assets (for example articulated/000) contain visual
            # meshes only. MuJoCo remains authoritative for their dynamics; do
            # not turn the visual Xform into an Isaac articulation.
            self.articulated_visual_objects[obj_name] = XFormPrim(
                prim_path=object_prim_path,
                position=obj_info.pose.position,
                orientation=obj_info.pose.quaternion,
            )
            self._add_articulated_visual_bodies(
                obj_name, obj_info, object_prim_path, asset_usd_path
            )
            return

        self.articulated_joint_names_by_link[obj_name] = {
            prim.GetParent().GetName(): prim.GetName()
            for prim in Usd.PrimRange(object_prim)
            if prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.FixedJoint)
        }

        position, orientation = self._articulated_root_pose(
            obj_name,
            obj_info.pose.position,
            obj_info.pose.quaternion,
        )
        self.articulated_objects[obj_name] = SingleArticulation(
            prim_path=str(articulation_root.GetPath()),
            position=position,
            orientation=orientation,
        )
        if articulation_roots:
            self.articulated_objects[obj_name].set_enabled_self_collisions(False)

        from omni.isaac.core.prims import GeometryPrim

        articulation_prim = self.articulated_objects[obj_name].prim.GetParent()

        for prim in Usd.PrimRange(articulation_prim):
            prim_name = prim.GetName()

            if prim_name.lower() == "collisions" or prim_name.lower() == "collision":
                prim_path = str(prim.GetPath())

                # unique_name = f"{obj_name}_{prim_path.split('/')[-2]}_col"

                try:
                    obj_collision_geom = GeometryPrim(
                        prim_path=prim_path,
                        # name=unique_name,
                        # translation=None,
                        # orientation=None
                    )

                    obj_collision_geom.set_collision_enabled(False)

                except Exception as e:
                    print(f"skip: {prim_path}")

    def add_robot(self):
        robot_ns = self.task.robot.robot_ns
        eef_prim_path = self.task.robot.eef_prim_path
        hand_prim_path = self.task.robot.hand_prim_path

        isaacsim_stage.add_reference_to_stage(
            usd_path=self.resolve_data_path(self.task.robot.usd_path),
            prim_path=f"{self.workspace_prim_path}/Robot",
        )

        robo_actor = self.task.layout.actors["robot"]
        self.robot = IsaacRobot(
            prim_path=f"{self.workspace_prim_path}/Robot/{robot_ns}",
            position=robo_actor.pose.position,  # does not work
            orientation=robo_actor.pose.quaternion,  # does not work
        )
        # TODO assert self.task.robot.FRANKA_FINGER_LENGTH == self.layout.robot_info["robot_eef_offset"], "bug"

        # Revo2 paths point to physical palm links, not synthetic task frames.
        eef_pose = {
            "translation": [0, 0, self.task.robot.robot_eef_offset],
            "orientation": [1.0, 0.0, 0.0, 0.0],
        }
        hand_pose = {"translation": [0, 0, 0], "orientation": [1.0, 0.0, 0.0, 0.0]}
        hand_spec = getattr(self.task.robot, "hand_spec", None)
        if hand_spec is not None and hand_spec.name == "brainco":
            eef_pose, hand_pose = {}, {}
        robot_eef_xform = XFormPrim(
            f"{self.workspace_prim_path}/Robot/{robot_ns}/{eef_prim_path}", **eef_pose
        )
        robot_hand = XFormPrim(
            f"{self.workspace_prim_path}/Robot/{robot_ns}/{hand_prim_path}", **hand_pose
        )
        self.robot_eef_xform = robot_eef_xform
        self.robot_hand = robot_hand

    def add_cameras(self):
        for cam_key, cam_info in self.task.layout.cameras.items():
            if cam_info.mount == "eye_in_hand":
                # FIXME
                if cam_key == "wrist_left":
                    wrist_cam_link = self.task.robot.wrist_cam_link.replace("right", "left")
                else:
                    wrist_cam_link = self.task.robot.wrist_cam_link
                cam_prim_path = (
                    f"{self.workspace_prim_path}/Robot/{self.task.robot.robot_ns}/{wrist_cam_link}"
                )
            elif cam_info.mount == "eye_on_base":
                cam_prim_path = f"{self.workspace_prim_path}/Robot/{self.task.robot.robot_ns}"
            elif cam_info.mount == "eye_in_head":
                head_prim_link = self.task.robot.head_cam_link
                cam_prim_path = (
                    f"{self.workspace_prim_path}/Robot/{self.task.robot.robot_ns}/{head_prim_link}"
                )
            else:
                raise ValueError(f"Unsupported camera mount: {cam_info.mount}")

            camera = IssacCamera(
                name=cam_key,
                prim_path=f"{cam_prim_path}/{cam_key}",
                resolution=cam_info.resolution,
            )
            camera.initialize()
            self.cameras[cam_key] = camera

    def add_lights(self):
        light_center = XFormPrim(f"{self.workspace_prim_path}/LightCenter")

        for light in self.task.layout.lights:
            prim_path = f"{self.workspace_prim_path}/LightCenter/{light.uid}"
            light = isaacsim_prims.create_prim(
                prim_path,
                light.type,
            )
            self.lights.append(light)

        self.light_center = light_center

    def update_visuals(self):
        stage = omni.usd.get_context().get_stage()
        stage_unit = UsdGeom.GetStageMetersPerUnit(stage)
        # points = [(0, 0, 0.14 / stage_unit)]
        dbg = _debug_draw.acquire_debug_draw_interface()

        for _, visual in self.task.layout.visuals.items():
            if isinstance(visual, VisualFrame):
                axis_length = 0.1
                origin = np.array(
                    visual.pose.position, dtype=np.float32
                )  # np.array([0.02013437, 0.08273282, 0.05279581])
                rotation_mat = t3d.quaternions.quat2mat(visual.pose.quaternion)

                x_axis = origin + rotation_mat[:3, 0] * axis_length
                y_axis = origin + rotation_mat[:3, 1] * axis_length
                z_axis = origin + rotation_mat[:3, 2] * axis_length

                dbg.draw_lines(
                    [origin, origin, origin],  # start points
                    [x_axis, y_axis, z_axis],  # end points
                    [
                        carb.ColorRgba(1, 0, 0, 1),  # X = red
                        carb.ColorRgba(0, 1, 0, 1),  # Y = green
                        carb.ColorRgba(0, 0, 1, 1),
                    ],  # Z = blue
                    [2.0, 2.0, 2.0],  # line widths
                )

            elif isinstance(visual, VisualGrasp):
                lines = visual.plot_lines()
                dbg.draw_lines(
                    [l[0] for l in lines],  # start points
                    [l[1] for l in lines],  # end points
                    [carb.ColorRgba(l[2]) for l in lines],  # colors
                    [l[3] for l in lines],  # line widths
                )

    def render(self, *args, **kwargs) -> dict[str, np.ndarray]:
        render_products = {}
        for cam_name, camera in self.cameras.items():
            # frame = camera.get_current_frame()
            # raw_rgb = frame['rgba'][...,:3].astype(np.uint8)
            frame = camera.get_rgba()
            raw_rgb = frame[..., :3].astype(np.uint8)
            render_products[cam_name] = raw_rgb
        return render_products

    def calc_surface_center(self, surface_prim):
        from omni.isaac.core.utils.bounds import compute_combined_aabb, create_bbox_cache

        bb_cache = create_bbox_cache()
        centroid, axes, half_extent = compute_obb(bb_cache, surface_prim.GetPrimPath())
        larger_xy_extent = (half_extent[0], half_extent[1], half_extent[2])
        obb_corners = get_obb_corners(centroid, axes, larger_xy_extent)
        # TODO test
        top_corners = [
            # obb_corners[0].tolist(),
            # obb_corners[1].tolist(),
            obb_corners[2].tolist(),
            obb_corners[3].tolist(),
            # obb_corners[4].tolist(),
            # obb_corners[5].tolist(),
            obb_corners[6].tolist(),
            obb_corners[7].tolist(),
        ]

        position = np.mean(top_corners, axis=0)

        return {
            "axes": axes,
            "half_extent": half_extent,
            "position": position,
        }

    def _setup_table(self, prim_path: str, table_box):
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(prim_path).IsValid():
            stage.RemovePrim(prim_path)

        omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform", prim_type="Cube", prim_path=prim_path
        )
        prim = stage.GetPrimAtPath(prim_path)

        p, q, s = table_box.pose.position, table_box.pose.quaternion, table_box.size
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*[float(x) for x in p]))
        # q is wxyz [qw, qx, qy, qz]; Gf.Quatd(real, i, j, k) expects (w, x, y, z)
        prim.GetAttribute("xformOp:orient").Set(
            Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        )
        prim.GetAttribute("xformOp:scale").Set(Gf.Vec3d(*[float(x) for x in s]))

        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim)

        mat_info = getattr(table_box, "material", None)
        if mat_info is None:
            return
        if mat_info.get("type") == "usd_preview_surface":
            self._bind_table_preview_surface(stage, prim, mat_info)
            return
        raw_path = mat_info["path"]
        if not os.path.isabs(raw_path) or not os.path.exists(raw_path):
            raw_path = resolve_data_path(raw_path, auto_download=True)
        created = [None]
        create_mdl_material(stage, raw_path, mat_info["name"], lambda p: created.__setitem__(0, p))
        if created[0] is not None:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(created[0]))

    @staticmethod
    def _bind_table_preview_surface(stage, prim, mat_info) -> None:
        material_name = Tf.MakeValidIdentifier(str(mat_info.get("name", "HumanoReplayTable")))
        material_path = f"{prim.GetPath()}/Looks/{material_name}"
        shader_path = f"{material_path}/Shader"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*[float(v) for v in mat_info["diffuse_color"]])
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            float(mat_info.get("roughness", 0.55))
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            float(mat_info.get("metallic", 0.0))
        )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
