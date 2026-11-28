import math
from collections import deque
from copy import deepcopy
from queue import Queue

import numpy as np
import torch
import transforms3d as t3d

from humano.core.actor import ObjectActor
from humano.core.asset import Asset
from humano.core.layout import Layout
from humano.core.object import SpatialAnnotated
from humano.core.task import Task
from humano.datagen.subtask_spec import (
    CloseGripperSpec,
    GraspObjectSpec,
    HeightAdjustSpec,
    LiftSpec,
    LowerSpec,
    MoveEEFToPoseSpec,
    OpenGripperSpec,
    PhaseBreakSpec,
    RetreatSpec,
    StandSpec,
    TurnSpec,
    WalkSpec,
)
from humano.grasps.bodex import Bodex
from humano.grasps.gsnet import GSNet
from humano.mp.planner import MotionPlanner
from humano.robots.protocols import DualArm, Graspable

from .base_agent import BaseAgent
from .primitive_agent import PrimitiveAgent


def pad_to_full_qpos(qpos: dict, init_qpos: dict):
    qpos_full = deepcopy(qpos)
    for jname, val in init_qpos.items():
        if jname not in qpos:
            qpos_full[jname] = val
    return qpos_full


class MotionPlannerAgent(PrimitiveAgent):
    plan_batch_size = 1

    def __init__(
        self,
        task: Task,
        planner: MotionPlanner,
        debug: bool = False,
        plan_batch_size: int | None = None,
    ):
        super().__init__(task.robot)

        self.task = task
        self.planner = planner
        self.plan_batch_size = (
            plan_batch_size if plan_batch_size is not None else self.plan_batch_size
        )

        self.debug = debug
        self._subtask_index = 0
        self._last_traj_qpos = None
        self._planning_current_joint_qpos: dict | None = None

        self._stored_grasps = {}  # Store grasps by key for reference,only for aloha
        self.target_stable_index = 0
        self.last_target_yaw = 0
        self.last_grasp_plan = None

    def _trajectory_action_kwargs(self, **kwargs) -> dict:
        """Keep CuRobo interpolation timing when waypoints become ActionCmds."""
        plan_dt = getattr(self.planner, "plan_dt", None)
        if plan_dt is not None and float(plan_dt) > 0.0:
            kwargs["source_dt"] = float(plan_dt)
        return kwargs

    def _locomotion_action_kwargs(self, **kwargs) -> dict:
        """Preserve the control-tick duration used to expand locomotion Specs."""
        source_dt = getattr(self.robot, "sim_dt", None)
        if source_dt is None:
            source_dt = getattr(self.task, "metadata", {}).get("physics_dt", 0.02)
        try:
            source_dt = float(source_dt)
        except (TypeError, ValueError):
            source_dt = 0.02
        kwargs["source_dt"] = source_dt if source_dt > 0.0 else 0.02
        return kwargs

    def set_planning_current_joint_qpos(self, qpos: dict | None) -> None:
        """Override the joint state used as the next cuRobo planning seed."""
        self._planning_current_joint_qpos = None if qpos is None else deepcopy(qpos)

    def _current_joint_qpos_for_planning(self) -> dict:
        planning_qpos = getattr(self, "_planning_current_joint_qpos", None)
        if planning_qpos is not None:
            return deepcopy(planning_qpos)
        return self.robot.get_robot_qpos()

    def _queue_qpos_hold(self, target_qpos: dict, duration_s: float, **kwargs) -> None:
        """Repeat a completed joint target long enough for the controller to settle."""
        if duration_s <= 0.0:
            return
        try:
            source_dt = float(kwargs.get("source_dt", 0.02))
        except (TypeError, ValueError):
            source_dt = 0.02
        source_dt = source_dt if source_dt > 0.0 else 0.02
        steps = max(1, int(math.ceil(duration_s / source_dt)))
        for _ in range(steps):
            self.queue_move_qpos(target_qpos, **kwargs)

    @staticmethod
    def _trace_vector(value) -> list[float] | None:
        if value is None:
            return None
        return np.asarray(value, dtype=np.float64).reshape(-1).tolist()

    @staticmethod
    def _trace_pose(pose) -> dict[str, list[float]] | None:
        if pose is None:
            return None
        return {
            "position": np.asarray(pose.position, dtype=np.float64).reshape(-1).tolist(),
            "quaternion": np.asarray(pose.quaternion, dtype=np.float64).reshape(-1).tolist(),
        }

    @staticmethod
    def _link_pose_wrt_robot(link_pose, robot_pose):
        """Return a copy of one Bodex link pose in the robot base frame."""
        result = deepcopy(link_pose)
        position = result.position.cpu().numpy().squeeze(0)
        quaternion = result.quaternion.cpu().numpy().squeeze(0)
        T_world_linkpose = np.eye(4)
        T_world_linkpose[:3, 3] = position
        T_world_linkpose[:3, :3] = t3d.quaternions.quat2mat(quaternion)
        T_robot_linkpose = np.linalg.inv(robot_pose) @ T_world_linkpose
        result.position = (
            torch.from_numpy(T_robot_linkpose[:3, 3])
            .float()
            .to(result.position.device)
            .unsqueeze(0)
        )
        result.quaternion = (
            torch.from_numpy(t3d.quaternions.mat2quat(T_robot_linkpose[:3, :3]))
            .float()
            .to(result.quaternion.device)
            .unsqueeze(0)
        )
        return result

    def _trajectory_end_eef_poses(self, trajs, jnames, robot_pose) -> dict:
        """Trace the cuRobo trajectory endpoint in both planning frames."""
        try:
            endpoint_qpos = dict(zip(jnames, trajs[-1][-1]))
            endpoint_pose = np.asarray(
                self.robot.get_link_pose(self.robot.eef_prim_path, endpoint_qpos),
                dtype=np.float64,
            ).reshape(-1)
            if endpoint_pose.size != 7:
                raise ValueError(
                    f"expected pose [x, y, z, qw, qx, qy, qz], got {endpoint_pose.shape}"
                )

            T_robot_eef = np.eye(4)
            T_robot_eef[:3, 3] = endpoint_pose[:3]
            T_robot_eef[:3, :3] = t3d.quaternions.quat2mat(endpoint_pose[3:])
            T_world_eef = np.asarray(robot_pose, dtype=np.float64) @ T_robot_eef
            return {
                "trajectory_end_eef_position_robot": endpoint_pose[:3].tolist(),
                "trajectory_end_eef_orientation_robot": endpoint_pose[3:].tolist(),
                "trajectory_end_eef_position_world": T_world_eef[:3, 3].tolist(),
                "trajectory_end_eef_orientation_world": t3d.quaternions.mat2quat(
                    T_world_eef[:3, :3]
                ).tolist(),
            }
        except Exception as exc:
            # FK tracing is diagnostic only. A planner-valid trajectory must not
            # be discarded when an optional robot FK implementation is absent.
            return {"trajectory_end_eef_trace_error": str(exc)}

    def _plan_bodex_candidates(self, grasps, spec, robot_pose):
        """Try one Bodex candidate at a time until cuRobo returns a trajectory."""
        assert isinstance(self.robot, Graspable)
        attempts = []
        lock_links_names = spec.meta.get("lock_links", None)
        current_joint_qpos = self._current_joint_qpos_for_planning()

        for attempt_index, grasp in enumerate(grasps, start=1):
            candidate_idx = grasp.get("grasp_candidate_idx", grasp.get("grasp_id"))
            attempt = {
                "candidate_idx": candidate_idx,
                "attempt": attempt_index,
                "grasp_position_world": self._trace_vector(grasp.get("position")),
                "grasp_orientation_world": self._trace_vector(grasp.get("orientation")),
            }
            try:
                plan_grasp = deepcopy(grasp)
                goal_position, goal_orientation = self.robot.get_grasp_pose_wrt_robot(
                    plan_grasp,
                    pregrasp=spec.meta.get("pregrasp", False),
                    robot_pose=robot_pose,
                )
                attempt["goal_position_robot"] = np.asarray(goal_position).tolist()
                attempt["goal_orientation_robot"] = np.asarray(goal_orientation).tolist()
                T_robot_goal = np.eye(4)
                T_robot_goal[:3, 3] = goal_position
                T_robot_goal[:3, :3] = t3d.quaternions.quat2mat(goal_orientation)
                T_world_goal = robot_pose @ T_robot_goal
                attempt["goal_position_world"] = T_world_goal[:3, 3].tolist()
                attempt["goal_orientation_world"] = t3d.quaternions.mat2quat(
                    T_world_goal[:3, :3]
                ).tolist()
                plan_grasp["position"] = goal_position
                plan_grasp["orientation"] = goal_orientation
                link_poses = {
                    key: self._link_pose_wrt_robot(pose, robot_pose)
                    for key, pose in grasp["link_poses"].items()
                }
                trajs, jnames = self.planner.batch_plan_for_approach_bodex(
                    [np.concatenate([goal_position, goal_orientation], axis=0)],
                    [plan_grasp["grasp_qpos"]],
                    [plan_grasp["squeeze_qpos"]],
                    [plan_grasp["lift_qpos"]],
                    link_poses=link_poses,
                    debug_grasp=grasp,
                    world_layout=self.task.layout,
                    lock_links_names=lock_links_names,
                    current_joint_qpos=current_joint_qpos,
                )
            except Exception as exc:
                attempt.update(status="failed", error=str(exc))
                attempts.append(attempt)
                print(
                    f"[MP Agent] Bodex candidate {candidate_idx} failed "
                    f"({attempt_index}/{len(grasps)}): {exc}"
                )
                continue

            attempt.update(
                status="success",
                trajectory_waypoints=int(len(trajs[-1])),
            )
            attempt.update(self._trajectory_end_eef_poses(trajs, jnames, robot_pose))
            attempts.append(attempt)
            print(
                f"[MP Agent] Bodex candidate {candidate_idx} planned successfully "
                f"({attempt_index}/{len(grasps)})."
            )
            return grasp, plan_grasp, trajs, jnames, attempts

        return None, None, None, None, attempts

    def synthesize(self):
        """Pre-generate the solution (trajectory) for the episode
        Returns:
            True if synthesis completed successfully
            False if planning failed
            "phase_break" if a phase break was encountered (multi-phase planning)
        """
        # 调用 task.decompose() 获取所有的子任务，需要任务自定义如何分解
        # 在多phrase任务中，这里应该都是相同结果，然后通过_subtask_index来跳过已经处理的子任务
        all_subtasks = self.task.decompose()

        # Skip already processed subtasks in multi-phase planning
        subtasks = all_subtasks[self._subtask_index :] if self._subtask_index > 0 else all_subtasks

        # for wholebody locomotion
        self.original_robot_pose = list(self.task.layout.actors["robot"].pose.position) + list(
            self.task.layout.actors["robot"].pose.quaternion
        )
        self.desired_robot_pose = self.original_robot_pose.copy()

        for i, spec in enumerate(subtasks):
            self._subtask_index += 1
            # Phase break
            if isinstance(spec, PhaseBreakSpec):
                return "phase_break"

            # get hand uid and update ee_link for plan
            hand_uid = spec.meta.get("hand_uid", None)
            from humano.robots.protocols import Humanoid

            if isinstance(self.robot, Humanoid):
                if hand_uid is not None:
                    self.robot.update_ee_link(hand_uid)  # type: ignore
                    self.planner.reinit(self.robot)  # type: ignore
            else:
                if hasattr(self.robot, "switch_arm"):
                    if self.robot.switch_arm(hand_uid):  # type: ignore
                        self.planner.reinit(self.robot)  # type: ignore
                print(self.robot.robot_cfg["kinematics"]["ee_link"])  # type: ignore
                print(f"[MP Agent] Switched to {hand_uid} arm")

            # 针对不同的子任务类型，调用不同的规划方法
            # TODO: 学习具体的规划方法
            if isinstance(spec, OpenGripperSpec):
                for _ in range(10):
                    self.queue_open_gripper(hand_uid=hand_uid)
            elif isinstance(spec, CloseGripperSpec):
                for _ in range(10):
                    self.queue_close_gripper(hand_uid=hand_uid)
            elif isinstance(spec, MoveEEFToPoseSpec):
                target_position = spec.meta.get("position", None)
                target_orientation = spec.meta.get("orientation", None)
                grasp_type = spec.meta.get("grasp_type", "parallel_gripper")
                eef_state = spec.meta.get("eef_state", "close_eef")

                if target_position is None:
                    raise ValueError("MoveEEFToPoseSpec requires 'position' in meta")

                if target_orientation is None:
                    try:
                        current_js = getattr(self, "_last_traj_qpos", None)
                        if current_js is not None:
                            # Convert dict to array if needed for FK
                            if isinstance(current_js, dict):
                                # Use motion_gen.joint_names to get correct order
                                joint_names = self.planner.motion_gen.joint_names
                                current_js_array = np.array(
                                    [current_js[name] for name in joint_names], dtype=np.float32
                                )
                                current_js_tensor = torch.tensor(
                                    current_js_array, device="cuda", dtype=torch.float32
                                ).unsqueeze(0)
                            else:
                                current_js_tensor = torch.tensor(
                                    current_js, device="cuda", dtype=torch.float32
                                ).unsqueeze(0)
                            hand_state = self.planner.init_js_ik_solver.fk(current_js_tensor)
                            target_orientation = hand_state.ee_quaternion.detach().cpu().numpy()[0]
                        else:
                            target_orientation = t3d.euler.euler2quat(np.pi, 0, 0)
                    except Exception as e:
                        print(
                            f"[MP Agent] Failed to infer current EE orientation, "
                            f"falling back to default. Error: {e}"
                        )
                        target_orientation = t3d.euler.euler2quat(np.pi, 0, 0)

                robot_actor = self.task.layout.actors["robot"]
                robot_pose = robot_actor.pose.as_matrix()

                T_world_goal = np.eye(4)
                if target_position.shape[0] == 2:
                    T_world_goal[:2, 3] = target_position
                else:
                    T_world_goal[:3, 3] = target_position
                if len(target_orientation) == 4:  # quaternion
                    T_world_goal[:3, :3] = t3d.quaternions.quat2mat(target_orientation)
                else:  # rotation matrix
                    T_world_goal[:3, :3] = target_orientation

                T_robot_goal = np.linalg.inv(robot_pose) @ T_world_goal

                goal_pose_robot_frame = {
                    "position": T_robot_goal[:3, 3],
                    "orientation": t3d.quaternions.mat2quat(T_robot_goal[:3, :3]),
                }

                try:
                    # self._last_traj_qpos = np.array(list(self._last_traj_qpos.values()), dtype=np.float32)[:self.planner.init_js_ik_solver.dof]
                    if grasp_type == "bodex":
                        if isinstance(self._last_traj_qpos, dict):
                            joint_names = self.planner.init_js_ik_solver.joint_names
                            q_list = [self._last_traj_qpos[name] for name in joint_names]
                            self._last_traj_qpos = np.array(q_list, dtype=np.float32)[
                                : self.planner.init_js_ik_solver.dof
                            ]
                        lock_links_names = spec.meta.get("lock_links", None)
                        target_orientation = spec.meta.get("orientation", None)
                        if target_orientation is not None:
                            trajs, jnames = self.planner.batch_plan_for_move_bodex(
                                position=T_robot_goal[:3, 3],
                                orientation=t3d.quaternions.mat2quat(T_robot_goal[:3, :3]),
                                current_joint_states=self._last_traj_qpos,
                                world_layout=self.task.layout,
                                lock_links_names=lock_links_names,
                            )
                        else:
                            trajs, jnames = self.planner.batch_plan_for_move_bodex(
                                position=T_robot_goal[:2, 3],
                                orientation=target_orientation,
                                current_joint_states=self._last_traj_qpos,
                                world_layout=self.task.layout,
                                lock_links_names=lock_links_names,
                            )
                    else:
                        trajs, jnames = self.planner.batch_plan_for_move(
                            [goal_pose_robot_frame],
                            current_joint_states=self._last_traj_qpos,
                            world_layout=self.task.layout,
                        )

                except Exception as e:
                    print(f"Planning to target pose failed with error: {e}")
                    return False

                traj = []
                for qpos in trajs[-1]:
                    from humano.robots.protocols import Humanoid

                    init_qpos = self.robot.init_joint_states
                    if isinstance(self.robot, Humanoid):
                        traj.append(pad_to_full_qpos(dict(zip(jnames, qpos)), init_qpos))
                    else:
                        traj.append(dict(zip(jnames, qpos)))

                self.queue_follow_path_with_eef(
                    traj,
                    eef_state,
                    **self._trajectory_action_kwargs(hand_uid=hand_uid),
                )
                self._last_traj_qpos = traj[-1]  # HACK store for retreat planning

                self._last_eef_orientation = target_orientation
                if self._last_eef_orientation is None:
                    self._last_eef_orientation = t3d.euler.euler2quat(np.pi, 0, 0)

            elif isinstance(spec, GraspObjectSpec):
                target_uid = spec.meta["target_uid"]
                target_actor = self.task.layout.actors["target"]
                robot_actor = self.task.layout.actors["robot"]

                assert target_uid == target_actor.uid
                object_pose = target_actor.pose

                assert isinstance(target_actor, ObjectActor)
                assert isinstance(target_actor.asset, SpatialAnnotated)

                # grasp poses in SIMPLE world frame
                grasp_type = spec.meta.get("grasp_type", "parallel_gripper")
                grasp_bias = spec.meta.get("grasp_bias", None)
                store_grasp_key = spec.meta.get("store_grasp_key", None)
                use_negative_x_filter = spec.meta.get("use_negative_x_filter", False)
                approach_axis = spec.meta.get("approach_axis", "z")

                if grasp_type == "parallel_gripper":
                    if use_negative_x_filter:
                        reference_grasps = None
                        if "left_arm_grasps" in self._stored_grasps:
                            reference_grasps = self._stored_grasps["left_arm_grasps"]
                            print(
                                f"[MP Agent] Using {len(reference_grasps)} stored left arm grasps as reference"
                            )

                        grasps = GSNet.load_cached_grasps_filter_by_negative_x(
                            target_actor.asset,
                            stable_idx=self.target_stable_index,
                            target_pose=object_pose,
                            max_grasps=self.plan_batch_size,
                            angle_threshold=20.0,
                            reference_grasps=reference_grasps,
                        )
                        print(
                            f"[MP Agent] Found {len(grasps)} grasps approaching from -X direction"
                        )
                        self.target_stable_index = grasps[0][
                            "stable_idx"
                        ]  # save target stable_idx of env
                    else:
                        grasps = GSNet.load_cached_grasps(
                            target_actor.asset,
                            target_pose=object_pose,
                            max_grasps=self.plan_batch_size,
                            bias=grasp_bias,
                        )
                        if grasps is None:
                            print(f"[MP Agent] No grasps found for object {target_uid}")
                            return False
                        self.target_stable_index = grasps[0][
                            "stable_idx"
                        ]  # save target stable_idx of env
                        if store_grasp_key is not None:
                            self._stored_grasps[store_grasp_key] = grasps
                            print(
                                f"[MP Agent] Stored {len(grasps)} grasps with key '{store_grasp_key}'"
                            )
                elif grasp_type == "bodex":
                    selection = spec.meta.get("grasp_selection", "ranked")
                    max_candidates = spec.meta.get(
                        "max_grasp_candidates", max(8, self.plan_batch_size)
                    )
                    grasps = Bodex.load_cached_grasps(
                        self.robot,
                        target_actor.asset,
                        target_pose=object_pose,
                        max_grasps=max_candidates,
                        hand_uid=hand_uid,
                        selection=selection,
                        grasp_idx=spec.meta.get("grasp_idx", None),
                        seed=spec.meta.get("grasp_seed", None),
                    )
                    if not grasps:
                        print(f"[MP Agent] No Bodex candidates found for object {target_uid}")
                        return False
                else:
                    raise ValueError(f"Unsupported grasp type: {grasp_type}")

                if grasp_type == "bodex":
                    selected_grasp, selected_plan_grasp, trajs, jnames, attempts = (
                        self._plan_bodex_candidates(
                            grasps,
                            spec,
                            robot_actor.pose.as_matrix(),
                        )
                    )
                    self.last_grasp_plan = {
                        "target_uid": target_uid,
                        "hand_uid": hand_uid,
                        "grasp_selection": selection,
                        "stable_idx": grasps[0].get("stable_idx"),
                        "attempts": attempts,
                        "layout_target_pose_world": self._trace_pose(object_pose),
                    }
                    if selected_grasp is None:
                        print(
                            f"[MP Agent] All {len(grasps)} Bodex candidates failed "
                            f"for object {target_uid}"
                        )
                        return False
                    self.last_grasp_plan.update(
                        {
                            "selected_candidate_idx": selected_grasp.get(
                                "grasp_candidate_idx",
                                selected_grasp.get("grasp_id"),
                            ),
                            "selected_grasp_position_world": self._trace_vector(
                                selected_grasp.get("position")
                            ),
                            "selected_grasp_orientation_world": self._trace_vector(
                                selected_grasp.get("orientation")
                            ),
                            "selected_goal_position_robot": self._trace_vector(
                                selected_plan_grasp.get("position")
                            ),
                            "selected_goal_orientation_robot": self._trace_vector(
                                selected_plan_grasp.get("orientation")
                            ),
                        }
                    )
                    grasps = [selected_grasp]
                    object_grasp_poses = grasps
                    plan_grasp_proses = [selected_plan_grasp]
                else:
                    # Pad more grasps if not enough.
                    object_grasp_poses = (
                        grasps[: self.plan_batch_size]
                        if len(grasps) >= self.plan_batch_size
                        else grasps + [grasps[0]] * (self.plan_batch_size - len(grasps))
                    )
                    plan_grasp_proses = deepcopy(object_grasp_poses)
                    assert isinstance(self.robot, Graspable)
                    for i in range(len(object_grasp_poses)):
                        # cuRobo plans in robot base frame.
                        goal_pose = self.robot.get_grasp_pose_wrt_robot(
                            object_grasp_poses[i],
                            pregrasp=spec.meta.get("pregrasp", False),
                            robot_pose=robot_actor.pose.as_matrix(),
                        )
                        plan_grasp_proses[i]["position"] = goal_pose[0]
                        plan_grasp_proses[i]["orientation"] = goal_pose[1]

                ### Visualization the first grasp for debugging
                ### convert from robot base frame to world frame for visualization
                if self.debug:
                    self.task.layout.add_visual_grasp("target", grasps[0])

                    # Visualize planned grasp in robot frame
                    robot_pose = robot_actor.pose.as_matrix()
                    T_plan_grasp = np.eye(4)
                    T_plan_grasp[:3, 3] = plan_grasp_proses[0]["position"]
                    T_plan_grasp[:3, :3] = t3d.quaternions.quat2mat(
                        plan_grasp_proses[0]["orientation"]
                    )
                    goal_in_world_frame = robot_pose @ T_plan_grasp
                    goal_pose_vec = np.concatenate(
                        [
                            goal_in_world_frame[:3, 3],
                            t3d.quaternions.mat2quat(goal_in_world_frame[:3, :3]),
                        ]
                    )
                    self.task.layout.add_visual_frame("eef", goal_pose_vec.tolist())

                    # visualize bodex hand pose
                    if grasp_type == "bodex":
                        for key, pose in grasps[0]["link_poses"].items():
                            T_world_linkpose = np.eye(4)
                            T_world_linkpose[:3, 3] = pose.position.cpu().numpy().squeeze(0)
                            T_world_linkpose[:3, :3] = t3d.quaternions.quat2mat(
                                pose.quaternion.cpu().numpy().squeeze(0)
                            )
                            # goal_in_world_frame = T_plan_bodexgrasp
                            bodex_goal_pose_vec = np.concatenate(
                                [
                                    T_world_linkpose[:3, 3],
                                    t3d.quaternions.mat2quat(T_world_linkpose[:3, :3]),
                                ]
                            )
                            self.task.layout.add_visual_frame(
                                f"bodex_{key}", bodex_goal_pose_vec.tolist()
                            )

                if grasp_type == "parallel_gripper":
                    try:
                        trajs, jnames = self.planner.batch_plan_for_approach(
                            plan_grasp_proses,
                            world_layout=self.task.layout,
                            approach_axis=approach_axis,
                        )
                    except Exception as exc:
                        print(f"Planning failed with error: {exc}")
                        return False
                elif grasp_type != "bodex":
                    raise ValueError(f"Unsupported grasp type: {grasp_type}")

                init_qpos = self.robot.init_joint_states
                traj = []
                for qpos in trajs[-1]:  # only taks the last successful one
                    from humano.robots.protocols import Humanoid

                    if isinstance(self.robot, Humanoid):
                        traj.append(pad_to_full_qpos(dict(zip(jnames, qpos)), init_qpos))
                    else:
                        # not support g1
                        traj.append(dict(zip(jnames, qpos)))
                # self.queue_follow_path_with_eef(traj, "open_eef")

                # FIXME only for g1 handover
                keep_force = spec.meta.get("keep_force", False)
                self.queue_follow_path(
                    traj,
                    **self._trajectory_action_kwargs(
                        hand_uid=hand_uid,
                        keep_force=keep_force,
                    ),
                )

                # A SONIC planner command can lag the cuRobo source waypoint
                # stream. Keep the completed squeeze pose before starting the
                # lift, but only when the task explicitly requests it.
                grasp_settle_s = float(spec.meta.get("grasp_settle_s", 0.0))
                self._queue_qpos_hold(
                    traj[-1],
                    grasp_settle_s,
                    **self._trajectory_action_kwargs(
                        hand_uid=hand_uid,
                        keep_force=keep_force,
                        hold_reason="grasp_settle",
                    ),
                )

                # self._grasp_qpos = traj[-1] # HACK store for lift planning
                self._last_traj_qpos = traj[-1]  # HACK store for lift planning

            elif isinstance(spec, LiftSpec):
                grasp_type = spec.meta.get("grasp_type", "parallel_gripper")
                lift_height = spec.meta.get("up", 0.1)
                step_distance = spec.meta.get("step_distance", 0.01)
                eef_state = spec.meta.get("eef_state", "close_eef")

                try:
                    if grasp_type == "bodex":
                        traj = []
                        init_qpos = self.robot.init_joint_states
                        plan_traj = self.planner.batch_plan_for_lift_bodex(
                            self._last_traj_qpos, lift_height, step_distance=0.002
                        )
                        for qpos in plan_traj:
                            traj.append(pad_to_full_qpos(qpos, init_qpos))
                    else:
                        traj = self.planner.batch_plan_for_lift(
                            self._last_traj_qpos, lift_height, step_distance=step_distance
                        )
                except Exception as e:
                    # raise e
                    print(f"Lift planning failed with error: {e}")
                    return False

                # if isinstance(self.robot, Humanoid):
                #     traj
                self.queue_follow_path_with_eef(
                    traj,
                    eef_state,
                    **self._trajectory_action_kwargs(hand_uid=hand_uid),
                )
                # self._lift_qpos_dict = traj[-1] # HACK store for move planning
                if grasp_type == "bodex":
                    self._last_traj_qpos = np.array(list(traj[-1].values()), dtype=np.float32)[
                        : self.planner.init_js_ik_solver.dof
                    ]
                else:
                    self._last_traj_qpos = traj[-1]

            elif isinstance(spec, LowerSpec):
                grasp_type = spec.meta.get("grasp_type", "parallel_gripper")
                lower_height = spec.meta.get("down")
                try:
                    if grasp_type == "bodex":
                        traj = []
                        init_qpos = self.robot.init_joint_states
                        plan_traj = self.planner.batch_plan_for_lift_bodex(
                            self._last_traj_qpos, lift_height=lower_height, step_distance=-0.002
                        )
                        for qpos in plan_traj:
                            traj.append(pad_to_full_qpos(qpos, init_qpos))
                    else:
                        traj = self.planner.batch_plan_for_lift(
                            self._last_traj_qpos, lift_height=lower_height, step_distance=-0.01
                        )  # type: ignore
                except Exception as e:
                    # raise e
                    print(f"Lower planning failed with error: {e}")
                    return False

                self.queue_follow_path_with_eef(
                    traj,
                    "close_eef",
                    **self._trajectory_action_kwargs(),
                )
                if grasp_type == "bodex":
                    self._last_traj_qpos = np.array(list(traj[-1].values()), dtype=np.float32)[
                        : self.planner.init_js_ik_solver.dof
                    ]
                else:
                    self._last_traj_qpos = traj[-1]  # HACK store for move planning
                # self._lower_qpos = np.array(list(traj[-1].values()), dtype=np.float32)

            elif isinstance(spec, RetreatSpec):
                if isinstance(self._last_traj_qpos, dict):
                    joint_names = self.planner.init_js_ik_solver.joint_names
                    q_list = [self._last_traj_qpos[name] for name in joint_names]
                    self._last_traj_qpos = np.array(q_list, dtype=np.float32)[
                        : self.planner.init_js_ik_solver.dof
                    ]
                grasp_type = spec.meta.get("grasp_type", "parallel_gripper")
                try:
                    if grasp_type == "bodex":
                        lock_links_names = spec.meta.get("lock_links", None)
                        trajs, jnames = self.planner.batch_plan_for_retreat_bodex(
                            current_joint_states=self._last_traj_qpos,
                            world_layout=self.task.layout,
                            lock_links_names=lock_links_names,
                        )
                        traj = []
                        for qpos in trajs[-1]:
                            from humano.robots.protocols import Humanoid

                            init_qpos = self.robot.init_joint_states
                            if isinstance(self.robot, Humanoid):
                                traj.append(pad_to_full_qpos(dict(zip(jnames, qpos)), init_qpos))
                            else:
                                traj.append(dict(zip(jnames, qpos)))

                    else:
                        traj = self.planner.batch_plan_for_lift(self._last_traj_qpos)
                except Exception as e:
                    # raise e
                    print(f"Retreat planning failed with error: {e}")
                    return False
                self.queue_follow_path_with_eef(
                    traj,
                    "open_eef",
                    **self._trajectory_action_kwargs(),
                )
                self._last_traj_qpos = traj[-1]

            # FOR WHOLEBODY LOCOMOTION
            # command: [vx, yaw,vy,height,torso yaw, torso pitch,torso roll]
            elif isinstance(spec, StandSpec):
                target_yaw = spec.meta.get("target_yaw", 0)
                command = [0, target_yaw, 0, 0, 0, 0, 0, 0]
                keep_waist_pose = spec.meta.get("keep_waist_pose", False)
                steps = spec.meta.get("steps", 60)
                for _ in range(steps):
                    self.queue_loco_command(
                        command=command.copy(),
                        motion_type="stand",
                        **self._locomotion_action_kwargs(keep_waist_pose=keep_waist_pose),
                    )

                """ # TODO DEBUG
                for i in range(20):
                    jnames = self.task.robot.joint_names
                    qpos = np.zeros((43,), dtype=np.float32)
                    qpos[-5] = -0.85/20 * (i+1) # right thumb 2
                    target_qpos = dict(zip(jnames, qpos))
                    self.queue_move_qpos(target_qpos) """

            elif isinstance(spec, WalkSpec):
                keep_waist_pose = spec.meta.get("keep_waist_pose", False)
                walk_distance = spec.meta.get("target_distance", 0.5)
                target_yaw = spec.meta.get("target_yaw", 0)
                command = [0, 0, 0, 0, 0, 0, 0, 0]
                vx = spec.meta.get("vx", 0.1)
                vy = spec.meta.get("vy", 0)
                command[0] = vx
                command[2] = vy
                command[1] = target_yaw
                command[3] = spec.meta.get("height", 0.0)
                command[4] = spec.meta.get("torso_yaw", 0)
                command[5] = spec.meta.get("torso_pitch", 0)
                command[6] = spec.meta.get("torso_roll", 0)

                x_origin = self.desired_robot_pose[0]
                y_origin = self.desired_robot_pose[1]
                # calculate the desired robot pose
                x_desired = x_origin + walk_distance * np.cos(target_yaw)
                y_desired = y_origin + walk_distance * np.sin(target_yaw)
                self.desired_robot_pose[0] = round(x_desired, 3)
                self.desired_robot_pose[1] = round(y_desired, 3)
                self.desired_robot_pose[3:] = t3d.euler.euler2quat(0, 0, target_yaw)

                desired_robot_pose = self.desired_robot_pose.copy()
                steps = spec.meta.get("steps")
                if steps is None:
                    if abs(vx) < abs(vy):
                        steps = int(walk_distance / 10 / 0.002 / (abs(vy) - 0.1) + 30)
                    else:
                        steps = int(walk_distance / 10 / 0.002 / (abs(vx) - 0.1) + 30)
                steps = int(steps)
                if steps <= 0:
                    raise ValueError("WalkSpec 'steps' must be positive")

                for _ in range(steps):
                    self.queue_loco_command(
                        command=command.copy(),
                        desired_robot_pose=desired_robot_pose,
                        motion_type="walk",
                        keep_waist_pitch=True,
                        **self._locomotion_action_kwargs(keep_waist_pose=keep_waist_pose),
                    )

            elif isinstance(spec, TurnSpec):
                keep_waist_pose = spec.meta.get("keep_waist_pose", False)
                target_yaw = spec.meta.get("target_yaw", 0)
                command = [0, 0, 0, 0, 0, 0, 0, 1]
                command[1] = target_yaw
                command[0] = spec.meta.get("vx", 0.0)
                command[3] = spec.meta.get("height", 0.0)
                self.desired_robot_pose[3:] = t3d.euler.euler2quat(0, 0, target_yaw)
                desired_robot_pose = self.desired_robot_pose.copy()

                total_steps = 100
                turn_ratio = 0.7
                active_steps = int(total_steps * turn_ratio)

                start_yaw = self.last_target_yaw
                yaw_diff = target_yaw - start_yaw

                for step in range(total_steps):
                    if step < active_steps:
                        ratio = step / (active_steps - 1)
                        curve_progress = ratio
                        yaw = start_yaw + yaw_diff * curve_progress

                    else:
                        yaw = target_yaw

                    command[1] = yaw

                    self.queue_loco_command(
                        command=command.copy(),
                        desired_robot_pose=desired_robot_pose,
                        motion_type="turn",
                        **self._locomotion_action_kwargs(keep_waist_pose=keep_waist_pose),
                    )

                self.last_target_yaw = target_yaw
            elif isinstance(spec, HeightAdjustSpec):
                keep_waist_pose = spec.meta.get("keep_waist_pose", False)
                target_yaw = spec.meta.get("target_yaw", 0)
                height = spec.meta.get("height", 0)
                vx = spec.meta.get("vx", 0)
                vy = spec.meta.get("vy", 0)
                command = [vx, target_yaw, vy, height, 0, 0, 0, 0]
                for _ in range(60):
                    self.queue_loco_command(
                        command=command,
                        desired_height=height,
                        motion_type="height_adjust",
                        **self._locomotion_action_kwargs(keep_waist_pose=keep_waist_pose),
                    )

            else:
                raise ValueError(f"Unknown subtask spec: {type(spec)}")

        return True

    def reset(self):
        """Reset the agent state, including subtask index for multi-phase planning."""
        super().reset()
        Bodex.reset()
        self._subtask_index = 0
        self._planning_current_joint_qpos = None
        self._last_traj_qpos = None
        self.last_target_yaw = 0
        self.last_grasp_plan = None
