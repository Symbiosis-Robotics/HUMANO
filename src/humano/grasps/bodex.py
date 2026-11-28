import numpy as np
import transforms3d as t3d
from typing import Optional
from pathlib import Path
from humano.core.asset import Asset
from humano.core.object import SpatialAnnotated, SemanticAnnotated
from humano.core.types import Pose
import torch
import os
import re
from humano.robots.protocols import HasDexterousHand
from humano.utils import resolve_data_path

class AnnoatedAsset(Asset, SpatialAnnotated, SemanticAnnotated):
    ...

class Bodex:
    def __init__(self):
        pass


    def detect_grasps(self, obj):
        pass
    @classmethod
    def reset(cls):
        # Kept for callers that reset grasp providers between episodes. Stable
        # pose selection is now local to each load call.
        return None
    @staticmethod
    def create_hand_curobo_model(config_path:str):
        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
        from curobo.types.base import TensorDeviceType
        from curobo.types.robot import RobotConfig
        from curobo.util_file import load_yaml
        from humano.utils import resolve_data_path

        tensor_args = TensorDeviceType()
        config_file_path = Path(config_path)
        if not config_file_path.is_absolute():
            config_file_path = Path(resolve_data_path(config_path, auto_download=True))

        config_file=load_yaml(str(config_file_path))["robot_cfg"]
        kinematics = config_file["kinematics"]

        collision_spheres = Path(kinematics["collision_spheres"])
        if not collision_spheres.is_absolute():
            kinematics["collision_spheres"] = str(config_file_path.parent / collision_spheres)

        external_asset_path = kinematics.get("external_asset_path")
        if external_asset_path is not None and not Path(external_asset_path).is_absolute():
            external_asset_path = resolve_data_path(external_asset_path, auto_download=True)
            kinematics["external_asset_path"] = external_asset_path

        external_robot_configs_path = kinematics.get("external_robot_configs_path")
        if (
            external_robot_configs_path is not None
            and not Path(external_robot_configs_path).is_absolute()
        ):
            kinematics["external_robot_configs_path"] = resolve_data_path(
                external_robot_configs_path,
                auto_download=True,
            )

        urdf_path = Path(kinematics["urdf_path"])
        if not urdf_path.is_absolute():
            if external_asset_path is not None:
                kinematics["urdf_path"] = str(Path(external_asset_path) / urdf_path)
            else:
                kinematics["urdf_path"] = resolve_data_path(str(urdf_path), auto_download=True)
        robot_config = RobotConfig.from_dict(config_file,tensor_args)

        hand_model = CudaRobotModel(robot_config.kinematics)
        return hand_model


    @staticmethod
    def transform_linkpose_to_world(link_pose , T_bodexworld_object: np.ndarray, T_world_object: np.ndarray):
        position = link_pose.position.cpu().numpy().squeeze(0)
        quaternion = link_pose.quaternion.cpu().numpy().squeeze(0)
        T_bodexworld_linkpose = np.eye(4)
        T_bodexworld_linkpose[:3, 3] = position
        T_bodexworld_linkpose[:3, :3] = t3d.quaternions.quat2mat(quaternion)

        T_object_linkpose =np.linalg.inv(T_bodexworld_object) @ T_bodexworld_linkpose

      
        T_world_linkpose = T_world_object @ T_object_linkpose

        new_position = T_world_linkpose[:3,3]

        new_quaternion = t3d.quaternions.mat2quat(T_world_linkpose[:3,:3])

        link_pose.position = torch.from_numpy(new_position).float().to(link_pose.position.device).unsqueeze(0)

        link_pose.quaternion = torch.from_numpy(new_quaternion).float().to(link_pose.position.device).unsqueeze(0)

        return link_pose


        
    
    @classmethod
    def load_cached_grasps_debug(
        cls, 
        robot: HasDexterousHand,
        asset: AnnoatedAsset, 
        stable_idx: Optional[str] = None, 
        target_pose: Optional[Pose] = None,
        max_grasps: Optional[int] = None,
        hand_uid: Optional[str] = None
    ) -> list:
        
        """  
        Load cached grasps for the given asset. transform grasps to world frame using target_pose.
        Args:
            asset (SpatialAnnotated): The asset to load grasps for.
            stable_idx (Optional[str], optional): The stable pose index. Defaults to None.
            target_pose (Optional[Pose], optional): The target pose of the asset in world frame
            max_grasps (Optional[int], optional): Maximum number of grasps to load. Defaults to None.
        Returns:
            list: A list of grasp dictionaries (in world frame) containing position, orientation, width, depth, score,
        """
        hand_model = cls.create_hand_curobo_model(
            resolve_data_path(robot.hand_yaml, auto_download=True)
        )
        hand_dof = robot.hand_dof
        object_stable_poses = asset.stable_poses[0] # np.load("data/assets/graspnet/stable/12_stable.npy", allow_pickle=True)[0]
        T_bodexworld_object = np.eye(4)
        T_bodexworld_object[:3, 3] = [0.4000000059604645, 0.0, object_stable_poses[2]]
        T_bodexworld_object[:3,:3] = t3d.quaternions.quat2mat([object_stable_poses[2]])

        bodex_asset_path = resolve_data_path(f"objects/graspnet/dex_grasp/{hand_uid}/12/12__1_pregrasp.pt", auto_download=True)
        bodex_asset_pregrasp_path = resolve_data_path(f"objects/graspnet/dex_grasp/{hand_uid}/12/12__1.pt", auto_download=True)

        dex_pregrasp_poses=torch.load(bodex_asset_path)
        dex_grasp_poses= torch.load(bodex_asset_pregrasp_path)
        
      
        kin_state= hand_model.get_state(dex_pregrasp_poses[:])
        link_poses = kin_state.link_poses

        dex_pregrasp_poses = dex_pregrasp_poses.cpu().numpy()
        # dex_grasp_poses = dex_grasp_poses.cpu().numpy()


        dex_grasp = np.load(
            resolve_data_path(
                f"objects/graspnet/dex_grasp/{hand_uid}/12/12__1.npy", auto_download=True
            ),
            allow_pickle=True,
        ).item()

        # get grasp pose in world frame
        T_bodexworld_grasp = np.eye(4)
        T_bodexworld_grasp[:3, 3] = dex_pregrasp_poses[:3]
        T_bodexworld_grasp[:3, :3] = t3d.quaternions.quat2mat(dex_pregrasp_poses[3:7])

        T_object_grasp = np.linalg.inv(T_bodexworld_object) @ T_bodexworld_grasp

        # FIXME rotation the success rate is very low , need to be fixed
        T_world_object = target_pose.as_matrix()
        T_world_object[:3, :3] = T_bodexworld_object[:3, :3]
        T_world_grasp = T_world_object @ T_object_grasp

        # get T_world_link_poses
        for key,pose in link_poses.items():
            link_poses[key] = cls.transform_linkpose_to_world(pose, T_bodexworld_object, T_world_object)


        position = T_world_grasp[:3,3]
        
        orientation = t3d.quaternions.mat2quat(T_world_grasp[:3,:3])

        #get grasp qpos,squeeze qpos,lift qpos
        grasp_qpos=dex_grasp["robot_pose"][:,-3,:]
        squeeze_qpos=dex_grasp["robot_pose"][:,-2,:]
        lift_qpos=dex_grasp["robot_pose"][:,-1,:]
        
        if "dex3" in robot.hand_yaml:
            grasp_qpos= np.concatenate([grasp_qpos[... , :-hand_dof] , grasp_qpos[..., -hand_dof:][..., [3,4,5,6,0,1,2]]],axis=1)
            squeeze_qpos= np.concatenate([squeeze_qpos[... , :-hand_dof] , squeeze_qpos[..., -hand_dof:][..., [3,4,5,6,0,1,2]]] , axis=1)
            lift_qpos = np.concatenate( [lift_qpos[... , :-hand_dof] , lift_qpos[..., -hand_dof:][..., [3,4,5,6,0,1,2]]] , axis=1 )

        grasp_poses = [{
            'position': position, 
            'orientation': orientation,
            'width': 0.04,
            'depth': 0, #0.005,
            'score': 1,
            # 'object_id': target_id,
            'grasp_id': 12,
            # "qpos": dex_grasp["pregrasp_qpos"][7:14]
            "link_poses": link_poses,
            "all_qpos" : dex_grasp["robot_pose"][0],
            "grasp_qpos": grasp_qpos,
            "squeeze_qpos": squeeze_qpos,
            "lift_qpos": lift_qpos,
            ## TODO pass links poses instead
        } ]

        return grasp_poses

    @classmethod
    def load_cached_grasps(
        cls,
        robot: HasDexterousHand,
        asset: AnnoatedAsset,
        stable_idx: Optional[int] = None,
        target_pose: Optional[Pose] = None,
        max_grasps: Optional[int] = None,
        hand_uid: Optional[str] = None,
        selection: str = "ranked",
        grasp_idx: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> list:
        """Load deterministic, target-specific Bodex candidates in world frame.

        ``ranked`` currently means increasing cached candidate id. Bodex cache
        files do not contain a comparable quality score, so this is an explicit
        deterministic baseline rather than an invented learned ranking.
        """
        if target_pose is None:
            raise ValueError("Bodex grasp loading requires target_pose")
        if hand_uid is None:
            raise ValueError("Bodex grasp loading requires hand_uid")

        uid = asset.uid
        grasp_folder = resolve_data_path(
            f"objects/graspnet/dex_grasp/{hand_uid}/{uid}/", auto_download=True
        )
        candidate_indices = cls._order_candidate_indices(
            cls._cached_candidate_indices(grasp_folder, uid),
            selection=selection,
            grasp_idx=grasp_idx,
            seed=seed,
        )
        if max_grasps is not None:
            candidate_indices = candidate_indices[: max(0, max_grasps)]
        if not candidate_indices:
            return []

        hand_model = cls.create_hand_curobo_model(
            resolve_data_path(robot.hand_yaml, auto_download=True)
        )
        hand_dof = robot.hand_dof
        selected_stable_idx = cls._select_stable_idx(
            asset.stable_poses, target_pose, stable_idx
        )
        stable_position = asset.stable_poses[selected_stable_idx][:3]
        stable_quaternion = asset.stable_poses[selected_stable_idx][3:]

        T_bodexworld_object = np.eye(4)
        T_bodexworld_object[:3, 3] = [0.4, 0.0, stable_position[2]]
        T_bodexworld_object[:3, :3] = t3d.quaternions.quat2mat(stable_quaternion)
        T_world_object = target_pose.as_matrix()

        grasp_poses = []
        for candidate_rank, candidate_idx in enumerate(candidate_indices):
            dex_pregrasp_poses = torch.load(
                os.path.join(grasp_folder, f"{uid}__{candidate_idx}_pregrasp.pt")
            )
            dex_grasp = np.load(
                os.path.join(grasp_folder, f"{uid}__{candidate_idx}.npy"),
                allow_pickle=True,
            ).item()
            kin_state = hand_model.get_state(dex_pregrasp_poses[:])
            link_poses = kin_state.link_poses
            pregrasp_pose = dex_pregrasp_poses.cpu().numpy()

            T_bodexworld_grasp = np.eye(4)
            T_bodexworld_grasp[:3, 3] = pregrasp_pose[:3]
            T_bodexworld_grasp[:3, :3] = t3d.quaternions.quat2mat(pregrasp_pose[3:7])
            T_world_grasp = T_world_object @ (
                np.linalg.inv(T_bodexworld_object) @ T_bodexworld_grasp
            )
            for key, pose in link_poses.items():
                link_poses[key] = cls.transform_linkpose_to_world(
                    pose, T_bodexworld_object, T_world_object
                )

            grasp_qpos = dex_grasp["robot_pose"][:, -3, :]
            squeeze_qpos = dex_grasp["robot_pose"][:, -2, :]
            lift_qpos = dex_grasp["robot_pose"][:, -1, :]
            if "dex3" in robot.hand_yaml:
                # FIXME(inspire): add an Inspire-specific Bodex cache format and
                # qpos mapping before enabling Bodex grasping for that hand.
                hand_order = [3, 4, 5, 6, 0, 1, 2]
                grasp_qpos = np.concatenate(
                    [grasp_qpos[..., :-hand_dof], grasp_qpos[..., -hand_dof:][..., hand_order]], axis=1
                )
                squeeze_qpos = np.concatenate(
                    [squeeze_qpos[..., :-hand_dof], squeeze_qpos[..., -hand_dof:][..., hand_order]], axis=1
                )
                lift_qpos = np.concatenate(
                    [lift_qpos[..., :-hand_dof], lift_qpos[..., -hand_dof:][..., hand_order]], axis=1
                )

            grasp_poses.append(
                {
                    "position": T_world_grasp[:3, 3],
                    "orientation": t3d.quaternions.mat2quat(T_world_grasp[:3, :3]),
                    "width": 0.04,
                    "depth": 0.0,
                    "score": 1.0,
                    "grasp_id": uid,
                    "grasp_candidate_idx": int(candidate_idx),
                    "candidate_rank": int(candidate_rank),
                    "stable_idx": int(selected_stable_idx),
                    "target_uid": uid,
                    "link_poses": link_poses,
                    "all_qpos": dex_grasp["robot_pose"][0],
                    "grasp_qpos": grasp_qpos,
                    "squeeze_qpos": squeeze_qpos,
                    "lift_qpos": lift_qpos,
                }
            )
        return grasp_poses

    @staticmethod
    def _select_stable_idx(stable_poses, target_pose: Pose, stable_idx: Optional[int]) -> int:
        if stable_idx is not None:
            if stable_idx < 0 or stable_idx >= len(stable_poses):
                raise ValueError(f"stable_idx {stable_idx} is outside [0, {len(stable_poses)})")
            return int(stable_idx)
        for index, pose in enumerate(stable_poses):
            if np.allclose(
                np.asarray(pose[2:3]),
                np.asarray(target_pose.position[2:3]),
                rtol=0,
                atol=1e-2,
            ):
                return index
        print(
            "[Bodex.load_cached_grasps] Warning: No exact stable pose match found. "
            "Using first stable pose (index 0)."
        )
        return 0

    @staticmethod
    def _cached_candidate_indices(grasp_folder: str, uid: str) -> list[int]:
        folder = Path(grasp_folder)
        pattern = re.compile(rf"^{re.escape(uid)}__(\d+)_pregrasp\.pt$")
        candidate_indices = []
        for path in folder.iterdir():
            match = pattern.match(path.name)
            if match is None:
                continue
            candidate_idx = int(match.group(1))
            if (
                (folder / f"{uid}__{candidate_idx}.pt").is_file()
                and (folder / f"{uid}__{candidate_idx}.npy").is_file()
            ):
                candidate_indices.append(candidate_idx)
        if not candidate_indices:
            raise FileNotFoundError(
                f"No complete Bodex grasp candidates for target {uid!r} in {folder}"
            )
        return sorted(candidate_indices)

    @staticmethod
    def _order_candidate_indices(
        candidate_indices: list[int],
        *,
        selection: str,
        grasp_idx: Optional[int],
        seed: Optional[int],
    ) -> list[int]:
        if selection == "fixed":
            if grasp_idx is None:
                raise ValueError("Bodex fixed selection requires grasp_idx")
            if grasp_idx not in candidate_indices:
                raise ValueError(
                    f"Bodex grasp_idx {grasp_idx} is not present in cached candidates {candidate_indices}"
                )
            return [int(grasp_idx)]
        if selection == "ranked":
            return list(candidate_indices)
        if selection == "random":
            ordered = list(candidate_indices)
            np.random.default_rng(seed).shuffle(ordered)
            return ordered
        raise ValueError(
            f"Unsupported Bodex grasp selection {selection!r}; expected fixed, ranked, or random"
        )

        


        
