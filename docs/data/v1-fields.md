# HUMANO training data

From the dataset root, select `train/<suite>/<task>/` for training. Each task directory contains its demonstrations, metadata and normalization statistics.

## Model inputs and targets

For the default 7-DoF-hand SONIC policy, build each training sample from the following fields, in the order shown.

| Role | Read from the dataset | What the model uses |
| --- | --- | --- |
| Image input | `observation.images.ego_view` | Current ego RGB frame, decoded from MP4 by the dataset reader |
| Language input | `task_index` → `meta/tasks.jsonl` | Task instruction text |
| State input | `observation.state` (43), then `observation.projected_gravity` (3) | Concatenate into one 46-dimensional vector |
| Action target per step | `action.motion_token` (64), then `teleop.left_hand_joints` (7), then `teleop.right_hand_joints` (7) | Predict one 78-dimensional vector in token, left-hand, right-hand order |

The tested GR00T recipe uses 40 action steps per sample. Hand observations and hand action targets use different joint orderings. Use the task's `meta/modality.json` for modality slices.

The table above gives the complete model inputs and targets for this recipe. The reader uses indices and timestamps to locate samples. The full schema below also includes low-level commands (`action.wbc`), simulation state for replay (`sim.*`), and orientation and hand metadata.

## Stored training fields

Rows are recorded at 50 FPS in LeRobot v2.1 format. Each Parquet has 25 columns. The additional video feature `observation.images.ego_view` is stored as MP4 under `videos/` and declared in `meta/info.json`.

| Parquet column | Stored dtype | Elements per row |
|---|---|---:|
| `observation.state` | float64 | 43 |
| `action.wbc` | float64 | 43 |
| `observation.root_orientation` | float64 | 4 |
| `observation.projected_gravity` | float64 | 3 |
| `observation.cpp_rotation_offset` | float64 | 4 |
| `observation.init_base_quat` | float64 | 4 |
| `action.motion_token` | float64 | 64 |
| `teleop.left_hand_joints` | float32 | 7 |
| `teleop.right_hand_joints` | float32 | 7 |
| `sim.time` | float64 | 1 |
| `sim.nq` | int64 | 1 |
| `sim.nv` | int64 | 1 |
| `sim.nu` | int64 | 1 |
| `sim.qpos` | float64 | 4096 |
| `sim.qvel` | float64 | 4096 |
| `sim.qacc` | float64 | 4096 |
| `sim.ctrl` | float64 | 4096 |
| `sim.robot_qpos` | float64 | 50 |
| `sim.robot_qvel` | float64 | 49 |
| `timestamp` | float32 | 1 |
| `frame_index` | int64 | 1 |
| `episode_index` | int64 | 1 |
| `index` | int64 | 1 |
| `task_index` | int64 | 1 |
| `hand_type` | string | 1 |

For the padded simulation arrays, read the first `sim.nq` values of `sim.qpos`, the first `sim.nv` values of `sim.qvel` and `sim.qacc`, and the first `sim.nu` values of `sim.ctrl`. `action.wbc` stores low-level commands. When present in episode metadata, `trajectory_physics_path` locates the 200 Hz physics trajectory used for terminal-success replay. Training requires no standalone SONIC episode files.
