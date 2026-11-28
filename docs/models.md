# Models and policy integration

Train models in their own repositories and environments. Run external policy servers in the model environment and evaluation in the HUMANO environment. Built-in policies can be started by the HUMANO launcher.

| Goal | Start here |
| --- | --- |
| Connect a policy | [Adapter and server](#1-connect-a-policy), [examples](../baselines/examples/README.md) |
| Prepare training data | [Dataset fields](#3-dataset-fields), [training and evaluation](#6-train-and-evaluate) |
| Match training to inference | [Observation and action interface](#2-observation-and-action-interface), [field mapping](#map-training-data-to-online-inputs) |
| Validate integration | [Adapter check](#5-check-your-adapter), [evaluation checks](#validate-the-integration) |
| Prepare dependencies | [Installation](install.md), [tasks](benchmark/tasks.md), [language fields](data/language-fields.md) |

## 1. Connect a policy

Copy the [adapter template](../baselines/examples/policy_adapter.py) to your model repository and implement:

- `load_policy(checkpoint)`: load and return the policy.
- `predict(observation)`: preprocess inputs, run inference, and return an unnormalized action chunk.
- `reset()`: clear history, queued actions, and other episode state. Stateless models may use a no-op.

The shared server handles HTTP, array serialization, episode/session metadata, and reset dispatch. Keep normalization, image transforms, and random seeds in the adapter. Apply `inference_seed` in your model framework when provided. The adapter does not need a simulation environment, robot object, or SONIC controller.

Start the server in your **model environment**, replacing the example paths:

```bash
PYTHONPATH=/path/to/HUMANO/src:/path/to/my_model \
python -m humano.policy.server \
  --adapter my_adapter:load_policy --checkpoint /path/to/checkpoint \
  --host 127.0.0.1 --port 22085
```

`humano.policy.server` requires Python 3.10+ and NumPy. It does not load simulation or model frameworks, so you do not need to install all of HUMANO or change your model's Torch/Transformers versions.

Use `127.0.0.1` on one machine. Across machines, bind to a reachable interface and use that host address on the evaluation side. Each server handles one stateful evaluation session at a time. Remote evaluation calls `/reset` at startup, allowing sequential tasks to reuse the server. Request metadata controls episode resets and rejects stale requests.

Connect from the **HUMANO environment**:

```bash
HUMANO_EVAL_INSTRUCTION_SOURCE=data python scripts/run/launch_evaluation.py \
  --external-server --policy remote --host 127.0.0.1 --port 22085 \
  --env-id humano/G1WholebodyGroundBoxPickTeleop-v0 \
  --data-dir data/eval/id/basic/G1WholebodyGroundBoxPickTeleop \
  --eval-dir runs/evals/my_policy/box_lifting --num-episodes 1
```

External-server mode leaves model process management to you. See the [evaluation guide](benchmark/evaluation.md) for suite runs and reporting.

## 2. Observation and action interface

The default interface supports G1 with 7-DoF hands and SONIC token actions. Other hands or action spaces require an adapter beyond changing array dimensions.

`predict` receives a dictionary:

| Field | Shape / type | Meaning |
| --- | --- | --- |
| `image.ego_view` | `(360, 640, 3)` RGB | Egocentric HWC image; handle uint8 or normalized floating-point inputs without dividing by 255 twice |
| `state.left_leg` / `right_leg` | `(6,)` each | Measured joint angles, rad |
| `state.waist` | `(3,)` | Measured joint angles, rad |
| `state.left_arm` / `right_arm` | `(7,)` each | Measured joint angles, rad |
| `state.left_hand` / `right_hand` | `(7,)` each | Measured joint angles, rad; ordering below |
| `state.projected_gravity` | `(3,)` | World downward unit vector in the robot body frame, dimensionless |
| `instruction` | `str` | Task instruction |
| `inference_seed` | Optional integer | Inference seed for the model to apply |

Concatenate joint state in this order: left leg, right leg, waist, left arm, left hand, right arm, right hand (**43 dimensions**). Within each group, use the corresponding left/right prefix:

- Legs: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll.
- Waist: waist_yaw, waist_roll, waist_pitch.
- Arms: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw.
- **Hand state**: index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2.

The shared server and current LeRobot integration require a finite NumPy `float32` array of shape **`(40, 78)`**. GR00T uses the horizon configured for its checkpoint.

| Slice | Meaning |
| --- | --- |
| `[:, :64]` | Continuous SONIC motion latent; neither discrete token IDs nor joint angles |
| `[:, 64:71]` | Absolute left-hand joint targets, rad |
| `[:, 71:78]` | Absolute right-hand joint targets, rad |

**Hand actions** use thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1. This differs from hand-state order. Return unnormalized predictions in this order, including both hands; do not fill missing predictions with zeros.

HUMANO queues actions, decodes SONIC tokens, and executes robot commands. The 40-row prediction horizon is not necessarily the number of rows executed per request. Keep execution length, token rate, and control timestep fixed when comparing models.

## 3. Dataset fields

Read `meta/info.json` (`codebase_version`, `fps`, `features`) and `meta/modality.json` before training. Use a LeRobot reader compatible with that dataset version.

The [training field reference](data/v1-fields.md) lists all 25 Parquet columns. Native SONIC datasets with 7-DoF hands use the following fields.

| Stored field | Shape / type | Training use |
| --- | --- | --- |
| `observation.images.ego_view` | `(360, 640, 3)` video | Current image; LeRobot typically decodes it to CHW floating point |
| `observation.state` | `(43,)` | Joint state in the order above; stored as float64, usually cast to float32 |
| `observation.projected_gravity` | `(3,)` | Append to joint state for the 46-dimensional input used by existing LeRobot baselines |
| `observation.root_orientation` | `(4,)` | World orientation quaternion, wxyz; absent from the default online interface |
| `action.motion_token` | `(64,)` | SONIC token for the row, stored as float64 |
| `teleop.left_hand_joints` / `right_hand_joints` | `(7,)` each | Command targets in direct motor order, stored as float32 |
| `action.wbc` | `(43,)` | Low-level joint command log; not the 78-dimensional action label |
| `hand_type` | String | Hand model; 6-DoF data is incompatible with the default interface |
| `task_index` | Index | Resolve task text through LeRobot metadata |
| `timestamp` / `sim.time` | Time | Nominal row-index/FPS time / actual simulation time |

### Read a training batch

This example reads a local dataset with the fields above and 7-DoF hands. It uses the modern `lerobot.datasets` import path; adapt that import for older versions. It maps current inputs to future action labels without training a model or computing dataset statistics.

```python
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path("/path/to/local/dataset")
info = json.loads((root / "meta/info.json").read_text())
horizon = 40
label_keys = (
    "action.motion_token", "teleop.left_hand_joints", "teleop.right_hand_joints"
)
for key, size in zip(label_keys, (64, 7, 7), strict=True):
    assert info["features"][key]["shape"] == [size], (key, info["features"][key])
assert info["features"]["observation.state"]["shape"] == [43]
dataset = LeRobotDataset(
    repo_id=root.name,
    root=root,
    delta_timestamps={key: [i / info["fps"] for i in range(horizon)] for key in label_keys},
)
batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
image = batch["observation.images.ego_view"]  # (B, 3, H, W), RGB, camera frame
state = torch.cat((batch["observation.state"], batch["observation.projected_gravity"]), dim=-1).float()
# (B, 46), joint rad + unit gravity, named joint order / body frame
labels = torch.cat([batch[key] for key in label_keys], dim=-1).float()
# (B, 40, 78), SONIC latent + hand rad, direct motor order
padding = torch.stack([batch[f"{key}_is_pad"] for key in label_keys]).any(dim=0)
# (B, 40), episode boundary mask, True means exclude from loss
instructions = batch["task"]
```

Apply the padding mask to the loss and keep episode boundaries intact. Save image preprocessing, training-only statistics, processor, and checkpoint together for inference.

## 4. Built-in baselines

See the [deployment examples](../baselines/examples/README.md) for launch commands.

| Location | Purpose |
| --- | --- |
| `baselines/<name>/` | Maintained model inference code and source records; external models can stay in their own repositories |
| `src/humano/policy/client.py` | Shared HTTP policy client |
| `src/humano/policy/server.py` | Shared policy server without simulation dependencies |
| `baselines/<name>/deploy/` | Model launchers and adapters |
| `src/humano/policy/sonic_policy.py` | Shared SONIC token execution bridge |
| `src/humano/policy/gr00t_n17.py` | Export of the existing GR00T class |

ACT, Diffusion, Pi0.5, and Flow Matching use the shared server. GR00T retains its own preprocessing and sampling. Psi0 uses SONIC token actions with its own runtime and launcher. Use `--external-server --policy remote` for external models.

## 5. Check your adapter

Run this in the model environment with the `PYTHONPATH` from section 1. Select a supported device in `load_policy`; the server does not choose it.

```python
import numpy as np
from humano.policy.server import IMAGE_SHAPE, STATE_PARTS, validate_action_chunk
from my_adapter import load_policy

policy = load_policy("/path/to/checkpoint")
state = {name: np.zeros(size, dtype=np.float32) for name, size in STATE_PARTS}
# named (D,) joint arrays, rad, public joint order
observation = {
    "image": {"ego_view": np.zeros(IMAGE_SHAPE, dtype=np.uint8)},
    "state": {**state, "projected_gravity": np.array([0, 0, -1], dtype=np.float32)},
    "instruction": "Move the chair.",
}
policy.reset()
action = validate_action_chunk(policy.predict(observation))
# (40, 78), SONIC latent + hand rad, public action order
policy.reset()
print(action.shape, action.dtype)
```

This synthetic observation checks loading, invocation, and finite output shape. It does not validate task performance, transport, or reset semantics. Next, check preprocessing and action order with a real fixed observation, verify reset, and run closed-loop evaluation. Keep the model environment and collection resources separate during these checks.

## 6. Train and evaluate

### Dataset splits

- **Training:** 50 demonstrations per task for supervision and statistics. These task metadata files declare LeRobot v2.1 at 50 FPS. Check each dataset's fields; recorded FPS is not the physics control rate.
- **ID evaluation:** 25 frozen initialization episodes per task. Each contains one frame. Evaluation restores the scene and robot state, then obtains observations from online policy-driven simulation.
- **OOD evaluation:** Background, Camera, Distractor, Language, Lighting, Robot Initial, and Sensor Noise. Each dimension has 100 tasks and 2,500 episodes, giving 20,000 episodes including ID. Paths use `eval/ood/<dimension>/<suite>/<task>/`.
- **Splits:** compute normalization statistics only from training data. Use a documented validation split for checkpoint selection. If none is provided, split training data by episode and record the split; do not tune on formal ID/OOD test sets or distribute adjacent frames from one trajectory across training and validation.

See the [task catalog](benchmark/tasks.md) and [data layout](../data/README.md).

### Locate the dataset

Resolve the training and evaluation paths and record the dataset manifest with the experiment.

```text
data/train/<suite>/<task>/
  meta/info.json       # Version, FPS, field shapes, and path templates
  meta/modality.json   # State/action slices and original_key mapping, if provided
  meta/tasks.jsonl     # v2.1 task_index to task text
  meta/episodes.jsonl # v2.1 episode metadata
  data/               # Parquet
  videos/             # Matching image videos

data/eval/id/<suite>/<task>/   # Frozen initial states and environment configurations
data/eval/ood/                # Conditions and tasks from the version manifest
```

Read the LeRobot format and sampling rate from `meta/info.json`.

Use training demonstrations for supervision and evaluation data for initialization. Keep Parquet files, videos, normalization statistics, and task metadata from the same version. Use the LeRobot-resolved `task` text rather than deriving instructions from directory names; see [language fields](data/language-fields.md).

### Map training data to online inputs

| Training field | Online interface | Adapter responsibility |
| --- | --- | --- |
| `observation.images.ego_view` | `image.ego_view`, HWC RGB | Match training channel order, resize/crop, and normalization |
| `observation.state` | Seven joint groups in `state` | Concatenate in section 2 order, 43 dimensions |
| `observation.projected_gravity` | `state.projected_gravity` | Append if used in training, giving 46 dimensions |
| LeRobot `task` | `instruction` | Use the matching tokenizer/text preprocessing |
| `action.motion_token` | Output `[:, 0:64]` | Unnormalize the 64-dimensional continuous SONIC latent |
| `teleop.left_hand_joints` | Output `[:, 64:71]` | Absolute left-hand targets in rad and action joint order |
| `teleop.right_hand_joints` | Output `[:, 71:78]` | Absolute right-hand targets in rad and action joint order |

For the default 7-DoF hands, split `observation.state` as follows:

| Stored slice | Online field |
| --- | --- |
| `[0:6]` | `state.left_leg` |
| `[6:12]` | `state.right_leg` |
| `[12:15]` | `state.waist` |
| `[15:22]` | `state.left_arm` |
| `[22:29]` | `state.left_hand` |
| `[29:36]` | `state.right_arm` |
| `[36:43]` | `state.right_hand` |

For a 46-dimensional input, `[43:46]` comes from the separate `observation.projected_gravity` field.

The training example predicts 40 action rows starting at the current observation, using offsets `0/fps .. 39/fps`. Recorded FPS, prediction horizon, and online execution frequency are distinct. Each online request supplies the current observation; maintain any history window in the model and clear it in `reset()`.

The default online interface omits `eef_state` and world root orientation. If training requires additional inputs, define how evaluation supplies them rather than silently substituting zeros. Joint-position or end-effector-pose actions also require a separate integration.

### Run inference in two terminals

1. In the model environment, start the server from section 1 with a real adapter implementation. Keep the checkpoint's processor, statistics, and model configuration available.
2. Check `curl http://127.0.0.1:22085/health`. This confirms service availability only.
3. In the HUMANO environment, run the following from the repository root. The direct entry point does not require tmux.

```bash
HUMANO_EVAL_INSTRUCTION_SOURCE=data python -m humano.cli.eval_task \
  humano/G1WholebodyGroundBoxPickTeleop-v0 remote \
  --host 127.0.0.1 --port 22085 \
  --data-format lerobot --sim-mode mujoco_isaac \
  --data-dir data/eval/id/basic/G1WholebodyGroundBoxPickTeleop \
  --eval-dir runs/evals/my_policy/ground_box_pick \
  --num-episodes 1 --episode-start 0 --num-workers 1 \
  --headless --no-save-video --seed 42
```

`HUMANO_EVAL_INSTRUCTION_SOURCE=data` selects the episode instruction. Check that `episodes.tasks`, the task table, and frozen language agree; Language OOD uses the episode's rewrite. Task `description` is an overview, not a policy instruction. See [language fields](data/language-fields.md).

The evaluation environment needs HUMANO's scene resources, Isaac, and SONIC dependencies; the model server does not. This command automatically applies the v1.2 budget for this Basic task. For a short connectivity check, add `--max-episode-steps 80` and label it as such. Use a fresh output directory and match the task ID, data, and model action space.

### Validate the integration

- **Real fixed observation:** verify inference, finite output shape, preprocessing, units, and hand order.
- **One full episode:** check the exit status and `eval_metrics.jsonl` for actual steps, success, and termination reason. Empty metrics do not count as a pass. A task timeout is a valid failure result, not necessarily a service failure.
- **At least two consecutive episodes:** verify that reset clears history and cached actions, then switch tasks and back on the same server. No exception alone does not prove all hidden state was cleared.
- **Formal comparison:** fix the checkpoint, data version, tasks/episodes, seeds, budget, execution settings, and simulation mode. Apply `inference_seed` for stochastic inference; equal seeds do not guarantee frame-identical asynchronous trajectories.

New models normally need only an adapter, launch command, and these validation results. One stateful server serves one evaluation session at a time; any parallel integration needs independent model instances and ports. See the evaluation guide for supported launcher modes.

### Report results

Record the data version, code commit, checkpoint, suite/task/condition, episode range, seeds and repeat counts, simulation mode, time budget, and action execution length. Report successes / evaluated episodes and success rate per task. Separate task failures from service or initialization errors; exclude empty results, missing tasks, and short connectivity checks from full evaluation scores. Retain `eval_metrics.jsonl` and logs.

## 7. Troubleshooting

| Symptom | Check |
| --- | --- |
| Cannot import the server or adapter | Model terminal's `PYTHONPATH` includes HUMANO/src and the adapter directory |
| `/health` works but inference fails | Input preprocessing, checkpoint statistics, finite `(40, 78)` outputs |
| Incorrect image colors or scale | RGB/BGR, HWC/CHW, uint8/float, duplicate division by 255 |
| Incorrect hand motion | State/action hand orders differ; actions must be unnormalized absolute rad targets |
| Problems after reset or task changes | History and action queues are cleared; clients do not share a stateful instance concurrently |
| HTTP 409 for stale requests | Old-session requests or concurrent clients; keep session validation enabled |
| Missing scenes or caches | Install the scene resources and extract the task ZIP with its frozen placement bindings |
