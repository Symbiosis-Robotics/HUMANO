# Quick Start

Set up HUMANO, prepare your model and data, and run evaluation. This walkthrough uses `GroundBoxPick` with your own policy server; each step links to the full guide. For suite runs and reporting, see [Evaluation](benchmark/evaluation.md).

## 1. Set up the environment

Complete [Installation](install.md). This example uses the Python 3.10 HUMANO environment, native SONIC, and MuJoCo with Isaac Sim rendering. SONIC uses the CPU by default; policy inference and rendering have separate device settings.

Prepare compatible SONIC models, robot assets, and Isaac scene resources. Keep the HUMANO and model environments in separate terminals.

## 2. Prepare your model and data

Follow [Models and datasets](models.md) for dataset fields, image preprocessing, state/action order, units, and policy adapters. You need:
- `GroundBoxPick` evaluation data and its referenced scene configurations and caches.
  The example uses `data/eval/id/basic/G1WholebodyGroundBoxPickTeleop`.
- A task-compatible checkpoint for the default 7-DoF hands, with its processor,
  normalization statistics, and model configuration.
- An [adapter](../baselines/examples/policy_adapter.py) implementing `load_policy`,
  `predict`, and `reset`. It must return denormalized `(40, 78)` action chunks in
  the order and units defined in [Models and datasets](models.md).

These resources are separate from the code repository. Replace example paths with your matching dataset and assets. Evaluation data supplies frozen initial states; subsequent images and robot states come from the running simulation. Task instructions come from the episode metadata. The adapter applies the same preprocessing and normalization conventions used during training.

### Start the policy server

In a terminal, activate your model environment. Place `my_adapter.py` in your model directory and replace the paths below:

```bash
PYTHONPATH=/path/to/HUMANO/src:/path/to/my_model \
python -m humano.policy.server \
  --adapter my_adapter:load_policy --checkpoint /path/to/checkpoint \
  --host 127.0.0.1 --port 22085
```

The server requires Python 3.10 or later, NumPy, and your model's dependencies. Keep it running and check its status from another terminal:

```bash
curl http://127.0.0.1:22085/health
```

A successful health check confirms that the server is running; the next step tests inference and simulation together.

## 3. Run evaluation

In a second terminal, enter the HUMANO repository and activate its environment. Use a new output directory and check that the dataset matches the environment ID.

```bash
source .venv/bin/activate
HUMANO_EVAL_INSTRUCTION_SOURCE=data python -m humano.cli.eval_task \
  humano/G1WholebodyGroundBoxPickTeleop-v0 remote \
  --host 127.0.0.1 --port 22085 \
  --data-format lerobot --sim-mode mujoco_isaac \
  --data-dir data/eval/id/basic/G1WholebodyGroundBoxPickTeleop \
  --eval-dir runs/evals/my_policy/ground_box_pick \
  --num-episodes 1 --episode-start 0 --num-workers 1 \
  --headless --no-save-video --seed 42
```

This connects to the existing server without tmux, uses the episode instruction from the dataset, and automatically applies the v1.2 budget for this Basic task. Video recording is off. Each stateful policy server must serve only one evaluation session at a time.

### Check the results

Open `eval_metrics.jsonl` and the logs in the output directory. Check that episode steps, success, and termination reason are recorded, and that the process completed without resource, server, or initialization errors. A task timeout can be a valid failed episode.

Follow the adapter checks in [Models and datasets](models.md) and the reporting rules in [Evaluation](benchmark/evaluation.md). Record the code commit, dataset version, checkpoint, task, episodes, seed, and budget. For import, inference, hand mapping, or cache errors, see the model guide's troubleshooting section; for runtime dependencies, see [Installation](install.md).
