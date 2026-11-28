# Fine-tune and evaluate GR00T N1.7

## Train a policy

The example below fine-tunes **GR00T N1.7** on **Box Lifting** (`B-8`). The same task identifier is used for training and evaluation, with separate episode sets.

1. Set up the [GR00T N1.7 training environment](https://github.com/NVIDIA/Isaac-GR00T) and obtain its base weights and Cosmos-Reason2-2B backbone. Run the commands below from the HUMANO repository root in that model environment.
2. Prepare a writable local copy of the v1 training task under `train/basic/G1WholebodyGroundBoxPickTeleop/`. It should contain `meta/`, `data/` and `videos/`. Set the paths below to your local packages. Model weights and benchmark data are distributed separately. [Dataset layout](../data/README.md)
3. Launch single-GPU fine-tuning with the bundled G1 modality configuration:

```bash
export HUMANO_DATA="$PWD/data"
export GR00T_BASE=/path/to/GR00T-N1.7-3B
export COSMOS_BACKBONE=/path/to/Cosmos-Reason2-2B

PYTHONPATH="$PWD/baselines/gr00t_n17" torchrun --standalone --nproc-per-node=1 \
  --module gr00t.experiment.launch_finetune \
  --base-model-path "$GR00T_BASE" --backbone-model-path "$COSMOS_BACKBONE" \
  --dataset-path "$HUMANO_DATA/train/basic/G1WholebodyGroundBoxPickTeleop" \
  --embodiment-tag UNITREE_G1_SONIC --num-gpus 1 \
  --output-dir "$PWD/runs/gr00t-box"
```

The policy sees the egocentric RGB image, task instruction, joint state and projected gravity. For the default seven-DoF hands, each training target contains 40 steps. Each step has 78 values: a 64-D SONIC motion token followed by seven left-hand and seven right-hand joint targets. For another trainer, start with the [training batch example](models.md#read-a-training-batch) and [v1 field definitions](data/v1-fields.md).

## Run inference

Start here if you already have a task-compatible GR00T checkpoint. Complete the [HUMANO simulator setup](install.md), including SONIC, robot assets and scene resources. From the repository root, point the launcher to your **saved checkpoint directory**, the Python executable from the GR00T environment, and the matching v1 evaluation data. Set `HUMANO_DATA` again if using a new terminal.

```bash
export GR00T_CHECKPOINT=/path/to/saved/checkpoint
export GR00T_PYTHON=/path/to/gr00t-env/bin/python

HUMANO_EVAL_INSTRUCTION_SOURCE=data .venv/bin/python scripts/run/launch_evaluation.py \
  --policy gr00t_n17 --server-python "$GR00T_PYTHON" --run-dir "$GR00T_CHECKPOINT" \
  --env-id humano/G1WholebodyGroundBoxPickTeleop-v0 \
  --data-dir "$HUMANO_DATA/eval/id/basic/G1WholebodyGroundBoxPickTeleop" \
  --eval-dir runs/evals/gr00t-box --num-episodes 1 --seed 42
```

The launcher starts the policy server and simulator. HUMANO restores a frozen initial state, sends live observations to the model, and executes its predictions. Inspect `eval_metrics.jsonl` in the evaluation directory for the episode outcome. For suite-level ID and HUMANO-Plus results, use the [evaluation protocol](benchmark/evaluation.md).

**Other policies.** Deployment entry points are available for ACT, Diffusion, Pi0.5, Flow Matching and Psi0. To use your own policy, implement `load_policy`, `predict` and `reset` in the supplied adapter. [Baseline launchers](../baselines/examples/README.md) · [Custom policy walkthrough](quickstart.md)
