# Evaluation

Complete the [quickstart](../quickstart.md) before running a benchmark. Select a data release, suite, condition, and episode list, then evaluate each checkpoint with the same initial states, policy seeds, and execution settings.

SONIC evaluation uses the native backend. Select `--sonic-provider cpu|gpu` (default: `cpu`) in `scripts/run/launch_evaluation.py`, or set `HUMANO_SONIC_PROVIDER=gpu` for direct `eval_task` runs. GPU mode requires TensorRT/CUDA providers and fails if they are unavailable. The policy model's device is configured separately. See [installation](../install.md) for dependencies.

## Suites and data releases

The [task catalog](tasks.md) defines 100 entries:

| Suite | Tasks |
| --- | ---: |
| Basic | 10 |
| Advanced | 10 |
| Spatial | 20 |
| Long | 20 |
| Goal | 10 |
| Condition | 10 |
| Object | 10 |
| Movement | 10 |
| **Total** | **100** |

Spatial and Movement reuse some Basic/Advanced environment IDs with different layouts or parameters. Keep them as separate suites. Each [task archive](../../data/README.md#download-and-extract-a-task) contains 50 training demonstrations, 25 ID evaluation episodes, and 25 evaluation episodes for each of seven OOD conditions: Background, Camera, Distractor, Language, Lighting, Robot Initial, and Sensor Noise. Report results separately for ID and each OOD condition. Each evaluation episode's `environment_config` stores its frozen scene and perturbations. Policy seeds do not resample these states.

## Success rate

Record one binary outcome per episode. An episode succeeds when it satisfies the task's success condition without a recorded fall or execution-ending collision. Reaching the step limit before success counts as failure. Within each suite and condition, compute the success rate for each task and average the task rates equally.

Single-task runs write `eval_metrics.jsonl` and `eval_stats.txt`. Suite runs write `summary.json` and per-task `summary.csv`. Check `evaluation_health.json` for missing episodes and failed tasks before comparing results.

## Time budgets

Evaluation time limits are set with reference to the duration of each task's training demonstrations. Each suite specifies its own limit for each task. ID and all seven OOD conditions use the same limit for a given suite and task.

The suite configuration lists each limit in steps at 200 Hz (`max_steps_200hz`). For example, Basic TableApplePick allows 5,640 steps (28.2 seconds of simulated time). If the environment uses a different step duration, the evaluator converts the time limit to environment steps and rounds up. Counting starts when the policy takes control.

The examples below load the budget automatically from `configs/suites/<suite>.json`. A single-task run uses the suite directory in the extracted dataset path and the task's environment ID. A suite run looks up each task in the selected suite configuration. An explicit `--max-episode-steps` overrides this limit in environment steps.

## Run one suite or one task

From the repository root, set the checkpoint and evaluation condition once. The examples use ACT and the ID split. For OOD, set `CONDITION` to the extracted condition, for example `ood/lighting`. Extract the selected task archives under `data/` first.

```bash
export RUN_DIR=/path/to/act-checkpoint
export POLICY=act HUMANO_EVAL_INSTRUCTION_SOURCE=data
CONDITION=id
export HUMANO_EVAL_ROOT="$PWD/data/eval/$CONDITION"
```

**Whole suite.** Change `SUITE` and run this block. Valid names are `basic`, `advanced`, `spatial`, `long`, `goal`, `condition`, `object` and `movement`. The script starts the policy service and evaluates 25 episodes per task.

```bash
SUITE=basic
bash scripts/run/evaluate_suite.sh "$SUITE" --num-episodes 25 \
  --eval-dir "runs/evals/$POLICY/$CONDITION/$SUITE/full"
```

**Single task.** Set `SUITE` and copy a task's environment name from the [catalog](tasks.md), omitting the `-v0` suffix. The same name selects the environment and the extracted data directory.

```bash
SUITE=basic
TASK=G1WholebodyGroundBoxPickTeleop
.venv/bin/python scripts/run/launch_evaluation.py --policy "$POLICY" --run-dir "$RUN_DIR" \
  --env-id "humano/${TASK}-v0" --data-dir "$HUMANO_EVAL_ROOT/$SUITE/$TASK" \
  --num-episodes 25 \
  --eval-dir "runs/evals/$POLICY/$CONDITION/$SUITE/$TASK"
```

Use a fresh output directory for each checkpoint and repeat. Append `--dry-run` to either command to inspect the launch commands.

## Frozen scene bindings

Each evaluation episode refers to a saved scene and placement in the task archive. Keep the extracted bindings under `data/cache/bindings/eval/<suite>/<task>/` so that all policies start from the same configuration.

## Report results

Record the code commit, dataset version, checkpoint, training-data scope, suite/condition, episode list, policy seeds, and repeat count. For each task, report successful episodes, returned results, success rate, planned episodes, and missing or failed runs. Do not silently omit missing episodes from the report.

Record the suite average, environment-step limit and execution mode. Distinguish catalog size, dataset metadata coverage, and completed policy evaluations.

## Single-task evaluation

Follow the [quickstart](../quickstart.md) to run one released task with your policy. In `scripts/run/launch_evaluation.py`, add `--fast` to skip unnecessary Isaac rendering steps during native SONIC evaluation. The launcher's default is off. Select `--parallel-jobs` independently to evaluate multiple episodes concurrently.

### Language inputs

Set `HUMANO_EVAL_INSTRUCTION_SOURCE=data` in the launching shell to read the episode instruction for ID and non-Language OOD runs. The training, ID, and non-Language OOD splits use direct action instructions. Language OOD uses equivalent paraphrases. Class-level `description` is not the formal model input. See [language fields](../data/language-fields.md) for field selection, conditional instructions, and overrides.

## Connect a policy service

Run the public server in the model's own environment. Follow the [model guide](../models.md) for adapters and input/output contracts; HUMANO does not need the model source or checkpoint when connecting to an external service.

```bash
HUMANO_EVAL_INSTRUCTION_SOURCE=data python scripts/run/launch_evaluation.py \
  --external-server --policy remote --host 127.0.0.1 --port 22085 \
  --env-id humano/G1WholebodyGroundBoxPickTeleop-v0 \
  --data-dir data/eval/id/basic/G1WholebodyGroundBoxPickTeleop --eval-dir runs/evals/box_pick
```

For the Basic ID suite, replace the single-task arguments `--env-id` and `--data-dir` with `--suite-config configs/suites/basic.json --data-root data/eval/id`. `--external-server` leaves service lifecycle management to the caller and requires neither `--run-dir` nor `--server-python`. Use the service's actual positive port. Native SONIC remains the default; built-in model auto-launch commands still work.

Give each evaluation exclusive access to its stateful model service. Episode reset calls the model's `reset()`. Keep action execution length, SONIC timing, and handoff mode fixed across comparisons. A returned 40-step chunk need not be fully executed before the next request.

## Parallel evaluation

Add `--parallel-jobs N` to a launcher command. The default 1 keeps the single-lane tmux workflow. Larger values run N independent local policy services and evaluators under the foreground scheduler, with `--num-workers 1` in each lane. Existing tmux sessions are not modified. Use your terminal's process management for background runs.

For 25 total episodes across two lanes:

```bash
MUJOCO_GL=egl HUMANO_EVAL_INSTRUCTION_SOURCE=data .venv/bin/python scripts/run/launch_evaluation.py \
  --policy act --run-dir /path/to/checkpoint \
  --env-id humano/G1WholebodyGroundBoxPickTeleop-v0 \
  --data-dir data/eval/id/basic/G1WholebodyGroundBoxPickTeleop \
  --eval-dir runs/evals/act_parallel --num-episodes 25 --episode-start 0 \
  --seed 42 --parallel-jobs 2 --no-save-video
```

Add `--dry-run` to inspect shards and commands without starting processes or creating output directories. Lanes receive balanced, non-overlapping continuous episode ranges, preserving original IDs and seeds. Lane count cannot exceed the episode count. Each lane owns one service; shared `--external-server` and remote machine scheduling are unsupported. Parallel evaluation requires native SONIC.

Select a GPU for a single job with `--gpu-id 1`. For parallel jobs, use
`--parallel-jobs 2 --gpu-ids 0,1` for one GPU per job, or `--gpu-ids 0,0`
to share GPU 0. Omitting `--parallel-jobs` uses the length of `--gpu-ids`. The order matches the episode shards in `parallel_manifest.json`.
Each pinned process receives its own CUDA visibility mask. The policy's selected
GPU is local `cuda:0`. GPU IDs use host CUDA indices in PCI bus order. If
Isaac's `gpu.foundation` table enumerates graphics devices differently, set
`HUMANO_ISAAC_GPU_MAP` to their Kit indices in CUDA PCI order (e.g., `1,0`).
Without GPU IDs, the existing `--device` option (default `cuda:0`) is passed to
every managed model service. An external policy service chooses its own GPU.

For a custom managed policy, pass `--policy remote --server-module MODULE` and
`--run-dir PATH`. The module is launched once per job with the policy server
arguments and must implement HUMANO's HTTP health, action and reset interface.

With `--suite-config`/`--data-root`, each lane follows suite order and evaluates its assigned range for every task. `--num-episodes` is the total **per task**, not per lane. There is no dynamic task queue.

The output root contains `parallel_manifest.json` with shards, ports, commands, status, and result paths. Each `job_000`-style directory contains task `eval_metrics.jsonl` or suite `summary.json`. The manifest does not calculate a new aggregate success rate. Merge shards by task and original episode ID, checking missing/duplicate episodes and all lane statuses. Use a fresh output directory.

The scheduler waits for each service to become healthy, then launches evaluation. On failure or interruption, it cleans up only processes it created. Datasets are not copied or sliced, so scene bindings and cache references stay unchanged. Parallelism does not alter `--fast`, video, time budgets, or success criteria. Validate a complete single-lane run before measuring hardware capacity.
