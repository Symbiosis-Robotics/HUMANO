# HUMANO Documentation

HUMANO supports dataset processing and policy evaluation for the G1 humanoid. MuJoCo provides physics, Isaac Sim renders scenes and cameras, and SONIC/WBC controls the robot. Train policies in their own environments and evaluate them through HUMANO's policy interface.

## Quick Start

Follow the [three-step walkthrough](quickstart.md) to evaluate your own policy:

1. **Set up the environment:** [install HUMANO](install.md), SONIC, and scene assets.
2. **Prepare your model and data:** use the [model and dataset guide](models.md)
   to match fields, implement an adapter, and start your inference service.
3. **Run evaluation:** complete a single-task run, then use the
   [evaluation guide](benchmark/evaluation.md) for suite runs and reporting.

Datasets, scene assets, and model weights are separate from the code repository. Use matching resource versions; example paths are local paths, not download links.

## Guides

| Goal | Guide |
| --- | --- |
| Install dependencies | [Installation](install.md) |
| Complete the three-step setup and first run | [Quick Start](quickstart.md) |
| Download and extract a training and evaluation task | [Task archives](../data/README.md#download-and-extract-a-task) |
| Browse tasks and suites | [Task catalog](benchmark/tasks.md) |
| Use datasets or connect a policy | [Models and datasets](models.md) |
| Set task instructions | [Language fields](data/language-fields.md) |
| Understand scene splits | [Scene splits](data/scene_split.md) |
| Evaluate a policy | [Evaluation](benchmark/evaluation.md) |

For a custom policy, start with the [adapter template](../baselines/examples/policy_adapter.py) and [examples](../baselines/examples/README.md). See [Models and datasets](models.md) for observation and action formats, session reset, and troubleshooting.

## Reproducibility and licenses

Record the code commit, dataset version, checkpoint, and evaluation settings. Compare policies on the same tasks, initial states, seeds and budgets. Report per-task successes and episode counts.

See the [code license](../license.md). Datasets, scene assets, and third-party models retain their own license and citation requirements.
