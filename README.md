<p align="center">
  <img src="resources/branding/logo-full.png" width="300" alt="HUMANO">
</p>

# HUMANO

**A Large-Scale Benchmark for Generalist Humanoid Loco-Manipulation**

[Project page](https://symbiosis-robotics.github.io/HUMANO/) · [Dataset](https://www.modelscope.cn/datasets/SymbiosisRobotics/HUMANO) · [Documentation](docs/README.md) · [Task catalog](docs/benchmark/tasks.md) · [Citation](#citation)

HUMANO evaluates how a Unitree G1 coordinates locomotion and manipulation across 100 tasks in eight suites. Use the v1 demonstrations to train a policy, then run its checkpoint on fixed evaluation episodes. HUMANO-Plus tests generalization across seven distribution shifts.

## Using HUMANO


| Goal                                  | Guide                                                                                                                                      |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Set up the environment                | [Install HUMANO, SONIC and simulation assets](docs/install.md)                                                                             |
| Download training and evaluation data | [Download and extract a task](data/README.md#download-and-extract-a-task) and [v1 observations and action targets](docs/data/v1-fields.md) |
| Train a baseline                      | [Fine-tune GR00T N1.7 on Box Lifting](docs/gr00t.md#train-a-policy)                                                                        |
| Run a checkpoint                      | [Run GR00T inference](docs/gr00t.md#run-inference) or [connect your own policy](docs/quickstart.md)                                        |
| Benchmark a policy                    | [Evaluate across tasks and report ID and HUMANO-Plus results](docs/benchmark/evaluation.md)                                                |

## Citation

The BibTeX entry will accompany the public paper release.

## License and credits

HUMANO code is distributed under the [MIT License](license.md). Portions derive from SIMPLE. See the [third-party notices](docs/third-party-notices.md) and the terms supplied with the datasets, assets and model weights.
