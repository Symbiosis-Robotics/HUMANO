# HUMANO data layout

## Download and extract a task

Choose the suite first. A task name can occur in more than one suite with different layouts. Each of the 100 task ZIPs across eight suites contains 50 training demonstrations, 25 ID evaluation episodes, and 25 episodes for each of seven OOD conditions. The ZIP includes the frozen placement bindings for all 200 evaluation episodes.

Open the [HUMANO dataset on ModelScope](https://www.modelscope.cn/datasets/SymbiosisRobotics/HUMANO), open `data/`, choose a suite, and download the task ZIP. For example, Box Lifting is `data/basic/G1WholebodyGroundBoxPickTeleop.zip`. To download and extract it with the ModelScope CLI, run from the HUMANO repository root:

```bash
archive='data/basic/G1WholebodyGroundBoxPickTeleop.zip'
modelscope download --repo-type dataset SymbiosisRobotics/HUMANO "$archive" --local-dir downloads
unzip "downloads/$archive" -d data/
```

For a browser download, use the saved ZIP path in the same `unzip` command. Extract additional tasks into the same `data/` directory. For Box Lifting, the training data is now in `data/train/basic/G1WholebodyGroundBoxPickTeleop/` and the ID evaluation data is in `data/eval/id/basic/G1WholebodyGroundBoxPickTeleop/`. HUMANO reads each evaluation episode’s frozen placement from the extracted cache.

| Use | Path under the HUMANO checkout |
| --- | --- |
| Training demonstrations | `data/train/<suite>/<task>/` |
| ID evaluation | `data/eval/id/<suite>/<task>/` |
| OOD evaluation | `data/eval/ood/<dimension>/<suite>/<task>/` |
| Frozen evaluation bindings | `data/cache/bindings/eval/<suite>/<task>/binding_XX/` |

For training, point the LeRobot reader at the extracted task directory. It contains Parquet sequences, camera videos, instructions, schema and statistics. For evaluation, pass the ID or OOD task directory to `--data-dir`. Each episode supplies an initial state and a configuration that locates its bundled cache. Install the matching robot and scene resources for simulation. [Training inputs](../docs/models.md) · [Evaluation protocol](../docs/benchmark/evaluation.md)

## Simulation resources

For the ModelScope archive list, download commands and external dependencies, see [Install simulation resources](../docs/data/resources.md).

See the [training field reference](../docs/data/v1-fields.md) for the stored schema, model inputs and action targets.
