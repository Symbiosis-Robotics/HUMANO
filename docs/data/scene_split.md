# Scene splits and evaluation bindings

The eight suites use iTHOR kitchens, living rooms and bedrooms. The FloorPlan split assigns 60 physical FloorPlans to the training pool and 30 to the evaluation pool. Individual tasks use a subset of rooms. ID episodes can use rooms from the training pool with different placements.

To evaluate a task, extract its ModelScope ZIP under `data/`. Select `data/eval/id/<suite>/<task>/` for ID or `data/eval/ood/<dimension>/<suite>/<task>/` for an OOD condition. Each episode supplies the configuration for its scene and placement. Keep the accompanying `data/cache/bindings/eval/<suite>/<task>/` files in their extracted locations. The simulator restores the saved binding on reset.

See the [task archive instructions](../../data/README.md) and [evaluation guide](../benchmark/evaluation.md).
