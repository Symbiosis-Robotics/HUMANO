# Policy deployment examples

Both integration paths implement the same model-side adapter:
`load_policy(checkpoint)` returns an object with `predict(observation)` and `reset()`.
Copy [policy_adapter.py](policy_adapter.py) into your model repository and replace
the three model-specific methods. Reuse training preprocessing and return
denormalized float32 actions shaped (40, 78): one 64-dimensional SONIC motion token, then 7 targets per hand on each step.

## Built-in policies: HUMANO environment

Run from the HUMANO repository root. Model adapters live beside their algorithm;
shared communication and robot execution live in `src/humano/policy/`.

| Model | Deployment module |
| --- | --- |
| ACT | `baselines.lerobot_runtime.deploy.act_serve` |
| Diffusion | `baselines.lerobot_runtime.deploy.diffusion_serve` |
| Pi0.5 | `baselines.lerobot_runtime.deploy.pi05_serve` |
| Flow Matching | `baselines.lerobot_runtime.deploy.flow_matching_serve` |
| GR00T N1.7 | `baselines.gr00t_n17.deploy.gr00t_serve` |
| Psi0 | `baselines.psi0.deploy.psi0_serve` |

Use `--help` on the relevant module for its checkpoint and model-specific options:

```bash
PYTHONPATH="$PWD:$PWD/src" .venv/bin/python -m baselines.lerobot_runtime.deploy.act_serve --help
```

The [evaluation launcher](../../scripts/run/launch_evaluation.py) starts the service for the selected built-in policy.

## External policies: model environment

Keep your algorithm and dependencies in your own repository and Python environment.
Only NumPy and Python 3.10+ are required by the shared server; it does not import
simulation, robot SDKs or model frameworks.

```bash
PYTHONPATH=/path/to/HUMANO/src:/path/to/my_model python -m humano.policy.server   --adapter policy_adapter:load_policy --checkpoint /path/to/checkpoint   --host 127.0.0.1 --port 22085
```

Then connect from the HUMANO environment:

```bash
HUMANO_EVAL_INSTRUCTION_SOURCE=data .venv/bin/python scripts/run/launch_evaluation.py \
  --external-server --policy remote --host 127.0.0.1 --port 22085 \
  --env-id humano/G1WholebodyGroundBoxPickTeleop-v0 \
  --data-dir data/eval/id/basic/G1WholebodyGroundBoxPickTeleop \
  --eval-dir runs/evals/my_policy/box_lifting --num-episodes 1
```

See [the integration guide](../../docs/models.md) for observation fields,
action ordering, reset semantics and evaluation boundaries.
