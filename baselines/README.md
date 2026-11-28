# Baselines

## Attribution

`gr00t_n17/gr00t/` contains code from [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T),
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, licensed under Apache 2.0.
See `gr00t_n17/LICENSE` for the full license text.

HUMANO-specific GR00T deployment wrappers live under `baselines/gr00t_n17/deploy/`
and are not part of the original NVIDIA distribution.

`lerobot_runtime/` contains the HUMANO-local inference contract shared by ACT,
Diffusion, Pi0.5, and Flow Matching. The package retains the upstream
`lerobot` import name under `lerobot_runtime/lerobot/`; see
`lerobot_runtime/UPSTREAM.md` and `lerobot_runtime/LICENSE`.

`psi0/` contains the HUMANO-local Psi0 inference closure. It is launched by
`baselines.psi0.deploy.psi0_serve` with the HUMANO interpreter; see `psi0/UPSTREAM.md`
and `psi0/LICENSE`.

## Policy integration and environments

External models keep their original repository and Python environment. Implement
`load_policy(checkpoint)`, `predict(observation)` and `reset()` using
[policy_adapter.py](examples/policy_adapter.py), then run the shared
`humano.policy.server` in that environment. It requires Python 3.10 or newer, NumPy and the Python
standard library; installing HUMANO's simulation dependencies is unnecessary.

HUMANO evaluates through the shared SONIC-token Agent using
`--external-server --policy remote`. Model-specific preprocessing and output
conversion live in the model adapter, not a new evaluation Agent.
See [the integration guide](../docs/models.md) for both launch commands, field
and action ordering, and a local LeRobot batch example.
