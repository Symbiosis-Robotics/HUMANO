# Psi0 Runtime

This directory contains HUMANO's inference-only snapshot of the separately maintained
Psi0 project. It deliberately excludes Psi0 training code, datasets, and model weights.

- Upstream repository: `git@github.com:Kinetra-Humanoids/Psi0.git`
- Integrated upstream commit: `fdff2aa25cb1dc796fd157a8e167d201a307b819`
- Runtime package: `psi`, vendored under this directory and loaded by HUMANO's interpreter
- Model contract: 46D state, 40-step 78D action

`baselines/psi0/deploy/psi0_serve.py` owns HUMANO request/session handling, checkpoint
configuration, normalization, model invocation, denormalization, and the fixed
`(40, 78)` response contract. The embedded `psi` package contains only the model
implementation and minimal utilities required for inference. See `LICENSE` for the
upstream license.
