# LeRobot runtime provenance

- Track B repository: `../baselines/lerobot` relative to the HUMANO checkout
- Upstream tag: `v0.6.0`
- Upstream commit: `30da8e687a6dfc617fcd94afc367ac7071c376ce`
- Runtime observation: 43 joint values followed by 3 projected-gravity values
- Runtime action: 40 by 78 SONIC-token and hand-action chunk

The checked-in `lerobot` package is the frozen source used by HUMANO Track B
evaluation. The HUMANO `.venv` supplies third-party dependencies, but policy,
processor, and configuration code resolves only from this directory.

LeRobot v0.6.0 requires Python 3.12 upstream. HUMANO evaluation uses Python
3.10, so the vendored source contains syntax-only compatibility edits for
`typing.Self`, `typing.Unpack`, PEP 695 generics, and type aliases. These edits
do not change model or processor behavior.
