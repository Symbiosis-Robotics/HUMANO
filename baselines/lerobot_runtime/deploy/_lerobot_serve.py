from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from humano.policy.server import (
    action_dim,
    ACTION_HORIZON,
    PolicyServer,
    StaleSessionRequest,
    serve_server,
)
from humano.policy.server import (
    RequestHandler as _RequestHandler,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LOGGER = logging.getLogger(__name__)


def _runtime_root() -> Path:
    configured = os.environ.get("HUMANO_LEROBOT_RUNTIME_ROOT")
    if not configured:
        return REPO_ROOT / "baselines" / "lerobot_runtime"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


RUNTIME_ROOT = _runtime_root()
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from lerobot.runtime.contract import (  # noqa: E402
    BASELINE_SPECS,
    build_runtime_observation,
)
from lerobot.runtime.loader import load_runtime  # noqa: E402


class LerobotPolicyAdapter:
    def __init__(self, runtime):
        self.runtime = runtime

    def reset(self) -> None:
        self.runtime.reset()

    def predict(self, observation: dict[str, Any]):
        return self.runtime.predict(build_runtime_observation(observation, self.runtime.hand_type))


class LerobotPolicyServer(PolicyServer):
    def __init__(self, *, baseline: str, run_dir: str, device: str, mock_policy: bool = False, hand_type: str = "dex3"):
        try:
            self.spec = BASELINE_SPECS[baseline]
        except KeyError as exc:
            raise ValueError(f"Unsupported LeRobot baseline: {baseline!r}") from exc
        self.runtime = load_runtime(self.spec, run_dir, device, mock_policy=mock_policy, hand_type=hand_type)
        self.mock_policy = mock_policy
        super().__init__(LerobotPolicyAdapter(self.runtime), name=baseline, hand_type=hand_type)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "baseline": self.spec.name,
            "policy_type": self.spec.policy_type,
            "backend": "fixed" if self.mock_policy else "checkpoint",
            "action_shape": [ACTION_HORIZON, action_dim(self.hand_type)],
        }


def main(default_baseline: str) -> None:
    parser = argparse.ArgumentParser(description=f"Serve the HUMANO {default_baseline} baseline.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21000)
    parser.add_argument("--policy", default=default_baseline)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ckpt-step", default=None)
    parser.add_argument("--mock-policy", action="store_true")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--hand-type", choices=["dex3", "brainco"], default="dex3")
    args = parser.parse_args()
    if args.tokenizer_path:
        os.environ["HUMANO_PI05_TOKENIZER_PATH"] = args.tokenizer_path

    normalized_policy = args.policy.lower().replace("-", "_").replace(".", "")
    if normalized_policy != default_baseline:
        raise ValueError(
            f"{default_baseline}_serve only accepts --policy {default_baseline!r}, "
            f"got {args.policy!r}"
        )
    policy_server = LerobotPolicyServer(
        baseline=default_baseline,
        run_dir=args.run_dir,
        device=args.device,
        mock_policy=args.mock_policy,
        hand_type=args.hand_type,
    )
    serve_server(policy_server, host=args.host, port=args.port)
