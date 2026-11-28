"""Copy into your model repository and replace the three model-specific methods.

Run with your model's Python, adding HUMANO/src and your repository to PYTHONPATH:
python -m humano.policy.server --adapter policy_adapter:load_policy --checkpoint PATH
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MyPolicy:
    def __init__(self, checkpoint: str):
        raise NotImplementedError("Load your model and its training preprocessors here")

    def reset(self) -> None:
        """Clear model history/cached actions; use pass for a stateless model."""
        raise NotImplementedError("Reset your model's episode state here")

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        """Consume image/state/instruction; return denormalized float32 (40, 78).

        Image: observation['image']['ego_view'], RGB (360, 640, 3).
        State: observation['state'], named joint groups and projected_gravity.
        Action columns: 64 SONIC tokens, 7 left hand targets, 7 right hand targets.
        Reuse training preprocessing and undo action normalization in this method.
        Optional inference_seed is model-side reproducibility metadata.
        """
        raise NotImplementedError("Map inputs, call your model, and convert its actions here")


def load_policy(checkpoint: str) -> MyPolicy:
    return MyPolicy(checkpoint)
