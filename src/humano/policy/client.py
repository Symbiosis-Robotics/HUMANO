"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from datetime import datetime
from typing import Any

import numpy as np
import requests

from humano.policy.server import (
    Message,
    RequestMessage,
    ResponseMessage,
    convert_numpy_in_dict,
    numpy_deserialize,
    numpy_serialize,
)


class HttpActionClient:
    def __init__(self, server_ip: str, server_port: int, timeout_s: float = 60.0):
        self.server_ip, self.server_port = server_ip, server_port
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")

    @property
    def timestamp(self):
        return str(datetime.now()).replace(" ", "_").replace(":", "-")

    def reset_session(self) -> None:
        """Start a new sequential evaluation on an already-running server."""
        url = f"http://{self.server_ip}:{self.server_port}/reset"
        try:
            response = requests.post(url, timeout=self.timeout_s)
            response.raise_for_status()
        except requests.Timeout as exc:
            raise RuntimeError(
                f"Policy session reset at {url} timed out after {self.timeout_s:.1f}s"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Cannot reset policy session at {url}; check the server supports POST /reset "
                "and no other evaluation is using it"
            ) from exc

    def query_action(
        self,
        image_dict: dict[str, Any],
        instruction: str,
        state_dict: dict[str, Any],
        condition_dict: dict[str, Any],
        history: dict[str, Any] | None = None,
        dataset="grasp",
        gt_action: np.ndarray | None = None,
        inference_seed: int | None = None,
    ) -> np.ndarray:
        if history is None:
            history = {k: [] for k in image_dict.keys()}
        if gt_action is None:
            gt_action = []

        request = RequestMessage(
            image_dict,
            instruction,
            history,
            state_dict,
            condition_dict,
            gt_action,
            dataset,
            self.timestamp,
        )
        payload = request.serialize()
        if inference_seed is not None:
            payload["inference_seed"] = int(inference_seed)

        try:
            response = requests.post(
                f"http://{self.server_ip}:{self.server_port}/act",
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.Timeout as exc:
            raise RuntimeError(f"Policy request timed out after {self.timeout_s:.1f}s") from exc
        except requests.RequestException as exc:
            raise RuntimeError("Policy server is unavailable") from exc

        if response.status_code != 200:
            raise Exception(f"Server is not up ?! {response.status_code}: {response.text}")
        try:
            response = ResponseMessage.deserialize(response.json())
        except Exception as e:
            raise RuntimeError(response.text) from e

        # Validate that the server returned a valid trajectory image
        if not isinstance(response.traj_image, np.ndarray) or response.traj_image.ndim != 3:
            print("[WARN] Server did not return a valid trajectory image (traj_image).")
            return response.action, response.err, None

        return response.action, response.err, response.traj_image
