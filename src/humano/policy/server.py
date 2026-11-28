"""Standalone HUMANO policy service; only NumPy and the Python standard library.

Expose predict(observation) and reset() in your model environment. No simulation
or model framework is imported here. The wire format remains compatible with
HUMANO's existing HTTP clients.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import threading
from base64 import b64decode, b64encode
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

import numpy as np
from numpy.lib.format import descr_to_dtype, dtype_to_descr

LOGGER = logging.getLogger(__name__)
ACTION_HORIZON = 40
ACTION_DIM = 78
IMAGE_SHAPE = (360, 640, 3)
STATE_PARTS = (
    ("left_leg", 6),
    ("right_leg", 6),
    ("waist", 3),
    ("left_arm", 7),
    ("left_hand", 7),
    ("right_arm", 7),
    ("right_hand", 7),
)


def state_parts(hand_type="dex3"):
    # BrainCo native order: thumb proximal/metacarpal, index, middle, ring, pinky.
    if hand_type not in {"dex3", "brainco"}:
        raise ValueError(f"Unsupported policy hand type: {hand_type}")
    dof = 6 if hand_type == "brainco" else 7
    return tuple((name, dof if name.endswith("_hand") else size) for name, size in STATE_PARTS)


def action_dim(hand_type="dex3"):
    return 64 + 2 * dict(state_parts(hand_type))["left_hand"]


def numpy_serialize(o):
    if isinstance(o, np.ndarray | np.generic):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": o.shape,
        }

    msg = f"Object of type {o.__class__.__name__} is not JSON serializable"
    raise TypeError(msg)


def numpy_deserialize(dct):
    if "__numpy__" in dct:
        np_obj = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return np_obj.reshape(shape) if (shape := dct["shape"]) else np_obj[0]
    return dct


def convert_numpy_in_dict(data, func):
    """
    Recursively processes a JSON-like dictionary, converting any NumPy arrays
    or lists of NumPy arrays into a serializable format using the provided function.

    Args:
        data: The JSON-like dictionary or object to process.
        func: A function to apply to each NumPy array to make it serializable.

    Returns:
        The processed dictionary or object with all NumPy arrays converted.
    """
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {key: convert_numpy_in_dict(value, func) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_in_dict(item, func) for item in data]
    elif isinstance(data, np.ndarray | np.generic):
        return func(data)
    else:
        return data


class Message:
    def __init__(self):
        pass

    def serialize(self):
        raise NotImplementedError

    @classmethod
    def deserialize(cls, response: dict[str, Any]):
        raise NotImplementedError


class RequestMessage(Message):
    def __init__(
        self,
        image: dict[str, Any],
        instruction: str,
        history: dict[str, Any],
        state: dict[str, Any],
        condition: dict[str, Any],
        gt_action: np.ndarray | list,
        dataset_name: str,
        timestamp: str,
    ):
        (
            self.image,
            self.instruction,
            self.history,
            self.state,
            self.gt_action,
            self.dataset_name,
            self.timestamp,
        ) = image, instruction, history, state, gt_action, dataset_name, timestamp
        self.condition = condition

    def serialize(self):
        msg = {
            "image": self.image,
            "instruction": self.instruction,
            "history": self.history,
            "state": self.state,
            "condition": self.condition,
            "gt_action": self.gt_action,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
        }
        return convert_numpy_in_dict(msg, numpy_serialize)

    @classmethod
    def deserialize(cls, response: dict[str, Any]):
        response = convert_numpy_in_dict(response, numpy_deserialize)
        return cls(
            image=response["image"],
            instruction=response["instruction"],
            history=response["history"],
            state=response["state"],
            condition=response["condition"],
            gt_action=response["gt_action"],
            dataset_name=response["dataset_name"],
            timestamp=response["timestamp"],
        )


class ResponseMessage(Message):
    def __init__(self, action: np.ndarray, err: float, traj_image: np.ndarray = None):
        self.action = action
        self.err = err
        self.traj_image = (
            traj_image if traj_image is not None else np.zeros((1, 1, 3), dtype=np.uint8)
        )

    def serialize(self):
        msg = {
            "action": self.action,
            "err": self.err,
            "traj_image": self.traj_image,
        }
        return convert_numpy_in_dict(msg, numpy_serialize)

    @classmethod
    def deserialize(cls, response: dict[str, Any]):
        response = convert_numpy_in_dict(response, numpy_deserialize)
        err = response["err"] if "err" in response else 0.0
        traj_image = response["traj_image"] if "traj_image" in response else None
        if isinstance(err, str):
            LOGGER.warning("Server error: %s", err)
        return cls(action=response["action"], err=err, traj_image=traj_image)


class Policy(Protocol):
    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        """Return unnormalized (40, 78) SONIC tokens and hand targets."""
        ...

    def reset(self) -> None:
        """Clear per-episode model memory and cached actions."""
        ...


def validate_action_chunk(action: Any, hand_type: str = "dex3") -> np.ndarray:
    result = np.asarray(action, dtype=np.float32)
    # (40, 78) unitless SONIC tokens followed by hand targets in radians
    expected = (ACTION_HORIZON, action_dim(hand_type))
    if result.shape != expected or not np.isfinite(result).all():
        raise ValueError(f"action must be finite with shape {expected}, got {result.shape}")
    return result


def validate_amo_chunk(action: Any, hand_dof: int) -> np.ndarray:
    return validate_controller_chunk(action, hand_dof, "amo")


def validate_controller_chunk(action: Any, hand_dof: int, controller: str) -> np.ndarray:
    chunk = np.atleast_2d(np.asarray(action, dtype=np.float32))
    width = {"amo": 22, "twist2": 35}[controller]
    widths = (width, width + 2 * hand_dof)
    if (
        chunk.ndim != 2
        or not len(chunk)
        or chunk.shape[1] not in widths
        or not np.isfinite(chunk).all()
    ):
        raise ValueError(
            f"Expected finite {controller.upper()} action rows of width {widths}, got {chunk.shape}"
        )
    return chunk


def policy_observation(payload: dict[str, Any], hand_type: str = "dex3") -> dict[str, Any]:
    request = RequestMessage.deserialize(payload)
    for name, size in (*state_parts(hand_type), ("projected_gravity", 3)):
        value = np.asarray(request.state[name])
        # (part_dim,) joint radians or unitless gravity, dataset/body-frame order
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"state.{name} must be finite with shape ({size},)")
    image = np.asarray(request.image["ego_view"])
    # (360, 640, 3) RGB in source scale, ego-camera frame
    if image.shape != IMAGE_SHAPE or not np.isfinite(image).all():
        raise ValueError(f"image.ego_view must be finite with shape {IMAGE_SHAPE}")
    if not isinstance(request.instruction, str):
        raise TypeError("instruction must be a string")
    observation = {
        "image": request.image,
        "state": request.state,
        "instruction": request.instruction,
    }
    if "inference_seed" in payload:
        observation["inference_seed"] = int(payload["inference_seed"])
    return observation


class StaleSessionRequest(RuntimeError):
    pass


class PolicyServer:
    """One policy/session per service process, as in the existing eval runner."""

    def __init__(self, policy: Policy, *, name: str = "external", hand_type: str = "dex3"):
        self.hand_type = hand_type
        state_parts(hand_type)
        self.adapter = policy
        self.name = name
        self._runtime_lock = threading.Lock()
        self._session_generation = 0

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "baseline": self.name,
            "action_shape": [ACTION_HORIZON, action_dim(self.hand_type)],
        }

    def reset(self) -> dict[str, Any]:
        with self._runtime_lock:
            self.adapter.reset()
            self._session_generation = 0
        return {"status": "ok", "baseline": self.name}

    def predict_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        observation = policy_observation(payload, self.hand_type)
        history = payload["history"]
        generation = history.get("session_generation", 0)
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("session_generation must be a non-negative integer")
        reset_requested = history.get("reset") is True
        with self._runtime_lock:
            if generation < self._session_generation:
                raise StaleSessionRequest(
                    f"request belongs to older session {generation}; "
                    f"current session is {self._session_generation}"
                )
            if generation > self._session_generation and not reset_requested:
                raise ValueError("a new session_generation requires reset=true")
            if reset_requested:
                self.adapter.reset()
            self._session_generation = generation
            prediction = self.adapter.predict(observation)
            if payload.get("dataset_name") in ("amo", "twist2"):
                action = validate_controller_chunk(
                    prediction,
                    dict(state_parts(self.hand_type))["left_hand"],
                    payload["dataset_name"],
                )
            else:
                action = validate_action_chunk(prediction, self.hand_type)
        return ResponseMessage(action=action, err=0.0).serialize()


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "HumanoPolicy/1.0"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json(404, {"error": "not found"})
            return
        self._send_json(200, self.server.policy_server.health())

    def do_POST(self) -> None:
        if self.path not in {"/act", "/reset"}:
            self._send_json(404, {"error": "not found"})
            return
        try:
            if self.path == "/reset":
                response = self.server.policy_server.reset()
            else:
                size = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(size))
                response = self.server.policy_server.predict_action(payload)
        except StaleSessionRequest as exc:
            self._send_json(409, {"error": str(exc)})
            return
        except (KeyError, TypeError, ValueError) as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except (RuntimeError, AssertionError, IndexError, OSError) as exc:
            LOGGER.exception("Policy inference failed")
            self._send_json(500, {"error": str(exc)})
            return
        self._send_json(200, response)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)


def serve_server(policy_server: Any, *, host: str, port: int) -> None:
    """Shared HTTP transport, also used by existing specialized model servers."""
    with ThreadingHTTPServer((host, port), RequestHandler) as httpd:
        httpd.policy_server = policy_server
        LOGGER.info("Serving policy on http://%s:%s", host, httpd.server_port)
        httpd.serve_forever()


def serve(policy: Policy, *, host: str = "127.0.0.1", port: int = 22085) -> None:
    serve_server(PolicyServer(policy), host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="Import path module:load_policy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=22085)
    args = parser.parse_args()
    module_name, factory_name = args.adapter.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    logging.basicConfig(level=logging.INFO)
    serve(factory(args.checkpoint), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
