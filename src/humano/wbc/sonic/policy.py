"""
ONNX Runtime inference wrapper for the SONIC decoder (policy) model.

SONIC defaults to CPU with automatic intra-op threading; GPU is selectable.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import onnxruntime as ort

TENSORRT_CACHE_DIR = Path(__file__).resolve().parents[4] / ".cache/sonic/tensorrt"


def _preload_cuda_12_libraries() -> None:
    """Expose CUDA 12 libraries installed with PyTorch to ONNX Runtime."""
    nvidia_root = Path(ort.__file__).resolve().parent.parent / "nvidia"
    for package, library in (
        ("cublas", "libcublasLt.so.12"),
        ("cublas", "libcublas.so.12"),
        ("cufft", "libcufft.so.11"),
        ("curand", "libcurand.so.10"),
    ):
        library_path = nvidia_root / package / "lib" / library
        if library_path.is_file():
            ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)


class SonicPolicy:
    """Loads the decoder ONNX model and runs ``obs → action[29]`` inference."""

    def __init__(self, model_path: str | Path, provider: str = "cpu") -> None:
        self._model_path = Path(model_path)
        if provider not in ("cpu", "gpu"):
            raise ValueError(f"Unknown SONIC provider: {provider}")
        self._provider = provider
        self._session: ort.InferenceSession | None = None
        self._input_name: str = ""
        self._output_name: str = ""
        self.input_dim: int = 0
        self.action_dim: int = 29
        self._load()
        # BEGIN NATIVE_LATENCY_DEBUG (temporary; removal: docs/dev/data.md)
        print(
            f"[NATIVE_LATENCY_DEBUG][decoder] providers={self._session.get_providers()}", flush=True
        )
        # END NATIVE_LATENCY_DEBUG

    # ------------------------------------------------------------------
    # Internal load / reload
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Create a fresh ONNX session from ``self._model_path``."""
        if not self._model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self._model_path}")

        # Close previous session if any.
        if self._session is not None:
            del self._session

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 0
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        if self._provider == "cpu":
            self._session = ort.InferenceSession(
                str(self._model_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            if not inputs or not outputs:
                raise RuntimeError("ONNX model has no inputs or outputs")
            self._input_name = inputs[0].name
            self._output_name = outputs[0].name
            self.input_dim = inputs[0].shape[-1]
            self.action_dim = outputs[0].shape[-1]
            return

        required_providers = (
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
        )
        available_providers = set(ort.get_available_providers())
        missing_providers = [
            provider for provider in required_providers if provider not in available_providers
        ]
        if missing_providers:
            raise RuntimeError(
                "SONIC native GPU inference requires "
                f"{', '.join(required_providers)}; available providers: "
                f"{', '.join(sorted(available_providers)) or 'none'}"
            )

        _preload_cuda_12_libraries()
        TENSORRT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        self._session = ort.InferenceSession(
            str(self._model_path),
            sess_options=opts,
            providers=[
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": 0,
                        "trt_fp16_enable": False,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(TENSORRT_CACHE_DIR),
                        "trt_timing_cache_enable": True,
                        "trt_timing_cache_path": str(TENSORRT_CACHE_DIR),
                    },
                ),
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ],
        )
        active_providers = set(self._session.get_providers())
        missing_active_providers = [
            provider for provider in required_providers if provider not in active_providers
        ]
        if missing_active_providers:
            self._session = None
            raise RuntimeError(
                "SONIC native GPU inference created a session without "
                f"{' and '.join(missing_active_providers)}; active providers: "
                f"{', '.join(sorted(active_providers)) or 'none'}"
            )

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError("ONNX model has no inputs or outputs")

        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

        input_shape = inputs[0].shape
        self.input_dim = input_shape[-1] if input_shape else 0
        if self.input_dim <= 0:
            raise RuntimeError(f"Could not determine input dimension from shape {input_shape}")

        output_shape = outputs[0].shape
        self.action_dim = output_shape[-1] if output_shape else 29

    def reload(self) -> None:
        """Close the current ONNX session and load the model from disk again.

        Useful for reset / experiment restart scenarios where the ONNX file
        may have been updated in-place.
        """
        self._load()

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """Run one forward pass.

        Parameters
        ----------
        obs:
            Observation vector, shape ``(input_dim,)`` or ``(1, input_dim)``,
            dtype float32.

        Returns
        -------
        np.ndarray
            Action vector, shape ``(action_dim,)``, dtype float64 (isaaclab order).
        """
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        ort_inputs = {self._input_name: obs.astype(np.float32, copy=False)}
        result = self._session.run([self._output_name], ort_inputs)
        return result[0].flatten().astype(np.float64, copy=False)
