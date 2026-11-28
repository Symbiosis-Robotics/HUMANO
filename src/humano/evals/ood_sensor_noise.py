"""Episode-configured RGB input corruptions; physics and depth stay untouched."""

from __future__ import annotations

import numpy as np

LIBERO_PLUS_BLUR_SIGMAS = tuple(range(1, 11))
LIBERO_PLUS_SOURCE = (
    "https://github.com/sylvestf/LIBERO-plus/blob/"
    "4976dc30028e805ff8094b55501d532c48fec182/libero/libero/envs/env_wrapper.py"
)


def gaussian_blur_rgb(image: np.ndarray, sigma: float) -> np.ndarray:
    """Match Plus gaussian(image/255, channel_axis=-1), then its uint8 cast.

    skimage.filters.gaussian delegates to this same SciPy separable filter.
    Explicit nearest padding and truncate=4 retain its defaults; sigma=0 on
    the channel axis prevents mixing RGB. Use float64, clip, then truncate
    toward zero exactly as ControlEnv.step/reset, rather than rounding.
    """
    from scipy.ndimage import gaussian_filter

    blurred = gaussian_filter(image / 255.0, sigma=(sigma, sigma, 0), mode="nearest", truncate=4.0)
    return (np.clip(blurred, 0, 1) * 255).astype(np.uint8)


class SensorNoise:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.kind = self.config.get("subtype")
        if self.kind not in (None, "gaussian", "blur"):
            raise ValueError(f"Unknown sensor noise: {self.kind}")
        self.sigma = float(self.config.get("sigma", 0.0))
        if self.sigma < 0 or not np.isfinite(self.sigma):
            raise ValueError("Noise sigma must be finite and non-negative")
        self.rng = np.random.default_rng(self.config.get("seed", 0))

    def apply(self, observation: dict) -> dict:
        if self.kind is None or self.sigma == 0:
            return observation
        result = dict(observation)
        for key, value in observation.items():
            if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
                continue
            if value.ndim != 3 or value.shape[-1] != 3:
                continue
            if "stereo" not in key and "rgb" not in key:
                continue
            if self.kind == "gaussian":
                noise = self.rng.normal(0, self.sigma, value.shape)
                result[key] = np.clip(np.rint(value.astype(np.float32) + noise), 0, 255).astype(
                    np.uint8
                )
            else:
                result[key] = gaussian_blur_rgb(value, self.sigma)
        return result


def derive_sensor_noise_variant(
    subtype,
    rng,
    *,
    gaussian_range=(0.0, 20.0),
    blur_range=None,
    blur_levels=LIBERO_PLUS_BLUR_SIGMAS,
    sigma=None,
    severity=None,
):
    if subtype not in ("gaussian", "blur"):
        raise ValueError(f"Unknown sensor noise: {subtype}")
    if subtype == "gaussian":
        if severity is not None:
            raise ValueError("LIBERO-Plus blur severity does not define additive Gaussian noise")
        low, high = gaussian_range
        if not np.all(np.isfinite([low, high])) or not 0 <= low < high:
            raise ValueError("Gaussian range requires finite 0 <= low < high")
        sampled = (
            float(rng.uniform(max(float(low), np.finfo(float).eps), high))
            if sigma is None
            else float(sigma)
        )
        details = {
            "protocol": "humano_extension",
            "libero_plus_defined": False,
            "noise_model": "additive_zero_mean_gaussian",
            "sigma_range": [low, high],
            "quantization": "round_to_nearest_uint8",
        }
    else:
        levels = [float(value) for value in blur_levels]
        if not levels or not np.all(np.isfinite(levels)) or min(levels) <= 0:
            raise ValueError("Blur levels require finite positive sigma values")
        if blur_range is not None:
            low, high = blur_range
            if not np.all(np.isfinite([low, high])) or not 0 <= low < high:
                raise ValueError("Blur range requires finite 0 <= low < high")
            levels = [value for value in levels if low <= value <= high]
            if not levels:
                raise ValueError("Blur range contains no configured discrete sigma levels")
        if severity is not None:
            if (
                isinstance(severity, bool)
                or not isinstance(severity, int | np.integer)
                or not 1 <= severity <= 10
            ):
                raise ValueError(
                    "LIBERO-Plus Gaussian blur severity must be an integer from 1 to 10"
                )
            sampled = float(LIBERO_PLUS_BLUR_SIGMAS[severity - 1])
            if sigma is not None and float(sigma) != sampled:
                raise ValueError("Blur sigma disagrees with the LIBERO-Plus severity")
        else:
            sampled = float(rng.choice(levels)) if sigma is None else float(sigma)
        if not np.isfinite(sampled) or sampled < 0:
            raise ValueError("Noise sigma must be finite and nonnegative")
        aligned = sampled in LIBERO_PLUS_BLUR_SIGMAS
        details = {
            "protocol": "libero_plus_gaussian_blur" if aligned else "humano_extension",
            "libero_plus_defined": aligned,
            "severity": int(sampled) if aligned else None,
            "official_sigma_levels": list(LIBERO_PLUS_BLUR_SIGMAS),
            "sampling_sigma_levels": levels,
            "sampling": "uniform_discrete" if sigma is None and severity is None else "fixed",
            "sampling_origin": "humano_episode_selection",
            "source": LIBERO_PLUS_SOURCE + "#L56-L59",
            "implementation": "scipy.ndimage.gaussian_filter",
            "reference_implementation": "skimage.filters.gaussian",
            "boundary_mode": "nearest",
            "truncate": 4.0,
            "kernel_size": 2 * int(4 * sampled + 0.5) + 1,
            "channel_axis": -1,
            "channel_sigma": 0,
            "working_dtype": "float64",
            "quantization": "truncate_to_uint8",
            "reference_camera": "agentview_image",
            "humano_camera_scope": "all_stereo_or_rgb_keys",
        }
    if not np.isfinite(sampled) or sampled < 0:
        raise ValueError("Noise sigma must be finite and nonnegative")
    return {
        "subtype": subtype,
        "sigma": sampled,
        "seed": int(rng.integers(0, 2**31)),
        "units": "0-255 RGB" if subtype == "gaussian" else "pixels",
        "applied_before_policy_resize": True,
        **details,
    }
