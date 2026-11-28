"""Opt-in lighting variants derived from a JSON LightingDR state.

Ranges here are our experimental choices, not a benchmark numerical standard.
RGB defaults to all lamps sharing a colored-LED emission color, with temperature
disabled but its numeric value unchanged. The diagnostic tint mode multiplies
base RGB and enables the base temperature; row scope retains the other row.
Local switches one row off without deleting lamps;
optional compensation preserves total scalar emission (before RGB/temperature).
Size changes radius only, outside training [0.08, 0.12] m, at fixed length.

USD CylinderLight has non-emitting end caps, hence emitting area A = 2*pi*r*L:
https://openusd.org/dev/user_guides/schemas/usdLux/CylinderLight.html
With inputs:normalize=False (explicit in Isaac), size intensity is I_base*A_base/A_new.
This preserves each lamp's emission power at its base color and temperature;
intensity is coupled to radius for power control, not an independent strength axis.
The analytic model assumes the existing unscaled CylinderLights. Renderer output
and shadow changes require an Isaac render check; this module does not render.
Nothing here changes training defaults or activates variants automatically.
RGB changes channel emission and perceived brightness at fixed scalar intensity;
it does not preserve photometric power or introduce an independent intensity axis.
"""

from __future__ import annotations

from copy import deepcopy
from math import pi
from typing import Any

import numpy as np


def _cylinder_area(light: dict[str, Any]) -> float:
    if light["type"] != "CylinderLight":
        raise ValueError("Power compensation requires CylinderLight")
    radius, length = float(light["light_radius"]), float(light["light_length"])
    if not (np.isfinite(radius) and np.isfinite(length) and radius > 0 and length > 0):
        raise ValueError("CylinderLight radius and length must be finite and positive")
    return 2 * pi * radius * length


def derive_lighting_variant(
    base_state: dict[str, dict[str, Any]],
    subtype: str,
    rng: np.random.Generator,
    *,
    side: int | None = None,
    hue: str | None = None,
    size_level: str | None = None,
    amount_range: tuple[float, float] = (0.65, 0.85),
    radius_ranges: dict[str, tuple[float, float]] | None = None,
    power_compensation: bool = False,
    color_mode: str = "rgb",
    rgb_scope: str = "all",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return an independent state and JSON manifest; consume only the supplied RNG.

    Input is the JSON-serializable uid->lamp mapping from LightingDR.state_dict().
    ``side`` is the tinted/off row (0 or 1) for row-scoped rgb/local. ``hue`` optionally fixes
    green/magenta/cyan. ``size_level`` optionally fixes low/high, sampled otherwise.
    ``amount_range`` and ``radius_ranges`` override our candidate ranges.
    ``power_compensation`` defaults to False for local (real row switch-off), and is recorded;
    size always compensates power. Unchanged legacy lamps may omit light_color,
    which LightingDR restores as white.
    RGB defaults to all lamps with final linear emission color, low channels
    0.15--0.35 and high channels 1. ``color_mode='rgb'`` disables temperature via
    light_enable_color_temperature=False; ``'tint'`` enables it and multiplies
    base RGB. Neither changes the temperature number. ``rgb_scope='row'`` selects
    one row; together with tint and amount_range=(0.2, 0.4) it reproduces the old control.
    """
    if subtype not in {"rgb", "local", "size"}:
        raise ValueError(f"Unknown lighting subtype: {subtype}")
    if not base_state:
        raise ValueError("Lighting state must contain lamps")
    state = deepcopy(base_state)
    params: dict[str, Any] = {"subtype": subtype, "numeric_standard": "project_defined"}

    if subtype in {"rgb", "local"}:
        groups = [[f"Light_{row}_{col}" for col in range(3)] for row in range(2)]
        if set(state) != set(groups[0] + groups[1]):
            raise ValueError("rgb/local require the existing 2x3 Light_<row>_<column> grid")
        if subtype == "rgb" and rgb_scope not in {"all", "row"}:
            raise ValueError("rgb_scope must be all or row")
        if subtype == "rgb" and rgb_scope == "all":
            if side is not None:
                raise ValueError("side requires rgb_scope='row'")
            selected, other = groups[0] + groups[1], []
        else:
            if side is None:
                side = int(rng.integers(2))
            if side not in (0, 1):
                raise ValueError("side must be 0 or 1")
            selected, other = groups[side], groups[1 - side]
        params.update(side=side, row_groups=groups)

    if subtype == "rgb":
        if color_mode not in {"rgb", "tint"}:
            raise ValueError("color_mode must be rgb or tint")
        targets = {"green": [0, 1, 0], "magenta": [1, 0, 1], "cyan": [0, 1, 1]}
        if hue is None:
            hue = str(rng.choice(list(targets)))
        if hue not in targets:
            raise ValueError("hue must be green, magenta or cyan")
        low, high = amount_range
        if not (np.isfinite(low) and np.isfinite(high) and 0 <= low < high <= 1):
            raise ValueError("amount_range must satisfy 0 <= low < high <= 1")
        amount = float(rng.uniform(low, high))
        tint = [1.0 - amount + amount * channel for channel in targets[hue]]
        for uid in selected:
            color = (
                base_state[uid].get("light_color", [1.0, 1.0, 1.0])
                if color_mode == "tint"
                else [1.0, 1.0, 1.0]
            )
            state[uid]["light_color"] = [float(c * t) for c, t in zip(color, tint, strict=True)]
            state[uid]["light_enable_color_temperature"] = color_mode == "tint"
        params.update(
            hue=hue,
            amount=amount,
            amount_sampling="uniform",
            amount_range=[float(low), float(high)],
            tint=tint,
            tinted_lamps=selected,
            base_lamps=other,
            color_mode=color_mode,
            rgb_scope=rgb_scope,
            temperature_enabled=color_mode == "tint",
            base_temperature_K={
                uid: base_state[uid]["light_color_temperature"] for uid in selected
            },
            final_input_rgb={uid: state[uid]["light_color"] for uid in selected},
            power_compensation=False,
        )

    elif subtype == "local":
        factor = 1.0
        if power_compensation:
            powers = {
                uid: _cylinder_area(light) * light["light_intensity"]
                for uid, light in base_state.items()
            }
            on_power = sum(powers[uid] for uid in other)
            if on_power <= 0:
                raise ValueError("Local power compensation requires positive on-row power")
            factor = float(sum(powers.values()) / on_power)
        for uid in selected:
            state[uid]["light_intensity"] = 0.0
        if power_compensation:
            for uid in other:
                state[uid]["light_intensity"] = float(base_state[uid]["light_intensity"] * factor)
        params.update(
            off_lamps=selected,
            on_lamps=other,
            power_compensation=power_compensation,
            intensity_factor=factor,
            power_model="scalar_intensity_times_emitting_area",
        )

    else:
        ranges = (
            radius_ranges
            if radius_ranges is not None
            else {"low": (0.03, 0.06), "high": (0.16, 0.24)}
        )
        if set(ranges) != {"low", "high"}:
            raise ValueError("radius_ranges must specify low and high")
        for level, (low, high) in ranges.items():
            if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high):
                raise ValueError("radius_ranges must be finite positive increasing intervals")
            if (level == "low" and high >= 0.08) or (level == "high" and low <= 0.12):
                raise ValueError("radius_ranges must lie strictly outside training [0.08, 0.12]")
        if size_level is None:
            size_level = str(rng.choice(list(ranges)))
        if size_level not in ranges:
            raise ValueError("size_level must be low or high")
        radius = float(rng.uniform(*ranges[size_level]))
        factors = {}
        for uid, light in base_state.items():
            state[uid]["light_radius"] = radius
            factor = _cylinder_area(light) / _cylinder_area(state[uid])
            state[uid]["light_intensity"] = float(light["light_intensity"] * factor)
            factors[uid] = factor
        params.update(
            size_level=size_level,
            radius_m=radius,
            radius_range_m=[float(value) for value in ranges[size_level]],
            training_radius_support_m=[0.08, 0.12],
            power_compensation=True,
            intensity_factors=factors,
            usd_normalize=False,
            power_model="2*pi*radius*length",
        )
    return state, params
