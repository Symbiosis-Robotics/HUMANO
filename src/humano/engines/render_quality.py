"""Named Isaac render-quality presets used by offline replay."""

from __future__ import annotations

import os
from dataclasses import dataclass

RENDER_QUALITY_ENV = "HUMANO_ISAAC_RENDER_QUALITY"


@dataclass(frozen=True)
class RenderQualityPreset:
    name: str
    app_renderer: str
    app_anti_aliasing: int
    replicator_mode: str
    anti_aliasing: str
    path_tracing_spp: int
    rt_subframes: int
    video_codec: str
    video_crf: int
    video_preset: str

    def to_metadata(self) -> dict[str, str | int]:
        return {
            "name": self.name,
            "renderer": self.app_renderer,
            "anti_aliasing": self.anti_aliasing,
            "path_tracing_spp": self.path_tracing_spp,
            "rt_subframes": self.rt_subframes,
            "video_codec": self.video_codec,
            "video_crf": self.video_crf,
            "video_preset": self.video_preset,
        }


_PRESETS = {
    "current": RenderQualityPreset(
        name="current",
        app_renderer="RayTracedLighting",
        app_anti_aliasing=0,
        replicator_mode="realtime",
        anti_aliasing="FXAA",
        path_tracing_spp=1,
        rt_subframes=1,
        video_codec="h264",
        video_crf=23,
        video_preset="medium",
    ),
    "medium": RenderQualityPreset(
        name="medium",
        app_renderer="RayTracedLighting",
        app_anti_aliasing=1,
        replicator_mode="realtime",
        anti_aliasing="TAA",
        path_tracing_spp=1,
        rt_subframes=1,
        video_codec="libx264",
        video_crf=18,
        video_preset="slow",
    ),
    "high": RenderQualityPreset(
        name="high",
        app_renderer="PathTracing",
        app_anti_aliasing=1,
        replicator_mode="pathtraced",
        anti_aliasing="TAA",
        path_tracing_spp=64,
        rt_subframes=1,
        video_codec="libx264",
        video_crf=18,
        video_preset="slow",
    ),
}


def get_render_quality_preset(name: str | None = None) -> RenderQualityPreset:
    selected = name or os.getenv(RENDER_QUALITY_ENV, "current")
    try:
        return _PRESETS[selected]
    except KeyError as exc:
        supported = ", ".join(sorted(_PRESETS))
        raise ValueError(
            f"Unsupported render quality {selected!r}; choose one of: {supported}"
        ) from exc
