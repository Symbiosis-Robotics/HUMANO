"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from humano.core import Layout

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from humano.core.actor import ObjectActor
from humano.core.randomizer import Randomizer, RandomizerCfg
from humano.utils import resolve_data_path

_UNSAFE_TASK_TABLE_PATH_PARTS = (
    "/glass/",
    "/gems/",
    "/liquids/",
    "/other/food/jelly",
)
_TRANSMISSIVE_PLASTIC_PATH_PARTS = (
    "clear",
    "cloudy",
    "spectral",
    "translucent",
)


def _is_safe_task_table_material(material: dict[str, Any]) -> bool:
    """Return whether a material can support visible task objects reliably."""
    if material.get("name") in _BLACK_TABLE_MATERIAL_NAMES:
        return False
    path = str(material.get("path", "")).replace("\\", "/").lower()
    if any(part in path for part in _UNSAFE_TASK_TABLE_PATH_PARTS):
        return False
    if "/plastic/" not in path:
        return True
    filename = path.rsplit("/", 1)[-1]
    return not any(part in filename for part in _TRANSMISSIVE_PLASTIC_PATH_PARTS)


class MaterialDR(Randomizer):
    # table_materials_infos: dict[str, list[dict]]
    # ground_materials_infos: dict[str, list[dict]]

    table_material: dict
    ground_material: dict

    material_mode: str

    def __init__(self, cfg: MaterialDRCfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.material_mode = cfg.material_mode
        self.material_split = np.load(
            resolve_data_path("materials/vMaterials_2/material_split.npy"), allow_pickle=True
        ).item()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads the state dict of the randomizer."""
        self._inner_state = state_dict

    def __call__(self, split: str, layout: Layout, **kwargs) -> dict[str, str]:
        if self._inner_state is not None:
            ret = self._inner_state
            # self._inner_state = None

            table = layout.actors.get("table", None)
            if table is not None:
                table.set_material(ret["table_material"])

            robot = layout.actors.get("robot", None)
            if robot is not None:
                robot.set_shaders(ret["robot_shader_params"])

            obj_idx = 0
            for key, obj in layout.actors.items():
                if isinstance(obj, ObjectActor):
                    obj.set_material(ret["object_shader_params"][obj_idx])
                    obj_idx += 1

            return ret

        # TODO wall materials
        self.table_materials_infos = self.material_split[split]["table"]
        self.ground_materials_infos = self.material_split[split]["ground"]

        if self.material_mode in ["rand_all", "rand_tableground"]:
            self.table_materials_infos = [
                material
                for material in self.table_materials_infos
                if _is_safe_task_table_material(material)
            ]
            if not self.table_materials_infos:
                raise RuntimeError(f"No safe task-table materials remain in split '{split}'")

            table_material = random.choice(self.table_materials_infos)
            ground_material = random.choice(self.ground_materials_infos)
        else:  # self.material_mode in ["fixed", "rand_objects"]:
            table_material = {
                "path": resolve_data_path(
                    "materials/vMaterials_2/Wood/Wood_Tiles_Ash.mdl", auto_download=True
                ),
                "name": "Wood_Tiles_Ash_Stackbond",
            }
            ground_material = {
                "path": resolve_data_path(
                    "materials/vMaterials_2/Concrete/Concrete_Floor_Damage.mdl", auto_download=True
                ),
                "name": "Concrete_Floor_Damage",
            }

        robot_shader_params = {
            "reflection_roughness_constant": np.random.uniform(0.0, 1.0)
            if self.material_mode != "fixed"
            else 0.5,
            "metallic_constant": np.random.uniform(0.0, 1.0)
            if self.material_mode != "fixed"
            else 0.0,
            "specular_level": np.random.uniform(0.0, 1.0) if self.material_mode != "fixed" else 0.0,
        }
        object_shader_params = []

        # num_objects = len([obj for obj in layout.actors.values() if isinstance(obj, ObjectActor)])

        # object_shader_params = [{
        #     'reflection_roughness_constant': np.random.uniform(0.0, 1.0 if self.material_mode != "fixed" else 0.5),
        #     'metallic_constant': np.random.uniform(0.0, 1.0) if self.material_mode != "fixed" else 0.,
        #     'specular_level': np.random.uniform(0.0, 1.0) if self.material_mode != "fixed" else 0.,
        #     } for _ in range(num_objects)
        # ]

        for key, obj in layout.actors.items():
            if isinstance(obj, ObjectActor):
                material = {
                    "reflection_roughness_constant": np.random.uniform(
                        0.0, 1.0 if self.material_mode != "fixed" else 0.5
                    ),
                    "metallic_constant": np.random.uniform(0.0, 1.0)
                    if self.material_mode != "fixed"
                    else 0.0,
                    "specular_level": np.random.uniform(0.0, 1.0)
                    if self.material_mode != "fixed"
                    else 0.0,
                }
                obj.set_material(material)
                object_shader_params.append(material)

        table = layout.actors.get("table", None)
        if table is not None:
            table.set_material(table_material)

        robot = layout.actors.get("robot", None)
        if robot is not None:
            robot.set_shaders(robot_shader_params)

        material_info = {
            "table_material": table_material,
            "ground_material": ground_material,  # TODO move this
            "robot_shader_params": robot_shader_params,
            "object_shader_params": object_shader_params,
        }
        return super()._transient(material_info)


@dataclass
class MaterialDRCfg(RandomizerCfg):
    material_mode: str = "fixed"  # fixed, rand_all, rand_tableground, rand_objects
    randmizer_class: Randomizer = MaterialDR


_BLACK_TABLE_MATERIAL_NAMES = frozenset(
    {
        "Polypropylene_Ivory",
        "Polycarbonate_Black",
        "Polyethylene_Ivory",
        "Polycarbonate_Dark_Blue",
        "Glass_Glazing_Spandrel_Dark_Blue",
        "Glass_Glazing_Spandrel_Bronze",
        "Glass_Glazing_Tinted_Dark_Bronze",
        "Glass_Clear_Large_Frit",
        "Devil_Black",
        "Polycarbonate_Light_Blue",
        "Polyethylene_Sky_Blue",
        "Styrofoam_no_SSS",
        "Glass_Glazing_Spandrel_White",
        "Glass_Glazing_Spandrel_Cool_Gray",
    }
)

_TRAIN_TABLE_MATERIAL_NAMES = frozenset(
    {
        "LegacySolidColorMatteBrown",
        "Paint_Eggshell_Teal",
        "Chalk_Paint_Pebbles_Deep_Emerald",
        "Wood_Tiles_Pine_Chantilly",
        "Wood_Tiles_Walnut_Chantilly",
        "Wood_Tiles_Fineline_Chantilly",
        "Terrazzo_Dry_Matt",
    }
)

_TEST_TABLE_MATERIAL_NAMES = frozenset(
    {
        "MattePlasticIvory",
        "MattePlasticCharcoal",
        "Aluminum_Brushed",
        "Iron_Brushed",
        "Wood_Tiles_Oak_Mountain_Herringbone",
        "Wood_Tiles_Oak_Mountain_Mosaic",
        "GlossyCyanPlasticG3",
        "PolishedSteelG3",
        "Gold_Brushed_Light_Brushing",
        "Bronze_Yellow_Rough_Worn",
        "Suede_Leather_Beige",
        "Aniline_Leather_Cognac_Red",
        "Suede_Leather_Turquoise",
        "Aniline_Leather_Shiny_Brown",
        "Mirror",
        "Mirror_Cool_Tint",
    }
)

_FALLBACK_MATERIALS = {
    "train": {
        "path": "data/resources/materials/vMaterials_2/Paint/Paint_Eggshell.mdl",
        "name": "Paint_Eggshell_Teal",
    },
    "test": {
        "type": "usd_preview_surface",
        "family": "Plastic",
        "name": "MattePlasticIvory",
        "diffuse_color": [0.72, 0.70, 0.63],
        "roughness": 0.82,
        "metallic": 0.0,
    },
}

_LEGACY_TRAIN_MATERIAL = {
    "type": "usd_preview_surface",
    "family": "SolidColor",
    "name": "LegacySolidColorMatteBrown",
    "diffuse_color": [0.24, 0.13, 0.07],
    "roughness": 0.82,
    "metallic": 0.0,
}


def resolve_render_table_material(
    recorded: dict[str, Any],
    *,
    distribution: str = "train",
    enforce_whitelist: bool = False,
    allowed_materials: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the recorded appearance or a safe fallback for the distribution."""
    if distribution not in _FALLBACK_MATERIALS:
        raise ValueError(f"Unsupported table material distribution: {distribution}")
    name = recorded.get("name")
    if allowed_materials is not None:
        allowed = {material["name"] for material in allowed_materials}
    elif distribution == "train":
        allowed = _TRAIN_TABLE_MATERIAL_NAMES
    else:
        allowed = _TEST_TABLE_MATERIAL_NAMES
    if name not in _BLACK_TABLE_MATERIAL_NAMES and (not enforce_whitelist or name in allowed):
        return deepcopy(recorded)
    fallback = (
        allowed_materials[0] if allowed_materials is not None else _FALLBACK_MATERIALS[distribution]
    )
    replacement = deepcopy(fallback)
    replacement["render_source"] = "table_material_fallback"
    replacement["fallback_from"] = name
    return replacement


def with_task_table_material(
    config: dict[str, Any], material: dict[str, Any], *, table_name: str = "table"
) -> dict[str, Any]:
    """Copy an episode config with its task-table material fields in agreement."""
    updated = deepcopy(config)
    state = updated.get("dr_state_dict") or {}
    scene = state.get("scene") or {}
    table = scene.get(table_name)
    if not isinstance(table, dict):
        return updated
    table["material"] = deepcopy(material)
    if isinstance(state.get("material"), dict):
        state["material"][f"{table_name}_material"] = deepcopy(material)
    layout = updated.get("layout") or {}
    for layout_table in (
        (layout.get("actors") or {}).get(table_name),
        (layout.get("scene") or {}).get(table_name),
    ):
        if isinstance(layout_table, dict):
            layout_table["material"] = deepcopy(material)
    return updated


def training_table_material_choices(
    assignment_manifest: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Use the materials actually assigned to new training renders, or historical brown."""
    if assignment_manifest is None:
        return [deepcopy(_LEGACY_TRAIN_MATERIAL)]
    choices: dict[str, dict[str, Any]] = {}
    for entry in assignment_manifest["assignments"]:
        material = resolve_render_table_material(
            {key: entry[key] for key in ("path", "name")}, enforce_whitelist=True
        )
        choices[material["name"]] = material
    if not choices:
        raise ValueError("Training table material assignment is empty")
    return list(choices.values())
