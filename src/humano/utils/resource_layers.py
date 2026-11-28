"""Re-anchor authored USD references when installing the grouped resources."""

from pathlib import Path


def relocate_resource_layer(path: Path) -> int:
    if path.name in {"Geometry.usda", "Geometry_visual_only.usda"}:
        if "molmo_ithor_ithor120_background" not in path.parts or path.parent.name != "Payload":
            return 0
        old = b"@../../../../objects/thor/"
        new = b"@../../../../../objects/thor/"
    elif path.parts[-5:] == (
        "storage_box",
        "storage_box_003_height2",
        "original",
        "output_usd",
        "storage_box_003_height2.usd",
    ):
        old = b"@../../storage_box_003/output_usd/"
        new = b"@../../../storage_box_003/original/output_usd/"
    else:
        return 0

    data = path.read_bytes()
    count = data.count(old)
    if count:
        path.write_bytes(data.replace(old, new))
    return count
