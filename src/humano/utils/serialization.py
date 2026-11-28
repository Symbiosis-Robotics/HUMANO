from __future__ import annotations

import json
from json import JSONEncoder
from typing import Any

import numpy as np


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return JSONEncoder.default(self, obj)


def dump_json(obj: Any, fp: Any, **kwargs: Any) -> Any:
    return json.dump(obj, fp, cls=NumpyArrayEncoder, **kwargs)
