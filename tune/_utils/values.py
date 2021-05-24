from typing import Any

import numpy as np


def normalize_hp(hp: Any) -> Any:
    if isinstance(hp, str):
        return hp
    elif isinstance(hp, list):
        return [normalize_hp(x) for x in hp]
    elif isinstance(hp, dict):
        return {k: normalize_hp(v) for k, v in hp.items()}
    elif isinstance(hp, np.generic):
        return hp.item()
    else:
        return hp
