import json

import numpy as np
from tune._utils import normalize_hp


def test_normalize_hp():
    assert isinstance(np.int64(10), np.int64)
    assert 10 == normalize_hp(np.int64(10))
    assert not isinstance(normalize_hp(np.int64(10)), np.int64)

    assert json.dumps(dict(a=[0, 1], b=1.1, c="x")) == json.dumps(
        normalize_hp(dict(a=[np.int64(0), 1], b=np.float64(1.1), c="x"))
    )
