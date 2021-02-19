from typing import Any, Dict, Tuple

import numpy as np
from pytest import raises

from tune.exceptions import TuneRuntimeError
from tune import SimpleTunable, Grid, Space


def test_tunable():
    t = _MockTunable()
    t.run(a=1, b=2)
    assert 3.0 == t.error
    assert t.metadata == {}
    assert t.hp == {"a": 1, "b": 2}

    t = _MockTunable()
    t.run(a=1, m=2, x=2)
    assert 5.0 == t.error
    assert t.metadata == {"m": 2}
    assert t.hp == {"x": 2}

    t = _MockTunable()
    raises(TuneRuntimeError, lambda: t.error)
    raises(TuneRuntimeError, lambda: t.hp)
    raises(TuneRuntimeError, lambda: t.metadata)
    raises(TuneRuntimeError, lambda: t.execution_engine)


def test_tunable_with_space():
    t = _MockTunable()
    s1 = Space(a=Grid(0, 1))
    s2 = Space(b=Grid(3, 4))
    assert [
        {"a": 0, "b": 3},
        {"a": 0, "b": 4},
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
    ] == list(t.space(s1, s2).space.encode())
    assert [{"a": 0, "b": 10}, {"a": 1, "b": 10}] == list(
        t.space(a=Grid(0, 1), b=10).space.encode()
    )
    assert [
        {"a": 0, "b": 3},
        {"a": 0, "b": 4},
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
    ] == list(t.space(s1, b=Grid(3, 4)).space.encode())


class _MockTunable(SimpleTunable):
    def tune(self, **kwargs: Any) -> Dict[str, Any]:
        error = np.double(sum(kwargs.values()))
        res = {"error": error}
        if "m" in kwargs:
            res["metadata"] = {"m": kwargs["m"]}
        if "x" in kwargs:
            res["hp"] = {"x": kwargs["x"]}
        return res
