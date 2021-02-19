from typing import Any, Dict, List, Tuple

from fugue import ExecutionEngine
from pytest import raises

from tune.tune import _to_tunable, tunable
from tune.exceptions import TuneCompileError, TuneRuntimeError


def test_to_tunable():
    def t1(a: int) -> float:
        pass

    def t2(b: str) -> Dict[str, Any]:
        pass

    t22 = _to_tunable(t2)

    def t3() -> Dict[str, Any]:
        pass

    def t4(a: int) -> Tuple[float, List[str]]:
        pass

    def t5(e: ExecutionEngine, a: int) -> float:
        pass

    def t6(a: int, e: ExecutionEngine) -> float:
        pass

    def t7(**kwargs) -> float:
        pass

    def t8(**kwargs: int) -> float:
        pass

    def t9(a: int, **kwargs: int) -> float:
        pass

    def t10(e: ExecutionEngine, a: int, **kwargs: int) -> float:
        pass

    assert t1 is _to_tunable(t1)._func
    assert _to_tunable(t1).distributable
    assert _to_tunable(t1, distributable=True).distributable
    assert not _to_tunable(t1, distributable=False).distributable
    assert t2 is _to_tunable(_to_tunable(t2))._func
    assert t2 is _to_tunable("t22")._func
    assert t1 is _to_tunable("t1")._func
    assert t2 is _to_tunable("t2")._func
    assert t5 is _to_tunable(t5)._func
    assert not _to_tunable(t5).distributable
    assert not _to_tunable(t5, distributable=False).distributable

    assert t8 is _to_tunable(t8)._func
    assert t9 is _to_tunable(t9)._func
    assert t10 is _to_tunable(t10)._func
    # with execution engine, distributable can't be true
    raises(TuneCompileError, lambda: _to_tunable(t5, distributable=True))

    # return type must be float or Tuple[float,Dict[str,Any]]
    # input must not be empty
    with raises(TuneCompileError):
        _to_tunable(t3)

    with raises(TuneCompileError):
        _to_tunable("t3")

    with raises(TuneCompileError):
        _to_tunable(t4)

    with raises(TuneCompileError):
        _to_tunable(t6)


def test_deco():
    @tunable
    def t1(a: int, b: int) -> float:
        return a + b

    @tunable()
    def t2(a: int, b: int) -> Dict[str, Any]:
        return dict(error=a + b, metadata={"x": 1}, hp={"y": 2})

    @tunable
    def t3(e: ExecutionEngine, a: int) -> float:
        return a

    @tunable()
    def t4(e: ExecutionEngine, a: int) -> float:
        return a

    t11 = _to_tunable(t1)  # it will copy, t1 change will not affect t11
    t12 = _to_tunable("t1")  # it will copy, t1 change will not affect t11

    assert 5 == t1(2, 3)
    t1.run(a=3, b=4)
    assert 7 == t1.error
    assert t1.metadata == {}
    assert t1.hp == dict(a=3, b=4)
    assert t1.distributable

    raises(TuneRuntimeError, lambda: t11.hp)
    t11.run(a=4, b=5)
    assert t11.hp == dict(a=4, b=5)
    raises(TuneRuntimeError, lambda: t12.hp)

    assert t2(2, 3) == {"error": 5, "hp": {"y": 2}, "metadata": {"x": 1}}
    t2.run(a=3, b=4)
    assert 7 == t2.error
    assert t2.metadata == {"x": 1}
    assert t2.hp == {"y": 2}

    assert not t3.distributable
    assert not t4.distributable