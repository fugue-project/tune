import json

import numpy as np
from pytest import raises
from scipy import stats
from triad import to_uuid
from tune.concepts.space import (
    Choice,
    Grid,
    NormalRand,
    NormalRandInt,
    Rand,
    RandInt,
    Space,
)
from tune.concepts.space.parameters import decode_params
from tune._utils import assert_close


def test_grid():
    v = Grid("a", "b")
    assert ["a", "b"] == list(v)

    v2 = Grid("b", "a")
    assert v == v and v != v2
    assert to_uuid(v) != to_uuid(v2)

    raises(ValueError, lambda: Grid())


def test_choice():
    raises(ValueError, lambda: Choice())
    v = Choice("a", "b", "c")
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    assert v.generate_many(20, 0) == v.generate_many(20, 0)
    assert v.generate_many(20, 0) != v.generate_many(20, 1)
    actual = set(v.generate_many(20, 0))
    assert set(["a", "b", "c"]) == actual

    assert to_uuid(v) != to_uuid(Grid("a", "b", "c"))
    assert v != Grid("a", "b", "c")

    v = Choice(1, 2, 3)
    assert json.loads(json.dumps({"x": v.generate(0)}))["x"] <= 3

    v = Choice("a", "b", "c")
    assert isinstance(json.loads(json.dumps({"x": v.generate(0)}))["x"], str)


def test_rand():
    with raises(ValueError):
        Rand(1.0, 0.9)

    with raises(ValueError):
        Rand(1.0, 10, q=-0.1)

    with raises(ValueError):
        Rand(1.0, 1.0, include_high=False)

    with raises(ValueError):
        Rand(0.0, 1.0, log=True)  # for log, low>=1.0

    v = Rand(1.0, 1.0, q=0.1, log=False)
    assert 1.0 == v.generate()
    assert 1.0 == v.generate(10)

    v = Rand(1.0, 1.0, q=0.1, log=True)
    assert 1.0 == v.generate()
    assert 1.0 == v.generate(10)

    v = Rand(1.0, 2.0, q=0.1, log=False)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    assert_close([x / 10 for x in range(10, 21)], v.generate_many(100, 0))

    v = Rand(1.0, 2.09999, q=0.1, log=False)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    assert_close([x / 10 for x in range(10, 21)], v.generate_many(100, 0))

    v = Rand(1.0, 2.0, q=0.1, log=False, include_high=False)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    assert_close([x / 10 for x in range(10, 20)], v.generate_many(100, 0))

    v = Rand(1.0, 2.09999, q=0.1, log=False, include_high=False)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    assert_close([x / 10 for x in range(10, 21)], v.generate_many(100, 0))

    v = Rand(1.0, 2.0, log=True, include_high=False)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    res = v.generate_many(10000, 0)
    for x in res:
        assert x >= 1.0 and x <= 2.0
    t = stats.kstest(np.log(res), "uniform", args=(np.log(1), np.log(2) - np.log(1)))
    assert t.pvalue > 0.4


def test_randint():
    with raises(ValueError):
        RandInt(0, 10, log=True)  # for log, low>=1.0

    v = RandInt(10, 20, log=False)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    actual = set(v.generate_many(100, 0))
    assert set(range(10, 21)) == actual

    v = RandInt(10, 20, include_high=False)
    actual = set(v.generate_many(100, 0))
    assert set(range(10, 20)) == actual

    v = RandInt(10, 20, q=5, include_high=False)
    actual = set(v.generate_many(100, 0))
    assert set([10, 15]) == actual

    v = RandInt(10, 20, q=5, include_high=True)
    actual = set(v.generate_many(100, 0))
    assert set([10, 15, 20]) == actual

    v = RandInt(3, 20, log=True)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    actual = set(v.generate_many(1000, 0))
    assert set(range(3, 21)) == actual


def test_normal_rand():
    with raises(ValueError):
        NormalRand(1.0, 0.0)

    with raises(ValueError):
        NormalRand(1.0, -1.0)

    v = NormalRand(0.05, 0.2)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    res = v.generate_many(100000, 0)
    t = stats.kstest(res, "norm", args=(0.05, 0.2))
    assert t.pvalue > 0.4

    v = NormalRand(0.05, 0.2, q=0.1)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    actual = [x for x in v.generate_many(1000, 0) if x >= -0.155 and x <= 0.255]
    assert_close([-0.15, -0.05, 0.05, 0.15, 0.25], actual)


def test_normal_randint():
    v = NormalRandInt(5, 2)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    actual = set(v.generate_many(50, 0))
    for x in [3, 4, 5, 6, 7]:
        assert x in actual

    v = NormalRandInt(5, 2, q=3)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(2)
    actual = set(v.generate_many(50, 0))
    for x in [-1, 2, 5, 8, 11]:
        assert x in actual
    assert 6 not in actual


def test_encode_decode_params():
    s1 = Space(
        a=Grid(1, 2),
        b=Rand(1.0, 7.1, 0.2, log=True),
        c=Choice(1, 2, 3),
        d=[Grid(1, 2), Rand(0, 2.0)],
        e={"x": "xx", "y": Choice("a", "b")},
        f=RandInt(0, 10, log=False),
        g=NormalRand(0.1, 1.0, q=0.1),
        h=NormalRandInt(0.1, 1.0),
    )
    actual = [decode_params(x) for x in s1.encode()]
    assert list(s1) == actual
    for x in s1.encode():
        print(json.dumps(x, indent=2))
