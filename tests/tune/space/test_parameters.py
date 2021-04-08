import json

import numpy as np
from pytest import raises
from triad import to_uuid
from tune.space import Choice, Grid, NormalRand, NormalRandInt, Rand, RandInt, Space
from tune.space.parameters import decode_params


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
    with raises(AssertionError):
        Rand(1.0, 0.9)

    with raises(AssertionError):
        Rand(1.0, 1.0, include_high=False)

    v = Rand(1.0, 1.0, q=0.1, log=False)
    assert 1.0 == v.generate()
    assert 1.0 == v.generate(10)

    v = Rand(1.0, 1.0, q=0.1, log=True)
    assert np.exp(1) == v.generate()
    assert np.exp(1) == v.generate(10)

    v = Rand(1.0, 2.0, q=0.1, log=False)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    np.random.seed(0)
    actual = set(int(v.generate() * 10 + 0.0001) for _ in range(100))
    assert set(range(10, 21)) == actual

    v = Rand(1.0, 2.09999, q=0.1, log=False)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    np.random.seed(0)
    actual = set(int(v.generate() * 10 + 0.0001) for _ in range(100))
    assert set(range(10, 21)) == actual

    v = Rand(1.0, 2.0, q=0.1, log=False, include_high=False)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    np.random.seed(0)
    actual = set(int(v.generate() * 10 + 0.0001) for _ in range(100))
    assert set(range(10, 20)) == actual

    v = Rand(1.0, 2.09999, q=0.1, log=False, include_high=False)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    np.random.seed(0)
    actual = set(int(v.generate() * 10 + 0.0001) for _ in range(100))
    assert set(range(10, 21)) == actual

    v = Rand(1.0, 2.0, q=0.1, log=True, include_high=False)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    np.random.seed(0)
    for _ in range(100):
        x = v.generate()
        assert x >= np.exp(1.0) and x <= np.exp(2.0)


def test_randint():
    v = RandInt(10, 20, log=False)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    actual = set(v.generate_many(100, 0))
    assert set(range(10, 21)) == actual

    v = RandInt(10, 20, include_high=False)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    actual = set(v.generate_many(100, 0))
    assert set(range(10, 20)) == actual

    v = RandInt(1, 2, log=True)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    actual = set(v.generate_many(100, 0))
    assert set(range(3, 8)) == actual


def test_normal_rand():
    with raises(AssertionError):
        NormalRand(1.0, 0.0)

    with raises(AssertionError):
        NormalRand(1.0, -1.0)

    v = NormalRand(0.05, 0.2, q=0.1, log=False)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    np.random.seed(0)
    actual = set(int(np.round(v.generate() * 100)) for _ in range(50))
    for x in [5, -5, -15, 15, 25]:
        assert x in actual


def test_normal_randint():
    v = NormalRandInt(5, 2, log=False)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    actual = set(v.generate_many(50, 0))
    for x in [3, 4, 5, 6, 7]:
        assert x in actual

    v = NormalRandInt(1, 2, log=True)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    actual = set(v.generate_many(50, 0))
    for x in [1, 2, 3, 4, 5]:
        assert x in actual


def test_encode_decode_params():
    s1 = Space(
        a=Grid(1, 2),
        b=Rand(0, 1.0, 0.2, log=True),
        c=Choice(1, 2, 3),
        d=[Grid(1, 2), Rand(0, 2.0)],
        e={"x": "xx", "y": Choice("a", "b")},
        f=RandInt(0, 10, log=False),
        g=NormalRand(0.1, 1.0, q=0.1, log=False),
        h=NormalRandInt(0.1, 1.0, log=False),
    )
    actual = [decode_params(x) for x in s1.encode()]
    assert list(s1) == actual
    for x in s1.encode():
        print(json.dumps(x, indent=2))
