import json

import numpy as np
import pandas as pd
from pytest import raises
from scipy import stats
from triad import to_uuid
from tune._utils import assert_close
from tune.concepts.space import (
    Choice,
    Grid,
    NormalRand,
    NormalRandInt,
    Rand,
    RandInt,
    TransitionChoice,
    TuningParametersTemplate,
    to_template,
    FuncParam,
)


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

    v2 = Choice("a", "b", "c")
    v3 = Choice("a", "b", "d")
    assert to_uuid(v) == to_uuid(v2)
    assert to_uuid(v2) != to_uuid(v3)


def test_transition_choice():
    raises(ValueError, lambda: TransitionChoice())
    v = TransitionChoice("a", "b", "c")
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    assert v.generate_many(20, 0) == v.generate_many(20, 0)
    assert v.generate_many(20, 0) != v.generate_many(20, 1)
    actual = set(v.generate_many(20, 0))
    assert set(["a", "b", "c"]) == actual

    assert to_uuid(v) != to_uuid(Grid("a", "b", "c"))
    assert v != Grid("a", "b", "c")

    v = TransitionChoice(1, 2, 3)
    assert json.loads(json.dumps({"x": v.generate(0)}))["x"] <= 3

    v = TransitionChoice("a", "b", "c")
    assert isinstance(json.loads(json.dumps({"x": v.generate(0)}))["x"], str)

    v2 = TransitionChoice("a", "b", "c")
    v3 = Choice("a", "b", "c")
    assert to_uuid(v) == to_uuid(v2)
    assert to_uuid(v2) != to_uuid(v3)


def test_rand():
    with raises(ValueError):
        Rand(1.0, 0.9)

    with raises(ValueError):
        Rand(1.0, 10, q=-0.1)

    with raises(ValueError):
        Rand(1.0, 1.0, include_high=False)

    with raises(ValueError):
        Rand(0.0, 1.0, log=True)  # for log, low>=1.0

    v = Rand(0.1, 0.1, q=0.1, log=False)
    assert 0.1 == v.generate()
    assert 0.1 == v.generate(10)

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

    v = Rand(0.1, 2.0, log=True, include_high=False)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    res = v.generate_many(10000, 0)
    for x in res:
        assert x >= 0.1 and x <= 2.0
    t = stats.kstest(
        np.log(res), "uniform", args=(np.log(0.1), np.log(2) - np.log(0.1))
    )
    assert t.pvalue > 0.4

    v1 = Rand(1.0, 2.0, q=0.1, log=False)
    v2 = Rand(1.0, 2.0, log=False, q=0.1)
    v3 = Rand(1.0, 2.0, log=False)
    assert to_uuid(v1) == to_uuid(v2)
    assert to_uuid(v1) != to_uuid(v3)


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

    v1 = RandInt(1, 20, q=2)
    v2 = RandInt(1, 20, q=2)
    v3 = Rand(1, 20, q=2)
    assert to_uuid(v1) == to_uuid(v2)
    assert to_uuid(v1) != to_uuid(v3)


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

    v2 = NormalRand(0.05, 0.2, q=0.1)
    v3 = Rand(0.05, 0.2, q=0.1)
    assert to_uuid(v) == to_uuid(v2)
    assert to_uuid(v) != to_uuid(v3)


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

    v2 = NormalRandInt(5, 2, q=3)
    v3 = NormalRand(5, 2, q=3)
    assert to_uuid(v) == to_uuid(v2)
    assert to_uuid(v) != to_uuid(v3)


def test_func_param():
    def tf(*args, x, y):
        return sum(args) + x + y

    f1 = FuncParam(tf, 4, x=1, y=2)
    assert 7 == f1()
    f2 = FuncParam(tf, 4, x=1, y=2)
    f3 = FuncParam(tf, 5, x=1, y=2)
    assert f1 == f2
    assert f1 != f3
    assert to_uuid(f1) == to_uuid(f2)
    assert to_uuid(f1) != to_uuid(f3)
    f1[0] = 5
    f1["y"] = 3
    assert 5 == f1[0]
    assert 3 == f1["y"]
    assert 9 == f1()


def test_tuning_parameters_template():
    data = dict(a=1)
    e = make_template(data)
    assert e.empty
    assert not e.has_grid
    assert not e.has_stochastic

    data = dict(a=Rand(0, 1))
    e = make_template(data)
    assert not e.empty
    assert not e.has_grid
    assert e.has_stochastic

    data = dict(a=Grid(0, 1))
    e = make_template(data)
    assert not e.empty
    assert e.has_grid
    assert not e.has_stochastic

    data = dict(
        a=Rand(0, 1),
        b=Grid(2, 3),
        c=dict(
            a=Rand(10, 20), b=[dict(x=Rand(100, 200))], c=[1, Rand(1000, 2000)], d=None
        ),
        d=None,
    )
    e = make_template(data)
    assert not e.empty
    assert e.has_grid
    assert e.has_stochastic
    assert [
        Rand(0, 1),
        Grid(2, 3),
        Rand(10, 20),
        Rand(100, 200),
        Rand(1000, 2000),
    ] == e.params
    res = e.fill([0.5, 2, 10.5, 100.5, 1000.5])
    res2 = e.fill([0.55, 2, 10.55, 100.5, 1000.5])
    assert (
        dict(
            a=0.5,
            b=2,
            c=dict(a=10.5, b=[dict(x=100.5)], c=[1, 1000.5], d=None),
            d=None,
        )
        == res
    )
    assert res2 is not res
    assert (
        dict(
            a=0.55,
            b=2,
            c=dict(a=10.55, b=[dict(x=100.5)], c=[1, 1000.5], d=None),
            d=None,
        )
        == res2
    )

    # extract and fill by dicts
    data = dict(
        a=Rand(0, 1),
        b=dict(x=[Grid(2, 3)]),
    )
    e = make_template(data)
    assert dict(p0=Rand(0, 1), p1=Grid(2, 3)) == e.params_dict
    assert dict(a=0.5, b=dict(x=[2])) == e.fill_dict(dict(p1=2, p0=0.5))

    # same express in template
    expr = Rand(0, 1)
    data = dict(a=expr, b=dict(x=expr), c=Rand(2, 4))
    e = make_template(data)
    assert dict(p0=Rand(0, 1), p1=Rand(2, 4)) == e.params_dict
    assert dict(a=0.5, b=dict(x=0.5), c=2) == e.fill_dict(dict(p1=2, p0=0.5))

    # special objects
    e = make_template(dict(a=Rand(0, 1), b=pd.DataFrame([[0]])))

    # func
    def tf(*args, x):
        return sum(args) + x[0]

    u = Grid(0, 1)
    e = make_template(dict(a=1, b=[FuncParam(tf, Rand(0, 1), u, x=[u])]))
    assert e.has_grid
    assert e.has_stochastic
    assert dict(a=1, b=[2.5]) == e.fill([0.5, 1])


def test_template_eq():
    data1 = make_template(dict())
    data2 = make_template(dict())
    assert data1 == data2

    data1 = make_template(dict(a=1, b=2))
    data2 = make_template(dict(a=1, b=2))
    data3 = make_template(dict(a=1, b=3))
    assert data1 == data2
    assert data1 != data3

    data1 = make_template(dict(a=1, b=Grid(0, 1)))
    data2 = make_template(dict(a=1, b=Grid(0, 1)))
    data3 = make_template(dict(a=1, b=Grid(0, 2)))
    assert data1 == data2
    assert data1 != data3

    u = Grid(0, 1)
    v = Grid(0, 1)
    data1 = make_template(dict(a=1, b=u, c=u))
    data2 = dict(a=1, b=v, c=v)
    data3 = dict(a=1, b=u, c=v)
    assert data1 == data2
    assert data1 != data3
    assert data2 == data1
    assert data3 != data1


def test_template_product():
    data = make_template(dict())
    assert [dict()] == list(data.product_grid())

    data = make_template(dict(a=1, b=2))
    assert [dict(a=1, b=2)] == list(data.product_grid())

    data = make_template(dict(a=1, b=Grid(0, 1)))
    assert [dict(a=1, b=0), dict(a=1, b=1)] == list(data.product_grid())

    u = Grid(0, 1)
    data = make_template(dict(a=u, b=1, c=[u], d=Grid(0, 1)))
    assert [
        dict(a=0, b=1, c=[0], d=0),
        dict(a=0, b=1, c=[0], d=1),
        dict(a=1, b=1, c=[1], d=0),
        dict(a=1, b=1, c=[1], d=1),
    ] == list(data.product_grid())

    data = make_template(dict(a=1, b=Grid(0, 1), c=Rand(0, 1)))
    assert [dict(a=1, b=0, c=Rand(0, 1)), dict(a=1, b=1, c=Rand(0, 1))] == list(
        data.product_grid()
    )


def test_template_sample():
    data = make_template(dict())
    raises(ValueError, lambda: list(data.sample(0, 0)))
    raises(ValueError, lambda: list(data.sample(-1, 0)))
    assert [dict()] == list(data.sample(100, 0))

    data = make_template(dict(a=1, b=2))
    assert [dict(a=1, b=2)] == list(data.sample(100, 0))

    data = make_template(dict(a=1, b=Rand(0, 1)))
    assert list(data.sample(10, 0)) == list(data.sample(10, 0))
    assert list(data.sample(10, 0)) != list(data.sample(10, 1))
    a = list(data.sample(10, 0))
    assert 10 == len(a)
    assert all(x.template["b"] >= 0 and x.template["b"] <= 1 for x in a)
    assert all(x.empty for x in a)
    assert all(not x.has_grid for x in a)
    assert all(not x.has_stochastic for x in a)

    u = Rand(0, 1)
    data = make_template(dict(a=1, b=u, c=Grid(0, 1), d=[u]))
    a = list(data.sample(10, 0))
    assert 10 == len(a)
    assert all(x.template["b"] >= 0 and x.template["b"] <= 1 for x in a)
    assert all(x.template["d"][0] == x.template["b"] for x in a)
    assert all(not x.empty for x in a)
    assert all(x.has_grid for x in a)
    assert all(not x.has_stochastic for x in a)


def test_template_concat():
    u = Grid(0, 1)
    t1 = TuningParametersTemplate(dict(a=1, b=u, c=Grid(2, 3)))
    t2 = TuningParametersTemplate(dict(d=2, e=u, f=Grid(2, 3)))
    t = t1.concat(t2)
    assert dict(a=1, b=u, c=Grid(2, 3), d=2, e=u, f=Grid(2, 3)) == t
    assert dict(a=1, b=0, c=2) == t1.fill([0, 2])
    assert dict(d=2, e=1, f=3) == t2.fill([1, 3])
    assert dict(a=1, b=1, c=2, d=2, e=1, f=3) == t.fill([1, 2, 3])

    raises(ValueError, lambda: t.concat(t1))


def test_template_misc():
    # to_template
    t = to_template(dict(a=1, b=Grid(0, 1)))
    assert isinstance(t, TuningParametersTemplate)
    t2 = to_template(t)
    assert t is t2
    t3 = to_template(t.encode())
    assert t == t3

    raises(ValueError, lambda: to_template(123))

    # uuid
    u = Grid(0, 1)
    t1 = make_template(dict(a=1, b=u, c=Grid(0, 1)))
    t2 = make_template(dict(a=1, b=u, c=Grid(0, 1)))
    t3 = make_template(dict(a=1, b=u, c=u))
    t4 = make_template(dict(a=1, b=u, c=u))
    assert to_uuid(t1) == to_uuid(t2)
    assert to_uuid(t2) != to_uuid(t3)
    assert to_uuid(t3) == to_uuid(t4)

    # simple value
    u = Grid(0, 1)
    t1 = make_template(dict(a=1, b=u, c=Grid(0, 1), d=FuncParam(lambda x: x + 1, u)))
    raises(ValueError, lambda: t1.simple_value)
    assert [
        dict(a=1, b=0, c=0, d=1),
        dict(a=1, b=0, c=1, d=1),
        dict(a=1, b=1, c=0, d=2),
        dict(a=1, b=1, c=1, d=2),
    ] == list(t1.product_grid())

    t2 = make_template(dict(a=1, b=2))
    dict(a=1, b=2) == t2.simple_value

    t2 = make_template(dict(a=1, b=FuncParam(lambda x: x + 1, x=2)))
    assert dict(a=1, b=3) == t2.simple_value


def test_repr():
    assert "Grid('a', 'b')" == repr(Grid("a", "b"))
    assert "Choice('a', 'b')" == repr(Choice("a", "b"))
    assert "TransitionChoice('a', 'b')" == repr(TransitionChoice("a", "b"))
    assert "Rand(low=0.2, high=1, q=0.1, log=True, include_high=False)" == repr(
        Rand(0.2, 1, 0.1, True, False)
    )
    assert "RandInt(low=2, high=10, q=2, log=True, include_high=False)" == repr(
        RandInt(2, 10, 2, True, False)
    )
    assert "NormalRand(mu=0.1, sigma=0.2, q=0.3)" == repr(NormalRand(0.1, 0.2, 0.3))
    assert "NormalRandInt(mu=2, sigma=0.2, q=3)" == repr(NormalRandInt(2, 0.2, 3))
    assert "FuncParam(make_template, a=Grid('a', 'b'))" == repr(
        FuncParam(make_template, a=Grid("a", "b"))
    )
    assert "FuncParam(<lambda>, Grid('a', 'b'))" == repr(
        FuncParam(lambda x: x + 1, Grid("a", "b"))
    )
    assert "{'a': 1, 'b': Grid(1, 2)}" == repr(
        TuningParametersTemplate(dict(a=1, b=Grid(1, 2)))
    )
    assert "{'a': 1, 'b': Grid(1, 2)}" == str(
        TuningParametersTemplate(dict(a=1, b=Grid(1, 2)))
    )


def make_template(d):
    x = TuningParametersTemplate(d).encode()
    return TuningParametersTemplate.decode(x)
