from tune.concepts.space import TuningParametersTemplate
from tune import Rand, Grid


def test_tuning_parameters_template():
    data = dict(
        a=Rand(0, 1),
        b=Grid(2, 3),
        c=dict(
            a=Rand(10, 20), b=[dict(x=Rand(100, 200))], c=[1, Rand(1000, 2000)], d=None
        ),
        d=None,
    )
    e = TuningParametersTemplate(data)
    assert [
        Rand(0, 1),
        Grid(2, 3),
        Rand(10, 20),
        Rand(100, 200),
        Rand(1000, 2000),
    ] == e.params
    res = e.fill([0.5, 2, 10.5, 100.5, 1000.5], copy=True)
    # res2 doesn't affect res, because res is a copy
    res2 = e.fill([0.51, 2, 10.55, 100.5, 1000.5])
    # the following will modify res2, because it doesn't use copy
    e.fill([0.55, 2, 10.55, 100.5, 1000.5])
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
    e = TuningParametersTemplate(data)
    assert dict(p0=Rand(0, 1), p1=Grid(2, 3)) == e.params_dict
    assert dict(a=0.5, b=dict(x=[2])) == e.fill_dict(dict(p1=2, p0=0.5), copy=True)
