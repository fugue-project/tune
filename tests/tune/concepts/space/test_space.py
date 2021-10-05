from tune import Choice, Grid, Rand, RandInt, Space, TuningParametersTemplate, FuncParam
from pytest import raises


def test_single_space():
    raises(ValueError, lambda: Space("abc"))
    raises(ValueError, lambda: Space(1))
    raises(ValueError, lambda: Space(1, 2))

    space = Space(a=1, b=Grid(2, 3, 4))
    dicts = list(space)
    dicts = list(space)  # make sure iteration doesn't alter the object
    assert 3 == len(dicts)
    assert dict(a=1, b=2) == dicts[0]
    assert dict(a=1, b=3) == dicts[1]

    dicts = list(Space(dict(a=Grid(None, "x"), b=Grid(2, 3))))
    assert 4 == len(dicts)

    dicts = list(Space(TuningParametersTemplate(dict(a=1, b=[Grid(2, 3), Grid(4, 5)]))))
    assert 4 == len(dicts)
    assert dict(a=1, b=[2, 4]) == dicts[0]
    assert dict(a=1, b=[2, 5]) == dicts[1]
    assert dict(a=1, b=[3, 4]) == dicts[2]
    assert dict(a=1, b=[3, 5]) == dicts[3]

    dicts = list(Space(a=1, b=dict(x=Grid(2, 3), y=Grid(4, 5))))
    assert 4 == len(dicts)
    assert dict(a=1, b=dict(x=2, y=4)) == dicts[0]
    assert dict(a=1, b=dict(x=2, y=5)) == dicts[1]
    assert dict(a=1, b=dict(x=3, y=4)) == dicts[2]
    assert dict(a=1, b=dict(x=3, y=5)) == dicts[3]


def test_single_space_sample():
    assert not Space(a=1).has_stochastic
    assert not Space(a=1, b=Grid(1, 2)).has_stochastic
    assert Space(a=1, b=[Grid(1, 2), Rand(0.0, 1.0)]).has_stochastic

    dicts = list(Space(a=1, b=Grid(1, 2)).sample(100))
    assert 2 == len(dicts)

    dicts = list(Space(a=1, b=RandInt(1, 2)).sample(100))
    assert 100 == len(dicts)

    space = Space(a=1, b=[Grid(1, 2), Rand(0.0, 1.0)], c=Choice("a", "b"))
    assert list(space.sample(5, 0)) == list(space.sample(5, 0))
    assert list(space.sample(5, 0)) != list(space.sample(5, 1))
    dicts = list(space.sample(5, 0))
    assert 10 == len(dicts)
    assert 5 == len(set(d.template["b"][1] for d in dicts))


def test_space_with_funct():
    s = Space(a=1, b=FuncParam(lambda x, y: x + y, x=Grid(0, 1), y=Grid(3, 4)))
    assert [
        dict(a=1, b=3),
        dict(a=1, b=4),
        dict(a=1, b=4),
        dict(a=1, b=5),
    ] == list(s)

    u = Grid(0, 1)
    s = Space(a=u, b=FuncParam(lambda x, y: x + y, x=u, y=u))
    assert [
        dict(a=0, b=0),
        dict(a=1, b=2),
    ] == list(s)


def test_operators():
    s1 = Space(a=1, b=Grid(2, 3))
    s2 = Space(c=Grid("a", "b"))
    assert [
        dict(a=1, b=2, c="a"),
        dict(a=1, b=2, c="b"),
        dict(a=1, b=3, c="a"),
        dict(a=1, b=3, c="b"),
    ] == list(s1 * s2)

    assert [
        dict(a=1, b=2),
        dict(a=1, b=3),
        dict(c="a"),
        dict(c="b"),
    ] == list(s1 + s2)

    assert [
        dict(a=1, b=2, c="a"),
        dict(a=1, b=2, c="b"),
        dict(a=1, b=3, c="a"),
        dict(a=1, b=3, c="b"),
    ] == list(s1 * (Space(c="a") + Space(c="b")))

    assert [
        dict(a=1, b=2),
        dict(a=1, b=3),
        dict(c="a"),
        dict(c="b"),
    ] == list(s1 + (Space(c="a") + Space(c="b")))

    s1 = Space(a=1, b=Grid(2, 3))
    s2 = Space(c=Grid("a", "b"))
    s3 = Space(d=5)
    assert (
        [
            dict(a=1, b=2),
            dict(a=1, b=3),
            dict(c="a"),
            dict(c="b"),
            dict(d=5),
        ]
        == list(sum([s1, s2, s3]))
        == list(sum([s1, s2, s3], None))
    )
