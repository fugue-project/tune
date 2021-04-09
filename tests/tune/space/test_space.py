import json

from pytest import raises

from tune import Choice, Grid, Rand, RandInt, Space
from tune.space import HorizontalSpace, VerticalSpace


def test_single_space():
    space = Space(a=1, b=Grid(2, 3, 4))
    dicts = list(space)
    dicts = list(space)  # make sure iteration doesn't alter the object
    assert 3 == len(dicts)
    assert dict(a=1, b=2) == dicts[0]
    assert dict(a=1, b=3) == dicts[1]

    dicts = list(Space(a=Grid(None, "x"), b=Grid(2, 3)))
    assert 4 == len(dicts)

    dicts = list(Space(a=1, b=[Grid(2, 3), Grid(4, 5)]))
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
    assert not Space(a=1).has_random_parameter
    assert not Space(a=1, b=Grid(1, 2)).has_random_parameter
    assert Space(a=1, b=[Grid(1, 2), Rand(0.0, 1.0)]).has_random_parameter

    dicts = list(Space(a=1, b=Grid(1, 2)).sample(100))
    assert 2 == len(dicts)

    space = Space(a=1, b=[Grid(1, 2), Rand(0.0, 1.0)], c=Choice("a", "b"))
    ec = list(space.encode())
    assert list(space.sample(5, 0)) == list(space.sample(5, 0))
    assert list(space.sample(5, 0)) != list(space.sample(5, 1))
    dicts = list(space.sample(5, 0))
    assert 10 == len(dicts)
    assert 5 == len(set(d["b"][1] for d in dicts))
    assert ec == list(space.encode())  # make sure the instance value was not altered


def test_horizontal_space():
    spaces = list(HorizontalSpace())
    assert 1 == len(spaces)
    assert {} == spaces[0]

    spaces = list(HorizontalSpace(a=10, b=[1, 2], c=dict(x=1)))
    assert 1 == len(spaces)
    assert dict(a=10, b=[1, 2], c=dict(x=1)) == spaces[0]

    spaces = list(HorizontalSpace(dict(a=10, b=[1, 2], c=dict(x=1))))
    assert 1 == len(spaces)
    assert dict(a=10, b=[1, 2], c=dict(x=1)) == spaces[0]

    spaces = list(HorizontalSpace(dict(a=10), dict(b=[1, 2], c=dict(x=1))))
    assert 1 == len(spaces)
    assert dict(a=10, b=[1, 2], c=dict(x=1)) == spaces[0]

    raises(ValueError, lambda: HorizontalSpace(10))

    assert not HorizontalSpace(dict(a=10, b=[1, 2], c=dict(x=1))).has_random_parameter

    space = HorizontalSpace(
        Space(a=Grid(1, 2), b=Rand(0.0, 1.0)), Space(c=Choice(3, 4))
    )
    ec = list(space.encode())
    assert space.has_random_parameter
    assert list(space.sample(5, 0)) == list(space.sample(5, 0))
    assert list(space.sample(5, 0)) != list(space.sample(5, 1))
    dicts = list(space.sample(5, 0))
    assert 10 == len(dicts)
    assert 5 == len(set(d["b"] for d in dicts))
    assert ec == list(space.encode())


def test_vertical_space():
    spaces = list(VerticalSpace())
    assert 0 == len(spaces)

    spaces = list(VerticalSpace(dict(a=10)))
    assert [dict(a=10)] == spaces

    spaces = list(VerticalSpace(dict(a=10), [dict(b=11), dict(c=12)]))
    assert [dict(a=10), dict(b=11), dict(c=12)] == spaces

    spaces = list(VerticalSpace(HorizontalSpace(a=10), dict(b=10)))
    assert [dict(a=10), dict(b=10)] == spaces

    raises(ValueError, lambda: VerticalSpace(10))

    assert not VerticalSpace(HorizontalSpace(a=10), dict(b=10)).has_random_parameter

    space = VerticalSpace(Space(a=Grid(1, 2), b=Rand(0.0, 1.0)), Space(c=Choice(3, 4)))
    ec = list(space.encode())
    assert space.has_random_parameter
    assert list(space.sample(5, 0)) == list(space.sample(5, 0))
    assert list(space.sample(5, 0)) != list(space.sample(5, 1))
    dicts = list(space.sample(5, 0))
    assert 15 == len(dicts)
    assert 5 == len(set(d["b"] for d in dicts if "b" in d))
    assert ec == list(space.encode())


def test_space_combo():
    spaces = list(HorizontalSpace(dict(a=10), []))
    assert [dict(a=10)] == spaces

    spaces = list(HorizontalSpace(dict(a=10), [dict(b=20), dict(c=30, a=11)]))
    assert 2 == len(spaces)
    assert dict(a=10, b=20) == spaces[0]
    assert dict(a=11, c=30) == spaces[1]

    spaces = list(
        HorizontalSpace(
            HorizontalSpace(a=10),
            VerticalSpace(dict(b=20), HorizontalSpace(c=30, a=None)),
        )
    )
    assert 2 == len(spaces)
    assert dict(a=10, b=20) == spaces[0]
    assert dict(a=None, c=30) == spaces[1]

    spaces = list(
        HorizontalSpace(
            dict(a=HorizontalSpace(dict(aa=10), VerticalSpace(dict(), dict(cc=12)))),
            VerticalSpace(dict(b=20), HorizontalSpace(c=30)),
        )
    )
    assert 4 == len(spaces)
    assert dict(a=dict(aa=10), b=20) == spaces[0]
    assert dict(a=dict(aa=10, cc=12), b=20) == spaces[1]
    assert dict(a=dict(aa=10), c=30) == spaces[2]
    assert dict(a=dict(aa=10, cc=12), c=30) == spaces[3]

    spaces = list(
        HorizontalSpace(a=VerticalSpace(HorizontalSpace(x=[1, 2]), dict(y=None)))
    )
    assert 2 == len(spaces)
    assert dict(a=dict(x=[1, 2])) == spaces[0]
    assert dict(a=dict(y=None)) == spaces[1]


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
        dict(a=1, b=3, c="a"),
        dict(a=1, b=2, c="b"),
        dict(a=1, b=3, c="b"),
    ] == list(s1 * [dict(c="a"), dict(c="b")])

    assert [
        dict(a=1, b=2),
        dict(a=1, b=3),
        dict(c="a"),
        dict(c="b"),
    ] == list(s1 + [dict(c="a"), dict(c="b")])

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
