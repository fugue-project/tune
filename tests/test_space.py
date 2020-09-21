from pytest import raises
from tune.space import Grid, SingleSpace, Space, Spaces


def test_single_space():
    dicts = list(SingleSpace(a=1, b=Grid(2, 3, 4)))
    assert 3 == len(dicts)
    assert dict(a=1, b=2) == dicts[0]
    assert dict(a=1, b=3) == dicts[1]

    dicts = list(SingleSpace(a=Grid(None, "x"), b=Grid(2, 3)))
    assert 4 == len(dicts)

    dicts = list(SingleSpace(a=1, b=[Grid(2, 3), Grid(4, 5)]))
    assert 4 == len(dicts)
    assert dict(a=1, b=[2, 4]) == dicts[0]
    assert dict(a=1, b=[2, 5]) == dicts[1]
    assert dict(a=1, b=[3, 4]) == dicts[2]
    assert dict(a=1, b=[3, 5]) == dicts[3]

    dicts = list(SingleSpace(a=1, b=dict(x=Grid(2, 3), y=Grid(4, 5))))
    assert 4 == len(dicts)
    assert dict(a=1, b=dict(x=2, y=4)) == dicts[0]
    assert dict(a=1, b=dict(x=2, y=5)) == dicts[1]
    assert dict(a=1, b=dict(x=3, y=4)) == dicts[2]
    assert dict(a=1, b=dict(x=3, y=5)) == dicts[3]


def test_space_simple_dict():
    spaces = list(Space())
    assert 1 == len(spaces)
    assert {} == spaces[0]

    spaces = list(Space(a=10, b=[1, 2], c=dict(x=1)))
    assert 1 == len(spaces)
    assert dict(a=10, b=[1, 2], c=dict(x=1)) == spaces[0]

    spaces = list(Space(dict(a=10, b=[1, 2], c=dict(x=1))))
    assert 1 == len(spaces)
    assert dict(a=10, b=[1, 2], c=dict(x=1)) == spaces[0]

    spaces = list(Space(dict(a=10), dict(b=[1, 2], c=dict(x=1))))
    assert 1 == len(spaces)
    assert dict(a=10, b=[1, 2], c=dict(x=1)) == spaces[0]

    raises(ValueError, lambda: Space(10))


def test_spaces():
    spaces = list(Spaces())
    assert 0 == len(spaces)

    spaces = list(Spaces(dict(a=10)))
    assert [dict(a=10)] == spaces

    spaces = list(Spaces(Space(a=10), dict(b=10)))
    assert [dict(a=10), dict(b=10)] == spaces


def test_space_combo():
    spaces = list(Space(dict(a=10), []))
    assert [dict(a=10)] == spaces

    spaces = list(Space(dict(a=10), [dict(b=20), dict(c=30, a=11)]))
    assert 2 == len(spaces)
    assert dict(a=10, b=20) == spaces[0]
    assert dict(a=11, c=30) == spaces[1]

    spaces = list(Space(Space(a=10), Spaces(dict(b=20), Space(c=30, a=None))))
    assert 2 == len(spaces)
    assert dict(a=10, b=20) == spaces[0]
    assert dict(a=None, c=30) == spaces[1]

    spaces = list(
        Space(
            dict(a=Space(dict(aa=10), Spaces(dict(), dict(cc=12)))),
            Spaces(dict(b=20), Space(c=30)),
        )
    )
    assert 4 == len(spaces)
    assert dict(a=dict(aa=10), b=20) == spaces[0]
    assert dict(a=dict(aa=10, cc=12), b=20) == spaces[1]
    assert dict(a=dict(aa=10), c=30) == spaces[2]
    assert dict(a=dict(aa=10, cc=12), c=30) == spaces[3]

    spaces = list(Space(a=Spaces(Space(x=[1, 2]), dict(y=None))))
    assert 2 == len(spaces)
    assert dict(a=dict(x=[1, 2])) == spaces[0]
    assert dict(a=dict(y=None)) == spaces[1]
