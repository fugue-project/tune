from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple, no_type_check

import numpy as np
from triad import assert_or_throw
from tune._utils import dict_product, product
from tune.concepts.space.parameters import Grid, StochasticExpression, _encode_params


class Space(object):
    """Search space. Please read |SpaceTutorial|.

    :param kwargs: parameters in the search space

    .. code-block:: python

        Space(a=1, b=1)  # static space
        Space(a=1, b=Grid(1,2), c=Grid("a", "b"))  # grid search
        Space(a=1, b=Grid(1,2), c=Rand(0, 1))  # grid search + level 2 search
        Space(a=1, b=Grid(1,2), c=Rand(0, 1)).sample(10, sedd=0)  # grid + random search

        # union
        Space(a=1, b=Grid(2,3)) + Space(b=Rand(1,5)).sample(10)

        # cross product
        Space(a=1, b=Grid(2,3)) * Space(c=Rand(1,5), d=Grid("a","b"))

        # combo (grid + random + level 2)
        space1 = Space(a=1, b=Grid(2,4))
        space2 = Space(b=RandInt(10, 20))
        space3 = Space(c=Rand(0,1)).sample(10)
        space = (space1 + space2) * space3

    .. code-block:: python

        assert Space(a=1, b=Rand(0,1)).has_random_parameter
        assert not Space(a=1, b=Rand(0,1)).sample(10).has_random_parameter
        assert not Space(a=1, b=Grid(0,1)).has_random_parameter
        assert not Space(a=1, b=1).has_random_parameter

        # get all configurations
        space = Space(a=Grid(2,4), b=Rand(0,1)).sample(100)
        for conf in space:
            print(conf)
        all_conf = list(space)

    """

    def __init__(self, **kwargs: Any):
        self._value = deepcopy(kwargs)
        self._grid: List[Tuple[Any, Any, Grid]] = []
        self._rand: List[Tuple[Any, Any, StochasticExpression]] = []
        for k in self._value.keys():
            self._search(self._value, k)

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        if len(self._grid) == 0:
            yield deepcopy(self._value)
            return
        rv = [[(x, y, z, v) for v in z] for x, y, z in self._grid]  # type: ignore
        for tps in product(rv, safe=True, remove_empty=True):  # type: ignore
            # overwrite Grid with one value
            for parent, key, _, v in tps:
                parent[key] = v
            res = deepcopy(self._value)
            # undo the overwrite
            for parent, key, orig, _ in tps:
                parent[key] = orig
            yield res

    @property
    def has_random_parameter(self):
        """Whether the space contains any
        :class:`~tune.concepts.space.parameters.StochasticExpression`
        """
        return len(self._rand) > 0

    def sample(self, n: int, seed: Any = None) -> "Space":
        """Draw random samples from the current space.
        Please read |SpaceTutorial|.

        :param n: number of samples to draw
        :param seed: random seed, defaults to None
        :return: a new Space containing all samples

        .. note::

            * it only applies to
              :class:`~tune.concepts.space.parameters.StochasticExpression`
            * if :meth:`~.has_random_parameter` is False, then it will return
              the original space
            * After sampling, no
              :class:`~tune.concepts.space.parameters.StochasticExpression`
              will exist in the new space.

        """
        if n <= 0 or not self.has_random_parameter:
            return self
        if seed is not None:
            np.random.seed(seed)
        return VerticalSpace(*self._sample_to_spaces(n))

    def encode(self) -> Iterable[Any]:
        """Extract all configurations from the current space
        and encode them to be json serializable

        :return: a list of json serializable objects
        """
        for s in self:  # type: ignore
            yield _encode_params(s)

    def __mul__(self, other: Any) -> "HorizontalSpace":
        """Operator ``*``"""
        return HorizontalSpace(self, other)

    def __add__(self, other: Any) -> "VerticalSpace":
        """Operator ``+``"""
        return VerticalSpace(self, other)

    def __radd__(self, other: Any) -> "Space":
        """Operator ``+``, this is to make compatible with ``sum``"""
        assert_or_throw(
            other is None or (isinstance(other, int) and other == 0), ValueError(other)
        )
        return self

    def _sample_to_spaces(self, n: int) -> List["Space"]:
        spaces: List["Space"] = []
        rv = [(x, y, z, z.generate_many(n)) for x, y, z in self._rand]
        for i in range(n):
            # overwrite StochasticExpressions with random values
            for parent, key, _, values in rv:
                parent[key] = values[i]
            space = Space(**self._value)
            # undo the overwrite
            for parent, key, orig, _ in rv:
                parent[key] = orig
            spaces.append(space)
        return spaces

    def _search(self, parent: Any, key: Any) -> None:
        node = parent[key]
        if isinstance(node, Grid):
            self._grid.append((parent, key, node))
        elif isinstance(node, StochasticExpression):
            self._rand.append((parent, key, node))
        elif isinstance(node, dict):
            for k in node.keys():
                self._search(node, k)
        elif isinstance(node, list):
            for i in range(len(node)):
                self._search(node, i)


class HorizontalSpace(Space):
    """Space from cross product.

    .. attention::

        Do not use this class directly, please use
        :class:`~.Space` and operator ``*`` instead. Read |SpaceTutorial|.

    """

    def __init__(self, *args: Any, **kwargs: Any):
        self._spaces: List[VerticalSpace] = []
        for x in args:
            if isinstance(x, HorizontalSpace):
                self._spaces.append(VerticalSpace(x))
            elif isinstance(x, VerticalSpace):
                self._spaces.append(x)
            elif isinstance(x, Space):
                self._spaces.append(VerticalSpace(x))
            elif isinstance(x, dict):
                self._spaces.append(VerticalSpace(HorizontalSpace(**x)))
            elif isinstance(x, list):
                self._spaces.append(VerticalSpace(*x))
            else:
                raise ValueError(f"{x} is invalid")
        self._dict = {k: _SpaceValue(v) for k, v in kwargs.items()}

    @no_type_check  # TODO: remove this?
    def __iter__(self) -> Iterable[Dict[str, Any]]:
        dicts = list(dict_product(self._dict, safe=True))
        for spaces in product(
            [g.spaces for g in self._spaces], safe=True, remove_empty=True
        ):
            for comb in product(list(spaces) + [dicts], safe=True, remove_empty=True):
                res: Dict[str, Any] = {}
                for d in comb:
                    res.update(d)
                yield res

    @property
    def has_random_parameter(self):
        return any(x.has_random_parameter for x in self._spaces)

    def _sample_to_spaces(self, n: int) -> List[Space]:
        lists = [s._sample_to_spaces(n) for s in self._spaces]
        return [HorizontalSpace(*args) for args in zip(*lists)]


class VerticalSpace(Space):
    """Space from union.

    .. attention::

        Do not use this class directly, please use
        :class:`~.Space` and operator ``+`` instead. Read |SpaceTutorial|.

    """

    def __init__(self, *args: Any):
        self._spaces: List[Space] = []
        for x in args:
            if isinstance(x, Space):
                self._spaces.append(x)
            elif isinstance(x, dict):
                self._spaces.append(Space(**x))
            elif isinstance(x, list):
                self._spaces.append(VerticalSpace(*x))
            else:
                raise ValueError(f"{x} is invalid")

    @property
    def spaces(self) -> List[Space]:
        """Spaces that unioned together"""
        return self._spaces

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for space in self._spaces:
            yield from space  # type: ignore

    @property
    def has_random_parameter(self):
        return any(x.has_random_parameter for x in self._spaces)

    def _sample_to_spaces(self, n: int) -> List[Space]:
        lists = [s._sample_to_spaces(n) for s in self._spaces]
        return [VerticalSpace(*args) for args in zip(*lists)]


class _SpaceValue(object):
    def __init__(self, value: Any):
        self.value = value

    @no_type_check  # TODO: remove this?
    def __iter__(self) -> Iterable[Any]:
        if isinstance(self.value, (HorizontalSpace, VerticalSpace)):
            yield from self.value
        elif isinstance(self.value, dict):
            yield from dict_product(
                {k: _SpaceValue(v) for k, v in self.value.items()}, safe=True
            )
        elif isinstance(self.value, list):
            yield from product([_SpaceValue(v) for v in self.value], safe=True)
        else:
            yield self.value
