from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple, no_type_check

from triad import assert_or_throw
from tune._utils import dict_product, product
from tune.space.parameters import Grid, StochasticExpression


# TODO: make this inherit from iterable?
class Space(object):
    def __init__(self, **kwargs: Any):
        self._value = deepcopy(kwargs)
        self._grid: List[List[Tuple[Any, Any, Any]]] = []
        for k in self._value.keys():
            self._search(self._value, k)

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for tps in product(self._grid, safe=True, remove_empty=True):  # type: ignore
            for tp in tps:
                tp[0][tp[1]] = tp[2]
            yield deepcopy(self._value)

    def encode(self) -> Iterable[Any]:
        for s in self:  # type: ignore
            yield self._encode_value(s)

    def __mul__(self, other: Any) -> "HorizontalSpace":
        return HorizontalSpace(self, other)

    def __add__(self, other: Any) -> "VerticalSpace":
        return VerticalSpace(self, other)

    def __radd__(self, other: Any) -> "Space":
        assert_or_throw(
            other is None or (isinstance(other, int) and other == 0), ValueError(other)
        )
        return self

    def _search(self, parent: Any, key: Any) -> None:
        node = parent[key]
        if isinstance(node, Grid):
            self._grid.append(self._grid_wrapper(parent, key))
        elif isinstance(node, dict):
            for k in node.keys():
                self._search(node, k)
        elif isinstance(node, list):
            for i in range(len(node)):
                self._search(node, i)

    def _grid_wrapper(self, parent: Any, key: Any) -> List[Tuple[Any, Any, Any]]:
        return [(parent, key, x) for x in parent[key]]

    def _encode_value(self, value: Any) -> Any:
        if isinstance(value, StochasticExpression):
            return value.jsondict
        elif isinstance(value, str):
            return value
        elif isinstance(value, list):
            return [self._encode_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._encode_value(v) for k, v in value.items()}
        return value


class HorizontalSpace(Space):
    def __init__(self, *args: Any, **kwargs: Any):
        self._groups: List[VerticalSpace] = []
        for x in args:
            if isinstance(x, HorizontalSpace):
                self._groups.append(VerticalSpace(x))
            elif isinstance(x, VerticalSpace):
                self._groups.append(x)
            elif isinstance(x, Space):
                self._groups.append(VerticalSpace(x))
            elif isinstance(x, dict):
                self._groups.append(VerticalSpace(HorizontalSpace(**x)))
            elif isinstance(x, list):
                self._groups.append(VerticalSpace(*x))
            else:
                raise ValueError(f"{x} is invalid")
        self._dict = {k: _SpaceValue(v) for k, v in kwargs.items()}

    @no_type_check  # TODO: remove this?
    def __iter__(self) -> Iterable[Dict[str, Any]]:
        dicts = list(dict_product(self._dict, safe=True))
        for spaces in product(
            [g.spaces for g in self._groups], safe=True, remove_empty=True
        ):
            for comb in product(list(spaces) + [dicts], safe=True, remove_empty=True):
                res: Dict[str, Any] = {}
                for d in comb:
                    res.update(d)
                yield res


class VerticalSpace(Space):
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
        return self._spaces

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for space in self._spaces:
            yield from space  # type: ignore


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
