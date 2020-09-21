from copy import deepcopy
from typing import Any, Dict, Iterable, List, Callable, Tuple

from tune.utils.iter import dict_product, product


class Grid(object):
    def __init__(self, *args: Any):
        self._values = list(args)

    def __iter__(self) -> Iterable[Any]:
        yield from self._values


class SingleSpace(object):
    def __init__(self, **kwargs: Any):
        self._value = deepcopy(kwargs)
        self._grid: List[List[Tuple[Any, Any, Any]]] = []
        for k in self._value.keys():
            self._search(self._value, k)

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for tps in product(self._grid, safe=True, remove_empty=True):
            for tp in tps:
                tp[0][tp[1]] = tp[2]
            yield deepcopy(self._value)

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


class Space(object):
    def __init__(self, *args: Any, **kwargs: Any):
        self._groups: List[Spaces] = []
        for x in args:
            if isinstance(x, Space):
                self._groups.append(Spaces(x))
            elif isinstance(x, Spaces):
                self._groups.append(x)
            elif isinstance(x, dict):
                self._groups.append(Spaces(Space(**x)))
            elif isinstance(x, list):
                self._groups.append(Spaces(*x))
            else:
                raise ValueError(f"{x} is invalid")
        self._dict = {k: _SpaceValue(v) for k, v in kwargs.items()}

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


class Spaces(object):
    def __init__(self, *args: Any):
        self._spaces = [x if isinstance(x, Space) else Space(x) for x in args]

    @property
    def spaces(self) -> List[Space]:
        return self._spaces

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for space in self._spaces:
            yield from space


class _SpaceValue(object):
    def __init__(self, value: Any):
        self.value = value

    def __iter__(self) -> Iterable[Any]:
        if isinstance(self.value, (Space, Spaces)):
            yield from self.value
        elif isinstance(self.value, dict):
            yield from dict_product(
                {k: _SpaceValue(v) for k, v in self.value.items()}, safe=True
            )
        elif isinstance(self.value, list):
            yield from product([_SpaceValue(v) for v in self.value], safe=True)
        else:
            yield self.value
