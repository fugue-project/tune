import itertools
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def normalize_hp(hp: Any) -> Any:
    if isinstance(hp, str):
        return hp
    elif isinstance(hp, list):
        return [normalize_hp(x) for x in hp]
    elif isinstance(hp, dict):
        return {k: normalize_hp(v) for k, v in hp.items()}
    elif isinstance(hp, np.generic):
        return hp.item()
    else:
        return hp


def dict_product(
    d: Dict[str, Iterable[Any]], safe: bool = True
) -> Iterable[Dict[str, Any]]:
    keys = d.keys()
    arrays = list(d.values())
    if len(arrays) == 0:
        if safe:
            yield {}
        return
    for element in _safe_product(arrays, safe):
        yield {k: v for k, v in zip(keys, element) if v is not _EMPTY_ITER}


def product(
    arrays: List[Iterable[Any]], safe: bool = False, remove_empty: bool = True
) -> Iterable[List[Any]]:
    if len(arrays) == 0:
        if safe:
            yield []
        return
    if remove_empty:
        for x in _safe_product(arrays, safe):
            yield [xx for xx in x if xx is not _EMPTY_ITER]
    else:
        for x in _safe_product(arrays, safe):
            yield [None if xx is _EMPTY_ITER else xx for xx in x]


def safe_iter(it: Iterable[Any], safe: bool = True) -> Iterable[Any]:
    if not safe:
        yield from it
    else:
        n = 0
        for x in it:
            yield x
            n += 1
        if n == 0:
            yield _EMPTY_ITER


def _safe_product(arrays: List[Iterable[Any]], safe: bool = True) -> Iterable[Tuple]:
    if not safe:
        yield from itertools.product(*arrays)
    else:
        arr = [safe_iter(t) for t in arrays]
        yield from itertools.product(*arr)


class _EmptyIter(object):
    pass


_EMPTY_ITER = _EmptyIter()
