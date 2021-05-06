import itertools
from multiprocessing import Pool
from tune.constants import TUNE_STOPPER_DEFAULT_CHECK_INTERVAL
from typing import Any, Callable, Dict, Iterable, List, Tuple, TypeVar

import cloudpickle
import numpy as np
from triad.utils.convert import to_timedelta
from tune.exceptions import TuneInterrupted


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

T = TypeVar("T")


def run_monitored_process(
    target: Callable[..., T],
    args: List[Any],
    kwargs: Dict[str, Any],
    stop_checker: Callable[[], bool],
    interval: Any = TUNE_STOPPER_DEFAULT_CHECK_INTERVAL,
) -> T:
    interval_sec = to_timedelta(interval).total_seconds()
    blob = cloudpickle.dumps((target, args, kwargs))
    pool = Pool(1)
    result = pool.map_async(_run_target, [blob])
    while True:
        result.wait(interval_sec)
        if result.ready():
            pool.close()
            pool.join()
            res = result.get(timeout=1)[0]
            return cloudpickle.pickle.loads(res)
        if stop_checker():
            pool.terminate()
            pool.join()
            raise TuneInterrupted()


def _run_target(blob: Any):
    tp = cloudpickle.loads(blob)
    return cloudpickle.dumps(tp[0](*tp[1], **tp[2]))
