from multiprocessing import Pool
from typing import Any, Callable, Dict, List, TypeVar

import cloudpickle
from triad.utils.convert import to_timedelta
from tune.constants import TUNE_STOPPER_DEFAULT_CHECK_INTERVAL
from tune.exceptions import TuneInterrupted

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
            return cloudpickle.loads(res)
        if stop_checker():
            pool.terminate()
            pool.join()
            raise TuneInterrupted()


def _run_target(blob: Any):
    tp = cloudpickle.loads(blob)
    return cloudpickle.dumps(tp[0](*tp[1], **tp[2]))
