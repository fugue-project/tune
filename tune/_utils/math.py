from typing import Any, List, Optional, Union

import numpy as np

_IGNORABLE_ERROR = 1e-8


def uniform_to_continuous(
    value: Any, low: float, high: float, log: bool = False, base: Optional[float] = None
) -> Any:
    if low >= high:
        return low if np.isscalar(value) else np.full(np.shape(value))
    if not log:
        return value * (high - low) + low
    if base is None:
        ll, lh = np.log(low), np.log(high)
        return np.exp(value * (lh - ll) + ll)
    else:
        b = np.log(base)
        ll, lh = np.log(low) / b, np.log(high) / b
        return np.power(base, value * (lh - ll) + ll)


def uniform_to_discrete(
    value: Any,
    low: float,
    high: float,
    q: float,
    log: bool = False,
    include_high: bool = True,
    base: Optional[float] = None,
) -> Any:
    if low >= high:
        return low if np.isscalar(value) else np.full(np.shape(value))
    _high = low + np.floor((high - low) / q + _IGNORABLE_ERROR) * q
    if abs(_high - high) < _IGNORABLE_ERROR:
        if include_high:
            _high = high + q
    else:
        _high += q
    _value = uniform_to_continuous(value, low, _high, log=log, base=base)
    return np.floor((_value - low) / q) * q + low


def uniform_to_integers(
    value: Any,
    low: int,
    high: int,
    q: int = 1,
    log: bool = False,
    include_high: bool = True,
    base: Optional[float] = None,
) -> Union[int, List[int]]:
    res = np.round(
        uniform_to_discrete(
            value, low, high, q=q, log=log, include_high=include_high, base=base
        )
    )
    if np.isscalar(res):
        return int(res)
    return [int(x) for x in res]


def uniform_to_choice(
    value: Any,
    choices: List[Any],
    log: bool = False,
    base: Optional[float] = None,
) -> Any:
    idx = uniform_to_integers(
        value, 1, len(choices), log=log, include_high=True, base=base
    )
    if isinstance(idx, int):
        return choices[idx]
    return [choices[x] for x in idx]
