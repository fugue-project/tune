from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from math import floor
from triad import assert_or_throw


class Grid(object):
    """Grid search, every value will be used

    :param args: values for the grid search
    """

    def __init__(self, *args: Any):
        self._values = list(args)

    def __iter__(self) -> Iterable[Any]:
        yield from self._values


class StochasticExpression(object):
    """Stochastic search base class"""

    @property
    def jsondict(self) -> Dict[str, Any]:  # pragma: no cover
        """Dict representation of the expression that is json serializable"""
        raise NotImplementedError

    def __eq__(self, other: Any):
        """Compare two ``StochasticExpression``"""
        return isinstance(other, type(self)) and self.jsondict == other.jsondict

    def get(self, seed: Any = None) -> Any:  # pragma: no cover
        """Return a randomly chosen value.

        :param seed: if set, it will be used to call :func:`~np:numpy.random.seed`
          , defaults to None
        """
        raise NotImplementedError


class Choice(StochasticExpression):
    """A random choice of values

    :param args: values to choose from
    """

    def __init__(self, *args: Any):
        self._values = list(args)

    @property
    def values(self) -> List[Any]:
        """values to choose from"""
        return self._values

    @property
    def jsondict(self) -> Dict[str, Any]:
        return dict(_expr_="choice", values=self.values)

    def get(self, seed: Any = None) -> Any:
        if seed is not None:
            np.random.seed(seed)
        return np.random.choice(self._values)


class RandBase(StochasticExpression):
    """Base class for continuous random variables

    :param q: step between adjacent values, if set, the value will be rounded
      using ``q``, defaults to None
    :param log: whether to apply ``exp`` to the random variable so that the logarithm
      of the return value is uniformly distributed, defaults to False
    """

    def __init__(
        self,
        q: Optional[float] = None,
        log: bool = False,
    ):
        if q is not None:
            assert_or_throw(q > 0, f"{q} <= 0")
        self.q = q
        self.log = log

    def distribution_func(self, seed: Any) -> float:  # pragma: no cover
        """The distribution function generating a random variable.
        Override this method in derived classes

        :param seed: if set, it will be used to call :func:`~np:numpy.random.seed`
          , defaults to None
        :return: a float number
        """
        raise NotImplementedError


class Rand(RandBase):
    """Continuous uniform random variables

    :param low: range low bound (inclusive)
    :param high: range high bound (exclusive)
    :param q: step between adjacent values, if set, the value will be rounded
      using ``q``, defaults to None
    :param log: whether to apply ``exp`` to the random variable so that the logarithm
      of the return value is uniformly distributed, defaults to False
    """

    def __init__(
        self,
        low: float,
        high: float,
        q: Optional[float] = None,
        log: bool = False,
        include_high: bool = True,
    ):
        if include_high:
            assert_or_throw(high >= low, f"{high} < {low}")
        else:
            assert_or_throw(high > low, f"{high} <= {low}")
        self.low = low
        self.high = high
        self.include_high = include_high
        super().__init__(q, log)

    @property
    def jsondict(self) -> Dict[str, Any]:
        res = dict(
            _expr_="rand",
            low=self.low,
            high=self.high,
            log=self.log,
            include_high=self.include_high,
        )
        if self.q is not None:
            res["q"] = self.q
        return res

    def distribution_func(self, seed: Any) -> float:
        if self.low == self.high:
            assert_or_throw(
                self.include_high,
                f"high {self.high} equals low but include_high = False",
            )
            return self.low
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(self.low, self.high)

    def get(self, seed: Any = None) -> float:
        high = self.high
        low = self.low
        while True:
            v = self.distribution_func(seed)
            if self.log:
                v = float(np.exp(v))
                low = np.exp(self.low)
                high = np.exp(self.high) + 0.0000001
            if low == high:
                return low
            if self.q is not None:
                v = low + self._round((v - low) / self.q) * self.q
            v = float(v)
            if (self.include_high and v <= high) or (
                not self.include_high and v < high
            ):
                return v

    def _round(self, v: Any) -> Any:
        if self.include_high:
            return np.round(v)
        return floor(v)


class RandInt(Rand):
    """Uniform distributed random integer values

    :param low: range low bound (inclusive)
    :param high: range high bound (exclusive)
    :param log: whether to apply ``exp`` to the random variable so that the logarithm
      of the return value is uniformly distributed, defaults to False
    """

    def __init__(
        self,
        low: int,
        high: int,
        log: bool = False,
        include_high: bool = True,
    ):
        super().__init__(low, high, q=1, log=log, include_high=include_high)

    @property
    def jsondict(self) -> Dict[str, Any]:
        return dict(
            _expr_="randint",
            low=self.low,
            high=self.high,
            log=self.log,
            include_high=self.include_high,
        )

    def get(self, seed: Any = None) -> int:
        return int(np.round(super().get(seed)))


class NormalRand(RandBase):
    """Continuous normally distributed random variables

    :param loc: mean of the normal distribution
    :param scale: standard deviation of the normal distribution
    :param q: step between adjacent values, if set, the value will be rounded
      using ``q``, defaults to None
    :param log: whether to apply ``exp`` to the random variable so that the logarithm
      of the return value is uniformly distributed, defaults to False
    """

    def __init__(
        self,
        loc: float,
        scale: float,
        q: Optional[float] = None,
        log: bool = False,
    ):
        assert_or_throw(scale > 0, f"{scale}<=0")
        self.loc = loc
        self.scale = scale
        super().__init__(q, log)

    @property
    def jsondict(self) -> Dict[str, Any]:
        res = dict(
            _expr_="randn",
            loc=self.loc,
            scale=self.scale,
            log=self.log,
        )
        if self.q is not None:
            res["q"] = self.q
        return res

    def get(self, seed: Any = None) -> float:
        v = self.distribution_func(seed)
        if self.log:
            v = float(np.exp(v))
        if self.q is not None:
            v = self.loc + np.round((v - self.loc) / self.q) * self.q
        v = float(v)
        return v

    def distribution_func(self, seed: Any) -> float:
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(self.loc, self.scale)


class NormalRandInt(NormalRand):
    """Normally distributed random integer values

    :param loc: mean of the normal distribution
    :param scale: standard deviation of the normal distribution
    :param log: whether to apply ``exp`` to the random variable so that the logarithm
      of the return value is uniformly distributed, defaults to False
    """

    def __init__(
        self,
        loc: int,
        scale: float,
        log: bool = False,
    ):
        super().__init__(loc, scale, q=1, log=log)

    @property
    def jsondict(self) -> Dict[str, Any]:
        return dict(
            _expr_="randnint",
            loc=self.loc,
            scale=self.scale,
            log=self.log,
        )

    def get(self, seed: Any = None) -> int:
        return int(np.round(super().get(seed)))


def _decode(value: Any) -> Any:
    if isinstance(value, str):
        return value
    elif isinstance(value, list):
        return [_decode(v) for v in value]
    elif isinstance(value, dict):
        if "_expr_" in value:
            e = value.pop("_expr_")
            if e == "choice":  # TODO: embeded rand is not supported
                return Choice(*value["values"])
            if e == "rand":
                return Rand(**value)
            if e == "randint":
                return RandInt(**value)
            if e == "randn":
                return NormalRand(**value)
            if e == "randnint":
                return NormalRandInt(**value)
            raise ValueError(e)  # pragma: no cover
        else:
            return {k: _decode(v) for k, v in value.items()}
    else:
        return value
