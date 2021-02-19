from typing import Any, Dict, Iterable, List, Optional

import numpy as np


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
        self.q = q
        self.log = log

    def get(self, seed: Any = None) -> float:
        v = self.distribution_func(seed)
        if self.log:
            v = float(np.exp(v))
        if self.q is not None:
            v = np.round(v / self.q) * self.q
        return float(v)

    def distribution_func(self, seed: Any) -> float:
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
    ):
        self.low = low
        self.high = high
        super().__init__(q, log)

    @property
    def jsondict(self) -> Dict[str, Any]:
        res = dict(
            _expr_="rand",
            low=self.low,
            high=self.high,
            log=self.log,
        )
        if self.q is not None:
            res["q"] = self.q
        return res

    def distribution_func(self, seed: Any) -> float:
        if self.low == self.high:
            return self.low
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(self.low, self.high)


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
    ):
        super().__init__(low, high, q=1, log=log)

    @property
    def jsondict(self) -> Dict[str, Any]:
        return dict(
            _expr_="randint",
            low=self.low,
            high=self.high,
            log=self.log,
        )

    def get(self, seed: Any = None) -> int:
        return int(super().get(seed))


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
        scale: int,
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
        return int(super().get(seed))


def decode(value: Any) -> Any:
    if isinstance(value, str):
        return value
    elif isinstance(value, list):
        return [decode(v) for v in value]
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
            return {k: decode(v) for k, v in value.items()}
    else:
        return value
