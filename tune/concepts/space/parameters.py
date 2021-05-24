from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from triad import assert_or_throw, to_uuid
from tune._utils.math import (
    normal_to_continuous,
    normal_to_discrete,
    normal_to_integers,
    uniform_to_continuous,
    uniform_to_discrete,
    uniform_to_integers,
)


class Grid(object):
    """Grid search, every value will be used

    :param args: values for the grid search
    """

    def __init__(self, *args: Any):
        assert_or_throw(
            len(args) > 0, ValueError("Grid must take at least one element")
        )
        self._values = list(args)

    def __iter__(self) -> Iterable[Any]:
        yield from self._values

    def __eq__(self, other: Any):
        """Compare two ``Grid``"""
        return isinstance(other, type(self)) and self._values == other._values

    def __uuid__(self) -> str:
        return to_uuid("grid", self._values)


class StochasticExpression(object):
    """Stochastic search base class"""

    @property
    def jsondict(self) -> Dict[str, Any]:  # pragma: no cover
        """Dict representation of the expression that is json serializable"""
        raise NotImplementedError

    def __eq__(self, other: Any):
        """Compare two ``StochasticExpression``"""
        return isinstance(other, type(self)) and self.jsondict == other.jsondict

    def generate(self, seed: Any = None) -> Any:  # pragma: no cover
        """Return a randomly chosen value.

        :param seed: if set, it will be used to call :func:`~np:numpy.random.seed`
          , defaults to None
        """
        raise NotImplementedError

    def generate_many(self, n: int, seed: Any = None) -> List[Any]:
        if seed is not None:
            np.random.seed(seed)
        return [self.generate() for _ in range(n)]

    def __uuid__(self) -> str:
        return to_uuid(self.jsondict)


class Choice(StochasticExpression):
    """A random choice of values

    :param args: values to choose from
    """

    def __init__(self, *args: Any):
        assert_or_throw(
            len(args) > 0, ValueError("Choice must take at least one element")
        )
        self._values = list(args)

    @property
    def values(self) -> List[Any]:
        """values to choose from"""
        return self._values

    @property
    def jsondict(self) -> Dict[str, Any]:
        return dict(_expr_="choice", values=self.values)

    def generate(self, seed: Any = None) -> Any:
        if seed is not None:
            np.random.seed(seed)
        value = np.random.choice(self._values)
        if isinstance(value, np.generic):
            return value.item()
        return value  # pragma: no cover


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
            assert_or_throw(high >= low, ValueError(f"{high} < {low}"))
        else:
            assert_or_throw(high > low, ValueError(f"{high} <= {low}"))
        assert_or_throw(q is None or q > 0, ValueError(q))
        if log:
            assert_or_throw(
                low >= 1.0,
                ValueError(
                    f"for log sampling, low ({low}) must be greater or equal to 1.0"
                ),
            )
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

    def generate(self, seed: Any = None) -> float:
        if seed is not None:
            np.random.seed(seed)
        value = np.random.uniform()
        if self.q is None:
            return float(
                uniform_to_continuous(value, self.low, self.high, log=self.log)
            )
        else:
            return float(
                uniform_to_discrete(
                    value,
                    self.low,
                    self.high,
                    q=self.q,
                    log=self.log,
                    include_high=self.include_high,
                )
            )


class RandInt(RandBase):
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
        q: int = 1,
        log: bool = False,
        include_high: bool = True,
    ):
        if include_high:
            assert_or_throw(high >= low, ValueError(f"{high} < {low}"))
        else:
            assert_or_throw(high > low, ValueError(f"{high} <= {low}"))
        assert_or_throw(q > 0, ValueError(q))
        if log:
            assert_or_throw(
                low >= 1.0,
                ValueError(
                    f"for log sampling, low ({low}) must be greater or equal to 1.0"
                ),
            )
        self.low = low
        self.high = high
        self.include_high = include_high
        super().__init__(q, log)

    @property
    def jsondict(self) -> Dict[str, Any]:
        return dict(
            _expr_="randint",
            low=self.low,
            high=self.high,
            q=self.q,
            log=self.log,
            include_high=self.include_high,
        )

    def generate(self, seed: Any = None) -> float:
        if seed is not None:
            np.random.seed(seed)
        value = np.random.uniform()
        return int(
            uniform_to_integers(  # type: ignore
                value,
                self.low,
                self.high,
                q=int(self.q),  # type: ignore
                log=self.log,
                include_high=self.include_high,
            )
        )


class NormalRand(RandBase):
    """Continuous normally distributed random variables

    :param mu: mean of the normal distribution
    :param sigma: standard deviation of the normal distribution
    :param q: step between adjacent values, if set, the value will be rounded
      using ``q``, defaults to None
    :param log: whether to apply ``exp`` to the random variable so that the logarithm
      of the return value is uniformly distributed, defaults to False
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        q: Optional[float] = None,
    ):
        assert_or_throw(sigma > 0, ValueError(sigma))
        assert_or_throw(q is None or q > 0, ValueError(q))
        self.mu = mu
        self.sigma = sigma
        super().__init__(q)

    @property
    def jsondict(self) -> Dict[str, Any]:
        res = dict(
            _expr_="randn",
            mu=self.mu,
            sigma=self.sigma,
        )
        if self.q is not None:
            res["q"] = self.q
        return res

    def generate(self, seed: Any = None) -> float:
        if seed is not None:
            np.random.seed(seed)
        value = np.random.normal()
        if self.q is not None:
            return normal_to_discrete(value, mean=self.mu, sigma=self.sigma, q=self.q)
        else:
            return normal_to_continuous(value, mean=self.mu, sigma=self.sigma)


class NormalRandInt(RandBase):
    """Normally distributed random integer values

    :param mu: mean of the normal distribution
    :param sigma: standard deviation of the normal distribution
    :param log: whether to apply ``exp`` to the random variable so that the logarithm
      of the return value is uniformly distributed, defaults to False
    """

    def __init__(
        self,
        mu: int,
        sigma: float,
        q: int = 1,
    ):
        assert_or_throw(sigma > 0, ValueError(sigma))
        assert_or_throw(q > 0, ValueError(q))
        self.mu = mu
        self.sigma = sigma
        self.q = q
        super().__init__(q)

    @property
    def jsondict(self) -> Dict[str, Any]:
        return dict(
            _expr_="randnint",
            mu=self.mu,
            sigma=self.sigma,
            q=self.q,
        )

    def generate(self, seed: Any = None) -> int:
        if seed is not None:
            np.random.seed(seed)
        value = np.random.normal()
        return int(
            normal_to_integers(  # type: ignore
                value,
                mean=self.mu,
                sigma=self.sigma,
                q=self.q,  # type: ignore
            )
        )


def encode_params(value: Any):
    if isinstance(value, StochasticExpression):
        return value.jsondict
    elif isinstance(value, str):
        return value
    elif isinstance(value, list):
        return [encode_params(v) for v in value]
    elif isinstance(value, dict):
        return {k: encode_params(v) for k, v in value.items()}
    return value


def decode_params(value: Any) -> Any:
    if isinstance(value, str):
        return value
    elif isinstance(value, list):
        return [decode_params(v) for v in value]
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
            return {k: decode_params(v) for k, v in value.items()}
    else:
        return value
