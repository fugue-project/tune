from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple, no_type_check

import numpy as np
from triad import assert_or_throw

from tune._utils import dict_product, product


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
