import base64
import cloudpickle
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from triad import assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path
from tune._utils import product
from tune._utils.math import (
    normal_to_continuous,
    normal_to_discrete,
    normal_to_integers,
    uniform_to_continuous,
    uniform_to_discrete,
    uniform_to_integers,
)


class TuningParameterExpression:
    """Base class of all tuning parameter expressions"""

    pass


class Grid(TuningParameterExpression):
    """Grid search, every value will be used.
    Please read |SpaceTutorial|.

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

    def __repr__(self) -> str:
        return "Grid(" + repr(self._values)[1:-1] + ")"


class StochasticExpression(TuningParameterExpression):
    """Stochastic search base class.
    Please read |SpaceTutorial|.
    """

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
        """Generate ``n`` randomly chosen values

        :param n: number of random values to generate
        :param seed: random seed, defaults to None
        :return: a list of values
        """
        if seed is not None:
            np.random.seed(seed)
        return [self.generate() for _ in range(n)]

    def __uuid__(self) -> str:
        """Unique id for the expression"""
        return to_uuid(self.jsondict)


class Choice(StochasticExpression):
    """A random choice of values.
    Please read |SpaceTutorial|.

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

    def __repr__(self) -> str:
        return "Choice(" + repr(self._values)[1:-1] + ")"


class TransitionChoice(Choice):
    """An ordered random choice of values.
    Please read |SpaceTutorial|.

    :param args: values to choose from
    """

    @property
    def jsondict(self) -> Dict[str, Any]:
        return dict(_expr_="tchoice", values=self.values)

    def __repr__(self) -> str:
        return "TransitionChoice(" + repr(self._values)[1:-1] + ")"


class RandBase(StochasticExpression):
    """Base class for continuous random variables.
    Please read |SpaceTutorial|.

    :param q: step between adjacent values, if set, the value will be rounded
      using ``q``, defaults to None
    :param log: whether to do uniform sampling in log space, defaults to False.
      If True, lower values get higher chance to be sampled
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
    """Continuous uniform random variables.
    Please read |SpaceTutorial|.

    :param low: range low bound (inclusive)
    :param high: range high bound (exclusive)
    :param q: step between adjacent values, if set, the value will be rounded
      using ``q``, defaults to None
    :param log: whether to do uniform sampling in log space, defaults to False.
      If True, ``low`` must be positive and lower values get higher chance to be sampled
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
                low > 0.0,
                ValueError(f"for log sampling, low ({low}) must be greater than 0.0"),
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
            q=self.q,
            log=self.log,
            include_high=self.include_high,
        )
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

    def __repr__(self) -> str:
        return (
            f"Rand(low={self.low}, high={self.high}, q={self.q},"
            f" log={self.log}, include_high={self.include_high})"
        )


class RandInt(RandBase):
    """Uniform distributed random integer values.
    Please read |SpaceTutorial|.

    :param low: range low bound (inclusive)
    :param high: range high bound (exclusive)
    :param log: whether to do uniform sampling in log space, defaults to False.
      If True, ``low`` must be ``>=1`` and lower values get higher chance to be sampled
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

    def __repr__(self) -> str:
        return (
            f"RandInt(low={self.low}, high={self.high}, q={self.q},"
            f" log={self.log}, include_high={self.include_high})"
        )


class NormalRand(RandBase):
    """Continuous normally distributed random variables.
    Please read |SpaceTutorial|.

    :param mu: mean of the normal distribution
    :param sigma: standard deviation of the normal distribution
    :param q: step between adjacent values, if set, the value will be rounded
      using ``q``, defaults to None
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

    def __repr__(self) -> str:
        return f"NormalRand(mu={self.mu}, sigma={self.sigma}, q={self.q})"


class NormalRandInt(RandBase):
    """Normally distributed random integer values.
    Please read |SpaceTutorial|.

    :param mu: mean of the normal distribution
    :param sigma: standard deviation of the normal distribution
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

    def __repr__(self) -> str:
        return f"NormalRandInt(mu={self.mu}, sigma={self.sigma}, q={self.q})"


class FuncParam:
    """Function paramter. It defers the function call after all its parameters
    are no longer tuning parameters

    :param func: function to generate parameter value
    :param args: list arguments
    :param kwargs: key-value arguments

    .. code-block:: python

        s = Space(a=1, b=FuncParam(lambda x, y: x + y, x=Grid(0, 1), y=Grid(3, 4)))
        assert [
            dict(a=1, b=3),
            dict(a=1, b=4),
            dict(a=1, b=4),
            dict(a=1, b=5),
        ] == list(s)
    """

    def __init__(self, func: Callable, *args: Any, **kwargs: Any):
        self._func = func
        self._args = list(args)
        self._kwargs = dict(kwargs)

    def __uuid__(self) -> str:
        """Unique id for this expression"""
        return to_uuid(get_full_type_path(self._func), self._args, self._kwargs)

    def __call__(self) -> Any:
        """Call the function to generate value"""
        return self._func(*self._args, **self._kwargs)

    def __setitem__(self, key: Any, item: Any) -> None:
        """Update argument value

        :param key: key to set, if int, then set in ``args`` else set in ``kwargs``
        :param item: value to use
        """
        if isinstance(key, int):
            self._args[key] = item
        else:
            self._kwargs[key] = item

    def __getitem__(self, key: Any) -> Any:
        """Get argument value

        :param key: key to get, if int, then get in ``args`` else get in ``kwargs``
        :return: the correspondent value
        """
        if isinstance(key, int):
            return self._args[key]
        else:
            return self._kwargs[key]

    def __eq__(self, other: Any) -> bool:
        """Whether the expression equals to the other one

        :param other: another ``FuncParam``
        :return: whether they are equal
        """
        return (
            self._func is other._func
            and self._args == other._args
            and self._kwargs == other._kwargs
        )

    def __repr__(self) -> str:
        a: List[str] = [self._func.__name__]
        a += [repr(x) for x in self._args]
        a += [f"{k}={repr(v)}" for k, v in self._kwargs.items()]
        return "FuncParam(" + ", ".join(a) + ")"


class _MapUnit:
    def __init__(self, expr: TuningParameterExpression):
        self.expr = expr
        self.positions: List[List[Any]] = []

    def __eq__(self, other: Any) -> bool:
        return self.expr == other.expr and self.positions == other.positions

    def __uuid__(self) -> str:
        return to_uuid(self.expr, self.positions)

    def copy(self) -> "_MapUnit":
        res = _MapUnit(self.expr)
        res.positions = list(self.positions)
        return res


class TuningParametersTemplate:
    """Parameter template to extract tuning parameter expressions from
    nested data structure

    :param raw: the dictionary of input parameters.

    .. note::

        Please use :func:`~tune.concepts.space.parameters.to_template`
        to initialize this class.

    .. code-block:: python

        # common cases
        to_template(dict(a=1, b=1))
        to_template(dict(a=Rand(0, 1), b=1))

        # expressions may nest in dicts or arrays
        template = to_template(
            dict(a=dict(x1=Rand(0, 1), x2=Rand(3,4)), b=[Grid("a", "b")]))

        assert [Rand(0, 1), Rand(3, 4), Grid("a", "b")] == template.params
        assert dict(
            p0=Rand(0, 1), p1=Rand(3, 4), p2=Grid("a", "b")
        ) == template.params_dict
        assert dict(a=1, x2=3), b=["a"]) == template.fill([1, 3, "a"])
        assert dict(a=1, x2=3), b=["a"]) == template.fill_dict(
            dict(p2="a", p1=3, p0=1)
        )

    """

    def __init__(self, raw: Dict[str, Any]):
        self._units: List[_MapUnit] = []
        self._has_grid = False
        self._has_stochastic = False
        self._func_positions: List[List[Any]] = []
        self._template: Dict[str, Any] = self._copy(raw, [], {})
        self._uuid = ""

    def __eq__(self, other: Any) -> bool:
        """Check if the other object represents the same template

        :param other: an object convertible to ``TuningParametersTemplate``
            by :func:`~tune.concepts.space.parameters.to_template`
        :return: whether they are the same
        """
        o = to_template(other)
        return (
            self._has_grid == o._has_grid
            and self._has_stochastic == o._has_stochastic
            and self._template == o._template
            and self._units == o._units
            and self._func_positions == o._func_positions
        )

    def __uuid__(self):
        """The unique id representing this template"""
        if self._uuid == "":
            self._uuid = to_uuid(self._units, self._template)
        return self._uuid

    def __repr__(self) -> str:
        return repr(self.fill([x.expr for x in self._units]))

    @property
    def template(self) -> Dict[str, Any]:
        """The template dictionary, all tuning
        expressions will be replaced by ``None``
        """
        return self._template

    @property
    def simple_value(self) -> Dict[str, Any]:
        """If the template contains no tuning expression, it's simple
        and it will return parameters dictionary, otherwise, ``ValueError``
        will be raised
        """
        assert_or_throw(self.empty, ValueError("template contains tuning expressions"))
        if len(self._func_positions) == 0:
            return self._template
        return self._fill_funcs(deepcopy(self._template))

    @property
    def empty(self) -> bool:
        """Whether the template contains any tuning expression"""
        return len(self._units) == 0

    @property
    def has_grid(self) -> bool:
        """Whether the template contains grid expressions"""
        return self._has_grid

    @property
    def has_stochastic(self) -> bool:
        """Whether the template contains stochastic expressions"""
        return self._has_stochastic

    @property
    def params(self) -> List[TuningParameterExpression]:
        """Get all tuning parameter expressions in depth-first order"""
        return [x.expr for x in self._units]

    @property
    def params_dict(self) -> Dict[str, TuningParameterExpression]:
        """Get all tuning parameter expressions in depth-first order,
        with correspondent made-up new keys p0, p1, p2, ...
        """
        return {f"p{i}": x for i, x in enumerate(self.params)}

    def fill(self, params: List[Any]) -> Dict[str, Any]:
        """Fill the original data structure with values

        :param params: the list of values to be filled into the original
          data structure, in depth-first order
        :param copy: whether to return a deeply copied paramters,
          defaults to False
        :return: the original data structure filled with values
        """
        assert_or_throw(
            len(self._units) == len(params),
            ValueError("params count does not match template requirment"),
        )

        template = deepcopy(self._template)
        i = 0
        for u in self._units:
            for path in u.positions:
                self._fill_path(template, path, params[i])
            i += 1
        return self._fill_funcs(template)

    def fill_dict(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fill the original data structure with dictionary of values

        :param params: the dictionary of values to be filled into the original
          data structure, keys must be p0, p1, p2, ...
        :param copy: whether to return a deeply copied paramters,
          defaults to False
        :return: the original data structure filled with values
        """
        temp = [params[f"p{i}"] for i in range(len(params))]
        return self.fill(temp)

    def encode(self) -> str:
        """Convert the template to a base64 string"""
        return base64.b64encode(cloudpickle.dumps(self)).decode("ascii")

    @staticmethod
    def decode(data: str) -> "TuningParametersTemplate":
        """Retrieve the template from a base64 string"""
        return cloudpickle.loads(base64.b64decode(data.encode("ascii")))  # type: ignore

    def product_grid(self) -> Iterable["TuningParametersTemplate"]:
        """cross product all grid parameters

        :yield: new templates with the grid paramters filled

        .. code-block:: python

            assert [dict(a=1,b=Rand(0,1)), dict(a=2,b=Rand(0,1))] == \
                list(to_template(dict(a=Grid(1,2),b=Rand(0,1))).product_grid())
        """
        if not self.has_grid:
            yield self
        else:
            gu: List[Tuple[int, List[Any]]] = [
                (i, list(u.expr))  # type: ignore
                for i, u in enumerate(self._units)
                if isinstance(u.expr, Grid)
            ]
            yield from self._partial_fill(
                [x[0] for x in gu],
                product([data for _, data in gu], safe=True, remove_empty=True),
            )

    def sample(self, n: int, seed: Any = None) -> Iterable["TuningParametersTemplate"]:
        """sample all stochastic parameters

        :param n: number of samples, must be a positive integer
        :param seed: random seed defaulting to None.
          It will take effect if it is not None.

        :yield: new templates with the grid paramters filled

        .. code-block:: python

            assert [dict(a=1.1,b=Grid(0,1)), dict(a=1.5,b=Grid(0,1))] == \
                list(to_template(dict(a=Rand(1,2),b=Grid(0,1))).sample(2,0))
        """
        assert_or_throw(n > 0, ValueError("sample count must be positive"))
        if not self.has_stochastic:
            yield self
        else:
            if seed is not None:
                np.random.seed(seed)
            gu: List[Tuple[int, List[Any]]] = [
                (i, u.expr.generate_many(n))
                for i, u in enumerate(self._units)
                if isinstance(u.expr, StochasticExpression)
            ]
            yield from self._partial_fill(
                [x[0] for x in gu], zip(*[data for _, data in gu])
            )

    def concat(self, other: "TuningParametersTemplate") -> "TuningParametersTemplate":
        """Concatenate with another template and generate a new template.

        .. note::

            The other template must not have any key existed in this template, otherwise
            ``ValueError`` will be raised

        :return: the merged template
        """
        res = TuningParametersTemplate({})
        res._units = [x.copy() for x in self._units]
        res._has_grid = self._has_grid | other._has_grid
        res._has_stochastic = self._has_stochastic | other._has_stochastic
        res._template = dict(self._template)
        res._func_positions = self._func_positions + other._func_positions
        for k, v in other._template.items():
            assert_or_throw(
                k not in res._template,
                ValueError(f"{k} already exists in the original template"),
            )
            res._template[k] = v
        if not other.empty:
            temp_map = {id(x.expr): x for x in res._units}
            for u in other._units:
                if id(u.expr) in temp_map:
                    temp_map[id(u.expr)].positions += u.positions
                else:
                    res._units.append(u.copy())
        return res

    def _fill_funcs(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        def realize_func(path: List[Any]) -> None:
            r: Any = obj
            for p in path[:-1]:
                r = r[p]
            r[path[-1]] = r[path[-1]]()

        for path in self._func_positions:
            realize_func(path)

        return obj

    def _copy(self, src: Any, keys: List[Any], idx: Dict[int, _MapUnit]) -> Any:
        if isinstance(src, dict):
            ddest: Dict[str, Any] = {}
            for k, v in src.items():
                nk = keys + [k]
                if isinstance(v, TuningParameterExpression):
                    ddest[k] = None
                    self._add(nk, v, idx)
                else:
                    ddest[k] = self._copy(v, nk, idx)
            return ddest
        elif isinstance(src, list):
            adest: List[Any] = []
            for i in range(len(src)):
                nk = keys + [i]
                if isinstance(src[i], TuningParameterExpression):
                    adest.append(None)
                    self._add(nk, src[i], idx)
                else:
                    adest.append(self._copy(src[i], nk, idx))
            return adest
        elif isinstance(src, FuncParam):
            self._func_positions.append(keys)
            args = self._copy(src._args, keys, idx)
            kwargs = self._copy(src._kwargs, keys, idx)
            return FuncParam(src._func, *args, **kwargs)
        else:
            return src

    def _add(
        self, keys: List[Any], expr: TuningParameterExpression, idx: Dict[int, _MapUnit]
    ) -> None:
        if id(expr) not in idx:
            mu = _MapUnit(expr)
            self._units.append(mu)
            idx[id(expr)] = mu
        else:
            mu = idx[id(expr)]
        mu.positions.append(keys)
        if isinstance(expr, Grid):
            self._has_grid = True
        else:
            self._has_stochastic = True

    def _fill_path(self, root: Dict[str, Any], path: List[Any], v: Any) -> None:
        r = root
        for p in path[:-1]:
            r = r[p]
        r[path[-1]] = v

    def _partial_fill(
        self, idx: List[int], params_list: Iterable[List[Any]]
    ) -> Iterable["TuningParametersTemplate"]:
        new_units = [u for i, u in enumerate(self._units) if i not in idx]
        has_grid = any(isinstance(x.expr, Grid) for x in new_units)
        has_stochastic = any(
            isinstance(x.expr, StochasticExpression) for x in new_units
        )
        for params in params_list:
            new_template = deepcopy(self._template)
            for pi, i in enumerate(idx):
                for path in self._units[i].positions:
                    self._fill_path(new_template, path, params[pi])

            t = TuningParametersTemplate({})
            t._units = new_units
            t._template = new_template
            t._has_grid = has_grid
            t._has_stochastic = has_stochastic
            t._func_positions = self._func_positions

            if t.empty and len(t._func_positions) > 0:
                t._template = t._fill_funcs(t._template)
                t._func_positions = []
            yield t


def to_template(data: Any) -> TuningParametersTemplate:
    """Convert an oject to ``TuningParametersTemplate``

    :param data: data object (``dict`` or ``TuningParametersTemplate``
        or ``str`` (encoded string))
    :return: the template object
    """
    if isinstance(data, TuningParametersTemplate):
        return data
    if isinstance(data, dict):
        return TuningParametersTemplate(data)
    if isinstance(data, str):
        return TuningParametersTemplate.decode(data)
    raise ValueError(f"can't convert to template: {data}")
