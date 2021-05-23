import math
from threading import RLock
from typing import Any, Callable, Dict, List, Tuple
from unittest import TestCase

from tune import (
    NonIterativeObjectiveLocalOptimizer,
    Rand,
    RandInt,
    Trial,
    noniterative_objective,
)
from tune._utils import assert_close
from tune.concepts.space.parameters import StochasticExpression
from tune.noniterative.objective import validate_noniterative_objective


class NonIterativeObjectiveLocalOptimizerTests(object):
    """DataFrame level general test suite.
    All new DataFrame types should pass this test suite.
    """

    class Tests(TestCase):
        def make_optimizer(self, **kwargs: Any) -> NonIterativeObjectiveLocalOptimizer:
            raise NotImplementedError

        def _test_rand(self):
            values = self._generate_values(Rand(-2.0, 3.0), lambda x: x ** 2)
            assert len(values) > 0
            assert all(x >= -2.0 and x <= 3.0 for x in values)

            values = self._generate_values(Rand(-2.0, 3.0, q=2.5), lambda x: x ** 2)
            assert_close(values, [-2.0, 0.5, 3.0])

            values = self._generate_values(Rand(-2.0, 3.0, q=3.0), lambda x: x ** 2)
            assert_close(values, [-2.0, 1.0])

            values = self._generate_values(Rand(-2.0, 3.0, log=True), lambda x: x ** 2)
            assert all(
                x >= math.exp(-2.0) - 1e-5 and x <= math.exp(3) + 1e-5 for x in values
            )

        def _test_randint(self):
            values = self._generate_values(RandInt(-2, 3), lambda x: x ** 2)
            assert_close(values, [-2, -1, 0, 1, 2, 3])
            assert all(isinstance(x, int) for x in values)

            values = self._generate_values(RandInt(-2, 3, 2), lambda x: x ** 2)
            assert_close(values, [-2, 0, 2])
            assert all(isinstance(x, int) for x in values)

        def test_optimization(self):
            params = dict(a=Rand(-10.0, 10.0), b=RandInt(-100, 100), c=2.0)
            trial = Trial("a", params, metadata={})
            o = self.make_optimizer(max_iter=200)

            @noniterative_objective
            def objective(a, b, c) -> Tuple[float, Dict[str, Any]]:
                return a ** 2 + b ** 2 + c, dict(a="x")

            def v(report):
                print(report.metric)
                assert report.metric < 7
                assert report.params["a"] ** 2 < 2
                assert report.params["b"] ** 2 < 2
                assert 2.0 == report.params["c"]
                assert "x" == report.metadata["a"]

            validate_noniterative_objective(objective, trial, v, optimizer=o)

        def _generate_values(
            self, expr: StochasticExpression, obj: Callable[..., float]
        ) -> List[Any]:
            params = dict(a=expr)
            trial = Trial("x", params, metadata={})
            o = self.make_optimizer(max_iter=30)
            lock = RLock()
            values: List[Any] = []

            @noniterative_objective
            def objective(a: Any) -> float:
                with lock:
                    values.append(a)
                return obj(a)

            o.run(objective, trial)  # type: ignore
            return values
