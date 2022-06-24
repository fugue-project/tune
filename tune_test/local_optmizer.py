from threading import RLock
from typing import Any, Callable, Dict, List, Tuple
from unittest import TestCase

from tune import (
    Choice,
    NonIterativeObjectiveLocalOptimizer,
    Rand,
    RandInt,
    StochasticExpression,
    TransitionChoice,
    Trial,
    noniterative_objective,
)
from tune._utils import assert_close
from tune.noniterative.objective import validate_noniterative_objective


class NonIterativeObjectiveLocalOptimizerTests:
    """DataFrame level general test suite.
    All new DataFrame types should pass this test suite.
    """

    class Tests(TestCase):
        def make_optimizer(self, **kwargs: Any) -> NonIterativeObjectiveLocalOptimizer:
            raise NotImplementedError

        def test_choice(self):
            d = dict(a=1, b=1, c=1)
            values = self._generate_values(Choice("a", "b", "c"), lambda x: d[x])
            assert len(values) > 0
            assert all(x in ["a", "b", "c"] for x in values)
            assert all(c in values for c in ["a", "b", "c"])

        def test_transition_choice(self):
            d = dict(a=1, b=1, c=1)
            values = self._generate_values(
                TransitionChoice("a", "b", "c"), lambda x: d[x]
            )
            assert len(values) > 0
            assert all(x in ["a", "b", "c"] for x in values)
            assert all(c in values for c in ["a", "b", "c"])

        def test_rand(self):
            # common case
            values = self._generate_values(Rand(-2.0, 3.0), lambda x: x**2)
            assert len(values) > 0
            assert all(x >= -2.0 and x <= 3.0 for x in values)

            # with q, and range%q == 0
            values = self._generate_values(Rand(-2.0, 3.0, q=2.5), lambda x: x**2)
            assert_close(values, [-2.0, 0.5, 3.0])

            values = self._generate_values(
                Rand(-2.0, 3.0, q=2.5, include_high=False), lambda x: x**2
            )
            assert_close(values, [-2.0, 0.5])

            # with q, and range%q != 0
            for ih in [True, False]:
                values = self._generate_values(
                    Rand(-2.0, 3.0, q=3.0, include_high=ih), lambda x: x**2
                )
                assert_close(values, [-2.0, 1.0])

            # with log
            values = self._generate_values(Rand(0.1, 3.0, log=True), lambda x: x**2)
            assert all(x >= 0.1 and x <= 3.0 for x in values)

            # with log and q, and range%q == 0
            values = self._generate_values(
                Rand(1.0, 6.0, q=2.5, log=True), lambda x: x**2
            )
            assert_close(values, [1.0, 3.5, 6.0])

            values = self._generate_values(
                Rand(1.0, 6.0, q=2.5, log=True, include_high=False), lambda x: x**2
            )
            assert_close(values, [1.0, 3.5])

            # with log and q, and range%q != 0
            for ih in [True, False]:
                values = self._generate_values(
                    Rand(1.0, 6.0, q=3.0, log=True, include_high=ih), lambda x: x**2
                )
                assert_close(values, [1.0, 4.0])

        def test_randint(self):
            for log in [True, False]:
                # common case
                values = self._generate_values(RandInt(1, 3, log=log), lambda x: x**2)
                assert_close(values, [1, 2, 3])
                assert all(isinstance(x, int) for x in values)

                values = self._generate_values(
                    RandInt(1, 3, include_high=False, log=log), lambda x: x**2
                )
                assert_close(values, [1, 2])

                # with q, range % q != 0
                for ih in [True, False]:
                    values = self._generate_values(
                        RandInt(1, 6, 2, include_high=ih, log=log), lambda x: x**2
                    )
                    assert_close(values, [1, 3, 5])
                    assert all(isinstance(x, int) for x in values)

                # with q, range % q == 0
                values = self._generate_values(
                    RandInt(1, 5, 2, log=log), lambda x: x**2
                )
                assert_close(values, [1, 3, 5])

                values = self._generate_values(
                    RandInt(1, 5, 2, include_high=False, log=log), lambda x: x**2
                )
                assert_close(values, [1, 3])

        def test_optimization_dummy(self):
            params = dict(a=1, b=2, c=3)
            trial = Trial("a", params, metadata={})
            o = self.make_optimizer(max_iter=5)

            @noniterative_objective
            def objective(a, b, c) -> Tuple[float, Dict[str, Any]]:
                return a**2 + b**2 + c, dict(a="x")

            def v(report):
                assert 1 == report.params.simple_value["a"]
                assert 2 == report.params.simple_value["b"]
                assert 3 == report.params.simple_value["c"]
                assert report.metric == 8
                assert "x" == report.metadata["a"]

            validate_noniterative_objective(objective, trial, v, optimizer=o)

        def test_optimization(self):
            params = dict(a=Rand(-10.0, 10.0), b=RandInt(-100, 100), c=2.0)
            trial = Trial("a", params, metadata={})
            o = self.make_optimizer(max_iter=200)

            @noniterative_objective
            def objective(a, b, c) -> Tuple[float, Dict[str, Any]]:
                return a**2 + b**2 + c, dict(a="x")

            def v(report):
                print(report.metric)
                assert report.metric < 7
                assert report.params.simple_value["a"] ** 2 < 2
                assert report.params.simple_value["b"] ** 2 < 2
                assert 2.0 == report.params.simple_value["c"]
                assert "x" == report.metadata["a"]

            validate_noniterative_objective(objective, trial, v, optimizer=o)

        def test_optimization_nested_param(self):
            params = dict(a=dict(x=Rand(-10.0, 10.0)), b=[RandInt(-100, 100)], c=[2.0])
            trial = Trial("a", params, metadata={})
            o = self.make_optimizer(max_iter=200)

            @noniterative_objective
            def objective(a, b, c) -> Tuple[float, Dict[str, Any]]:
                return a["x"] ** 2 + b[0] ** 2 + c[0], dict(a="x")

            def v(report):
                print(report.metric)
                assert report.metric < 7
                assert report.params.simple_value["a"]["x"] ** 2 < 2
                assert report.params.simple_value["b"][0] ** 2 < 2
                assert 2.0 == report.params.simple_value["c"][0]
                assert "x" == report.metadata["a"]

            validate_noniterative_objective(objective, trial, v, optimizer=o)

        def _generate_values(
            self,
            expr: StochasticExpression,
            obj: Callable[..., float],
            logger: Any = None,
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

            o.run(objective, trial, logger=logger)  # type: ignore
            return values
