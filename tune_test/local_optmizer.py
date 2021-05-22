from tune.noniterative.objective import validate_noniterative_objective
from typing import Any, Tuple, Dict
from unittest import TestCase

from tune import (
    NonIterativeObjectiveLocalOptimizer,
    Rand,
    RandInt,
    Trial,
    noniterative_objective,
)


class NonIterativeObjectiveLocalOptimizerTests(object):
    """DataFrame level general test suite.
    All new DataFrame types should pass this test suite.
    """

    class Tests(TestCase):
        def make_optimizer(self, **kwargs: Any) -> NonIterativeObjectiveLocalOptimizer:
            raise NotImplementedError

        def test_rand_randint(self):
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
