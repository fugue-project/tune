from typing import Any

from tune import parse_noniterative_local_optimizer
from tune.noniterative.objective import NonIterativeObjectiveLocalOptimizer
from tune_test.local_optmizer import NonIterativeObjectiveLocalOptimizerTests

from tune_optuna.optimizer import OptunaLocalOptimizer


class HyperoptLocalOptimizerTests(NonIterativeObjectiveLocalOptimizerTests.Tests):
    def make_optimizer(self, **kwargs: Any) -> NonIterativeObjectiveLocalOptimizer:
        if "max_iter" in kwargs and len(kwargs) == 1:
            return parse_noniterative_local_optimizer(
                "optuna:" + str(kwargs["max_iter"])
            )
        return OptunaLocalOptimizer(**kwargs)


def test_parse_noniterative_local_optimizer():
    o1 = parse_noniterative_local_optimizer("optuna")
    assert isinstance(o1, OptunaLocalOptimizer)
    assert o1._max_iter == 30

    o2 = parse_noniterative_local_optimizer("optuna:10")
    assert isinstance(o2, OptunaLocalOptimizer)
    assert o2._max_iter == 10
