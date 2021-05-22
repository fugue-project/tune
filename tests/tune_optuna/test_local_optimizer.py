from typing import Any

from tune.noniterative.objective import NonIterativeObjectiveLocalOptimizer
from tune_test.local_optmizer import NonIterativeObjectiveLocalOptimizerTests

from tune_optuna.optimizer import OptunaLocalOptimizer


class HyperoptLocalOptimizerTests(NonIterativeObjectiveLocalOptimizerTests.Tests):
    def make_optimizer(self, **kwargs: Any) -> NonIterativeObjectiveLocalOptimizer:
        return OptunaLocalOptimizer(**kwargs)
