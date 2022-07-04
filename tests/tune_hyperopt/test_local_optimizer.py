from typing import Any

from hyperopt.early_stop import no_progress_loss
from tune import parse_noniterative_local_optimizer
from tune.noniterative.objective import NonIterativeObjectiveLocalOptimizer
from tune_test.local_optmizer import NonIterativeObjectiveLocalOptimizerTests

from tune_hyperopt import HyperoptLocalOptimizer


class HyperoptLocalOptimizerTests(NonIterativeObjectiveLocalOptimizerTests.Tests):
    def make_optimizer(self, **kwargs: Any) -> NonIterativeObjectiveLocalOptimizer:
        kwargs = {"seed": 0, "kwargs_func": _add_conf, **kwargs}
        return HyperoptLocalOptimizer(**kwargs)


def _add_conf(func, trial):
    return dict(early_stop_fn=no_progress_loss(50))


def test_parse_noniterative_local_optimizer():
    o1 = parse_noniterative_local_optimizer("hyperopt")
    assert isinstance(o1, HyperoptLocalOptimizer)
    assert o1._max_iter == 30

    o2 = parse_noniterative_local_optimizer("hyperopt:10")
    assert isinstance(o2, HyperoptLocalOptimizer)
    assert o2._max_iter == 10
