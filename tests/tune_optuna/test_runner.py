from typing import Any, Dict, Tuple

from hyperopt.early_stop import no_progress_loss
from tune import Rand, RandInt
from tune.concepts.flow import Trial
from tune.noniterative.convert import noniterative_objective
from tune.noniterative.objective import validate_noniterative_objective

from tune_optuna.optimizer import OptunaLocalOptimizer


def test_optuna():
    params = dict(a=Rand(-10.0, 10.0), b=RandInt(-100, 100), c=2.0)
    trial = Trial("a", params, metadata={})
    h = OptunaLocalOptimizer(max_iter=200)

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

    validate_noniterative_objective(objective, trial, v, optimizer=h)
