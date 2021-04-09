from typing import Any, Dict, Tuple

from tune import Rand, RandInt
from tune.noniterative.convert import noniterative_objective
from tune.noniterative.objective import validate_noniterative_objective
from tune_hyperopt import HyperoptRunner
from tune.trial import Trial


def test_hyperopt():
    params = dict(a=Rand(-10.0, 10.0), b=RandInt(-100, 100), c=2.0)
    trial = Trial("a", params, metadata={})
    h = HyperoptRunner(max_iter=200, seed=0)

    @noniterative_objective
    def objective(a, b, c) -> Tuple[float, Dict[str, Any]]:
        return a ** 2 + b ** 2 + c, dict(a=1)

    def v(report):
        assert report.metric < 7
        assert report.params["a"] ** 2 < 2
        assert report.params["b"] ** 2 < 2
        assert 2.0 == report.params["c"]

    validate_noniterative_objective(objective, trial, v, runner=h)
