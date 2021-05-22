from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna
from optuna.study import Study
from tune.concepts.flow import Trial, TrialReport
from tune.concepts.space.parameters import Rand, StochasticExpression
from tune.noniterative.objective import (
    NonIterativeObjectiveFunc,
    NonIterativeObjectiveLocalOptimizer,
)


class OptunaLocalOptimizer(NonIterativeObjectiveLocalOptimizer):
    def __init__(
        self, max_iter: int, create_study: Optional[Callable[[], Study]] = None
    ):
        self._max_iter = max_iter
        self._create_study = create_study or optuna.create_study

    def run(self, func: NonIterativeObjectiveFunc, trial: Trial) -> TrialReport:
        static_params, stochastic_params = _split(trial.params)
        if len(stochastic_params) == 0:
            return func.run(trial)
        lock = RLock()
        best_report: List[TrialReport] = []

        def obj(otrial: optuna.trial.Trial) -> float:
            params = _convert(otrial, stochastic_params)
            params.update(static_params)
            report = func.run(trial.with_params(params))
            with lock:
                if len(best_report) == 0:
                    best_report.append(report)
                elif report.sort_metric < best_report[0].sort_metric:
                    best_report[0] = report
            return report.sort_metric

        study = self._create_study()
        study.optimize(obj, n_trials=self._max_iter)
        assert 1 == len(best_report)
        return best_report[0]


def _split(
    kwargs: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, StochasticExpression]]:
    static_params: Dict[str, Any] = {}
    stochastic_params: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, StochasticExpression):
            stochastic_params[k] = v
        else:
            static_params[k] = v
    return static_params, stochastic_params


def _convert(
    trial: optuna.trial.Trial, params: Dict[str, StochasticExpression]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, Rand):
            result[k] = trial.suggest_float(
                name=k, low=v.low, high=v.high, step=v.q, log=v.log
            )
        else:
            raise NotImplementedError
    return result
