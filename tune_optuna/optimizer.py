from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna
from optuna.study import Study
from tune import (
    NonIterativeObjectiveFunc,
    NonIterativeObjectiveLocalOptimizer,
    Rand,
    RandInt,
    StochasticExpression,
    Trial,
    TrialReport,
)
from tune._utils.math import _IGNORABLE_ERROR, uniform_to_discrete, uniform_to_integers
from tune.concepts.space.parameters import Choice


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
        if isinstance(v, RandInt):
            if v.log and v.q is not None:
                value = trial.suggest_float(name=k, low=0, high=1.0)
                result[k] = uniform_to_integers(
                    value,
                    low=v.low,
                    high=v.high,
                    q=v.q,  # type: ignore
                    log=True,
                    include_high=v.include_high,
                )
            else:
                _high: Any = v.high if v.include_high else v.high - 1
                result[k] = trial.suggest_int(
                    name=k, low=v.low, high=_high, step=v.q, log=v.log
                )
        elif isinstance(v, Rand):
            if v.log and v.q is not None:
                value = trial.suggest_float(name=k, low=0, high=1.0)
                result[k] = uniform_to_discrete(
                    value,
                    low=v.low,
                    high=v.high,
                    q=v.q,
                    log=True,
                    include_high=v.include_high,
                )
            else:
                _high = v.high
                if v.q is not None and not v.include_high:
                    _high -= _IGNORABLE_ERROR
                result[k] = trial.suggest_float(
                    name=k, low=v.low, high=_high, step=v.q, log=v.log
                )
        elif isinstance(v, Choice):
            result[k] = trial.suggest_categorical(name=k, choices=v.values)
        else:  # pragma: no cover
            raise NotImplementedError
    return result
