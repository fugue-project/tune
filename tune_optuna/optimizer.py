from threading import RLock
from typing import Any, Callable, Dict, List, Optional

import optuna
from optuna.study import Study
from tune import (
    Choice,
    NonIterativeObjectiveFunc,
    NonIterativeObjectiveLocalOptimizer,
    Rand,
    RandInt,
    TransitionChoice,
    Trial,
    TrialReport,
)
from tune._utils.math import _IGNORABLE_ERROR, uniform_to_discrete, uniform_to_integers
from tune.concepts.space import TuningParametersTemplate


class OptunaLocalOptimizer(NonIterativeObjectiveLocalOptimizer):
    def __init__(
        self, max_iter: int, create_study: Optional[Callable[[], Study]] = None
    ):
        self._max_iter = max_iter
        self._create_study = create_study or optuna.create_study

    def run(self, func: NonIterativeObjectiveFunc, trial: Trial) -> TrialReport:
        template = trial.params
        if len(template.params) == 0:
            return func.run(trial)
        lock = RLock()
        best_report: List[TrialReport] = []

        def obj(otrial: optuna.trial.Trial) -> float:
            params = template.fill_dict(_convert(otrial, template))
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


def _convert(
    trial: optuna.trial.Trial, template: TuningParametersTemplate
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for k, v in template.params_dict.items():
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
        elif isinstance(v, TransitionChoice):
            result[k] = v.values[
                trial.suggest_int(name=k, low=0, high=len(v.values) - 1)
            ]
        elif isinstance(v, Choice):
            result[k] = trial.suggest_categorical(name=k, choices=v.values)
        else:  # pragma: no cover
            raise NotImplementedError
    return result
