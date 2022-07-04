from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from tune._utils.math import adjust_high
from tune.concepts.flow import Trial, TrialReport
from tune.concepts.logger import make_logger
from tune.concepts.space import (
    Choice,
    Rand,
    RandInt,
    TransitionChoice,
    TuningParametersTemplate,
)
from tune.noniterative.objective import (
    NonIterativeObjectiveFunc,
    NonIterativeObjectiveLocalOptimizer,
)


class HyperoptLocalOptimizer(NonIterativeObjectiveLocalOptimizer):
    def __init__(
        self,
        max_iter: int,
        seed: int = 0,
        kwargs_func: Optional[
            Callable[[NonIterativeObjectiveFunc, Trial], Dict[str, Any]]
        ] = None,
    ):
        self._max_iter = max_iter
        self._seed = seed
        self._kwargs_func = kwargs_func

    def run(
        self, func: NonIterativeObjectiveFunc, trial: Trial, logger: Any
    ) -> TrialReport:
        template = trial.params
        if template.empty:
            tmp = NonIterativeObjectiveLocalOptimizer()
            return tmp.run(func, trial, logger=logger)
        proc = self._process(template)
        lock = RLock()
        best_report: List[TrialReport] = []

        with make_logger(logger) as p_logger:
            with p_logger.create_child(name=repr(trial)) as c_logger:

                def obj(args) -> Dict[str, Any]:
                    with c_logger.create_child(is_step=True) as s_logger:
                        params = template.fill([p[1](v) for p, v in zip(proc, args)])
                        report = func.safe_run(trial.with_params(params))
                        with lock:
                            if len(best_report) == 0:
                                best_report.append(report)
                            elif report.sort_metric < best_report[0].sort_metric:
                                best_report[0] = report
                            s_logger.log_report(best_report[0])
                        return {
                            "loss": report.sort_metric,
                            "status": STATUS_OK,
                            "report": report,
                        }

                trials = Trials()
                kwargs: Dict[str, Any] = dict(
                    algo=tpe.suggest,
                    max_evals=self._max_iter,
                    trials=trials,
                    show_progressbar=False,
                    rstate=np.random.default_rng(self._seed),
                )
                if self._kwargs_func is not None:
                    kwargs.update(self._kwargs_func(func, trial))

                fmin(obj, space=[p[0] for p in proc], **kwargs)
                report = trials.best_trial["result"]["report"]
                c_logger.log_report(
                    report, log_params=True, extract_metrics=True, log_metadata=True
                )
                return report

    def _process(self, template: TuningParametersTemplate) -> List[Tuple[Any, Any]]:
        res: List[Tuple[Any, Any]] = []
        for i, v in enumerate(template.params):
            k = f"p{i}"
            if isinstance(v, RandInt):
                res.append(_convert_randint(k, v))
            elif isinstance(v, Rand):
                res.append(_convert_rand(k, v))
            elif isinstance(v, TransitionChoice):
                res.append(_convert_transition_choice(k, v))
            elif isinstance(v, Choice):
                res.append(_convert_choice(k, v))
            else:
                # TODO: normal rand and normal randint
                raise NotImplementedError(v)  # pragma: no cover
        return res


def _convert_randint(k: str, v: RandInt) -> Any:
    _high = adjust_high(
        0, v.high - v.low, v.q, include_high=v.include_high  # type:ignore
    )
    n = int(np.round(_high / v.q))
    if not v.log:
        return hp.randint(k, 0, n) * v.q + v.low, lambda x: int(np.round(x))
    return (
        hp.qloguniform(k, np.log(v.q), np.log(_high), q=v.q) + v.low - v.q,
        lambda x: int(np.round(x)),
    )


def _convert_transition_choice(k: str, v: TransitionChoice) -> Any:
    return hp.randint(k, 0, len(v.values)), lambda x: v.values[int(np.round(x))]


def _convert_rand(k: str, v: Rand) -> Any:
    if v.q is None:
        if not v.log:
            return hp.uniform(k, v.low, v.high), lambda x: x
        else:
            return hp.loguniform(k, np.log(v.low), np.log(v.high)), lambda x: x
    else:
        _high = adjust_high(0, v.high - v.low, v.q, include_high=v.include_high)
        n = int(np.round(_high / v.q))
        if not v.log:
            return hp.randint(k, 0, n) * v.q + v.low, lambda x: x
        else:
            _high = max(0, _high - v.q)
            return hp.qloguniform(k, 0, np.log(_high), q=v.q) + v.low, lambda x: x


def _convert_choice(k: str, v: Choice) -> Any:
    return hp.choice(k, v.values), lambda x: x
