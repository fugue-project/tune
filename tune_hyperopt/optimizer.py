from tune._utils.math import adjust_high
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from tune.noniterative.objective import (
    NonIterativeObjectiveFunc,
    NonIterativeObjectiveLocalOptimizer,
)
from tune.concepts.space.parameters import Choice, Rand, RandInt, StochasticExpression
from tune.concepts.flow import Trial, TrialReport


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

    def run(self, func: NonIterativeObjectiveFunc, trial: Trial) -> TrialReport:
        static_params, stochastic_params, postprocess = self._split(trial.params)
        stochastic_keys = list(stochastic_params.keys())
        if len(stochastic_keys) == 0:
            return func.run(trial)

        def obj(args) -> Dict[str, Any]:
            params = {k: postprocess[k](v) for k, v in zip(stochastic_keys, args)}
            params.update(static_params)
            report = func.run(trial.with_params(params))
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
            rstate=np.random.RandomState(self._seed),
        )
        if self._kwargs_func is not None:
            kwargs.update(self._kwargs_func(func, trial))

        fmin(obj, space=list(stochastic_params.values()), **kwargs)
        return trials.best_trial["result"]["report"]

    def _split(
        self, kwargs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        static_params: Dict[str, Any] = {}
        stochastic_params: Dict[str, Any] = {}
        postprocess: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if isinstance(v, StochasticExpression):
                if isinstance(v, RandInt):
                    stochastic_params[k], postprocess[k] = _convert_randint(k, v)
                elif isinstance(v, Rand):
                    stochastic_params[k], postprocess[k] = _convert_rand(k, v)
                elif isinstance(v, Choice):
                    stochastic_params[k], postprocess[k] = _convert_choice(k, v)
                else:
                    # TODO: normal rand and normal randint
                    raise NotImplementedError(v)  # pragma: no cover
            else:
                static_params[k] = v
        return static_params, stochastic_params, postprocess


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
