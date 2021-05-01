from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from tune.noniterative.objective import (
    NonIterativeObjectiveFunc,
    NonIterativeObjectiveRunner,
)
from tune.space.parameters import Choice, Rand, RandInt, StochasticExpression
from tune.trial import Trial, TrialReport


class HyperoptRunner(NonIterativeObjectiveRunner):
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
        static_params, stochastic_params = self._split(trial.params)
        stochastic_keys = list(stochastic_params.keys())
        if len(stochastic_keys) == 0:
            return func.run(trial)

        def obj(args) -> Dict[str, Any]:
            params = {k: v for k, v in zip(stochastic_keys, args)}
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

    def _split(self, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        static_params: Dict[str, Any] = {}
        stochastic_params: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if isinstance(v, StochasticExpression):
                if isinstance(v, RandInt):
                    stochastic_params[k] = _convert_randint(k, v)
                elif isinstance(v, Rand):
                    stochastic_params[k] = _convert_rand(k, v)
                elif isinstance(v, Choice):
                    stochastic_params[k] = _convert_choice(k, v)
                else:
                    raise NotImplementedError(v)  # pragma: no cover
            else:
                static_params[k] = v
        return static_params, stochastic_params


def _convert_randint(k: str, v: RandInt) -> Any:
    if not v.log:
        return hp.randint(k, v.low, v.high + (1 if v.include_high else 0))
    raise NotImplementedError(k, v)  # pragma: no cover


def _convert_rand(k: str, v: Rand) -> Any:
    if v.q is None:
        if not v.log:
            return hp.uniform(k, v.low, v.high)
        else:
            return hp.loguniform(k, v.low, v.high)
    raise NotImplementedError(k, v)  # pragma: no cover


def _convert_choice(k: str, v: Choice) -> Any:
    return hp.choice(k, v.values)
