from typing import Any, Dict, Tuple

import numpy as np
from triad import assert_or_throw
from tune.space.parameters import Choice, Rand, RandInt, StochasticExpression
from tune.trial.objective import ObjectiveRunner, Objective
from tune.trial.trial import ReportHandler, Trial

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


class HyperoptRunner(ObjectiveRunner):
    class Handler(ReportHandler):
        def __init__(self, data: Dict[str, Any]):
            self._data = data

        def __call__(
            self,
            trial: Trial,
            params: Dict[str, Any],
            metric: float,
            metadata: Dict[str, Any],
        ):
            assert_or_throw("loss" not in self._data, "can't report multiple times")
            self._data["loss"] = metric
            self._data["status"] = STATUS_OK
            self._data["metric"] = metric
            self._data["static_params"] = params
            self._data["metadata"] = metadata

    def __init__(self, max_iter: int, seed: int = 0):
        self._max_iter = max_iter
        self._seed = seed

    def run(self, objective: Objective, trial: Trial) -> None:
        static_params, stochastic_params = self._split(trial.params)
        stochastic_keys = list(stochastic_params.keys())
        if len(stochastic_keys) == 0:
            return super().run(objective, trial)

        def obj(args) -> Dict[str, Any]:
            params = {k: v for k, v in zip(stochastic_keys, args)}
            params.update(static_params)
            result: Dict[str, Any] = {}
            handler = HyperoptRunner.Handler(result)
            static_trial = (
                trial.with_params(params)
                .with_checkpoint_basedir_fs(None)
                .with_report_handler(handler)
            )
            objective(static_trial)
            assert_or_throw("loss" in result, "must report once and only once")
            return result

        trials = Trials()
        fmin(
            obj,
            space=list(stochastic_params.values()),
            algo=tpe.suggest,
            max_evals=self._max_iter,
            trials=trials,
            show_progressbar=False,
            rstate=np.random.RandomState(self._seed),
        )

        trial.report(
            metric=trials.best_trial["result"]["metric"],
            params=trials.best_trial["result"]["static_params"],
            metadata=trials.best_trial["result"]["metadata"],
        )

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
