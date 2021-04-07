from typing import Callable, Optional
from tune.trial import Trial, TrialReport


class NonIterativeObjectiveFunc:
    def generate_sort_metric(self, value: float) -> float:  # pragma: no cover
        return value

    def run(self, trial: Trial) -> TrialReport:  # pragma: no cover
        raise NotImplementedError


class NonIterativeObjectiveRunner:
    @property
    def distributable(self) -> bool:
        return True

    def run(self, func: NonIterativeObjectiveFunc, trial: Trial) -> TrialReport:
        # TODO: how to utilize execution_engine?
        return func.run(trial)


def validate_noniterative_objective(
    func: NonIterativeObjectiveFunc,
    trial: Trial,
    validator: Callable[[TrialReport], None],
    runner: Optional[NonIterativeObjectiveRunner] = None,
) -> None:
    _runner = runner or NonIterativeObjectiveRunner()
    validator(_runner.run(func, trial))
