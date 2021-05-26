from tune.constants import TUNE_STOPPER_DEFAULT_CHECK_INTERVAL
from typing import Any, Callable, Optional

from tune._utils import run_monitored_process
from tune.concepts.flow import Trial, TrialReport


class NonIterativeObjectiveFunc:
    def generate_sort_metric(self, value: float) -> float:  # pragma: no cover
        return value

    def run(self, trial: Trial) -> TrialReport:  # pragma: no cover
        raise NotImplementedError

    def safe_run(self, trial: Trial) -> TrialReport:
        report = self.run(trial)
        return report.with_sort_metric(self.generate_sort_metric(report.metric))


class NonIterativeObjectiveLocalOptimizer:
    @property
    def distributable(self) -> bool:
        return True

    def run(self, func: NonIterativeObjectiveFunc, trial: Trial) -> TrialReport:
        # TODO: how to utilize execution_engine?
        return func.safe_run(trial)

    def run_monitored_process(
        self,
        func: NonIterativeObjectiveFunc,
        trial: Trial,
        stop_checker: Callable[[], bool],
        interval: Any = TUNE_STOPPER_DEFAULT_CHECK_INTERVAL,
    ) -> TrialReport:
        return run_monitored_process(
            self.run, [func, trial], {}, stop_checker=stop_checker, interval=interval
        )


def validate_noniterative_objective(
    func: NonIterativeObjectiveFunc,
    trial: Trial,
    validator: Callable[[TrialReport], None],
    optimizer: Optional[NonIterativeObjectiveLocalOptimizer] = None,
) -> None:
    _optimizer = optimizer or NonIterativeObjectiveLocalOptimizer()
    validator(_optimizer.run_monitored_process(func, trial, lambda: False, "1sec"))
