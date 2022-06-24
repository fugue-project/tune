from typing import Any, Callable, Optional

from tune._utils import run_monitored_process
from tune.concepts.flow import Trial, TrialReport
from tune.concepts.logger import make_logger
from tune.constants import TUNE_STOPPER_DEFAULT_CHECK_INTERVAL


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

    def run(
        self, func: NonIterativeObjectiveFunc, trial: Trial, logger: Any
    ) -> TrialReport:
        # TODO: how to utilize execution_engine?
        if logger is None:
            report = func.safe_run(trial)
        else:
            with make_logger(logger) as p_logger:
                with p_logger.create_child(name=repr(trial)) as c_logger:
                    report = func.safe_run(trial)
                    c_logger.log_report(
                        report, log_params=True, extract_metrics=True, log_metadata=True
                    )
        return report

    def run_monitored_process(
        self,
        func: NonIterativeObjectiveFunc,
        trial: Trial,
        stop_checker: Callable[[], bool],
        logger: Any,
        interval: Any = TUNE_STOPPER_DEFAULT_CHECK_INTERVAL,
    ) -> TrialReport:
        return run_monitored_process(
            self.run,
            [func, trial],
            {"logger": logger},
            stop_checker=stop_checker,
            interval=interval,
        )


def validate_noniterative_objective(
    func: NonIterativeObjectiveFunc,
    trial: Trial,
    validator: Callable[[TrialReport], None],
    optimizer: Optional[NonIterativeObjectiveLocalOptimizer] = None,
    logger: Any = None,
) -> None:
    _optimizer = optimizer or NonIterativeObjectiveLocalOptimizer()
    validator(
        _optimizer.run_monitored_process(
            func, trial, lambda: False, interval="1sec", logger=logger
        )
    )
