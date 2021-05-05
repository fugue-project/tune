from datetime import datetime, timedelta
from threading import RLock
from typing import Any, Callable, Dict, List, Set

from triad.utils.convert import to_timedelta
from tune.concepts.flow import (
    Trial,
    TrialDecision,
    TrialJudge,
    TrialReport,
    TrialReportLogger,
)
from tune.constants import TUNE_STOPPER_DEFAULT_CHECK_INTERVAL


class TrialReportCollection(TrialReportLogger):
    def __init__(self, new_best_only: bool = False):
        super().__init__(new_best_only=new_best_only)
        self._reports: List[TrialReport] = []

    def log(self, report: TrialReport) -> None:
        self._reports.append(report.reset_log_time())

    @property
    def reports(self) -> List[TrialReport]:
        with self._lock:
            return list(self._reports)


class NonIterativeStopper(TrialJudge):
    def __init__(self, log_best_only: bool = False):
        super().__init__()
        self._lock = RLock()
        self._log_best_only = log_best_only
        self._logs: Dict[str, TrialReportCollection] = {}

    def should_stop(self, trial: Trial) -> bool:  # pragma: no cover
        return False

    def on_report(self, report: TrialReport) -> bool:
        self.monitor.on_report(report)
        with self._lock:
            key = str(report.trial.keys)
            if key not in self._logs:
                self._logs[key] = TrialReportCollection(self._log_best_only)
            return self._logs[key].on_report(report)

    def can_accept(self, trial: Trial) -> bool:
        return not self.should_stop(trial)

    def judge(self, report: TrialReport) -> TrialDecision:
        self.on_report(report)
        return TrialDecision(report, 0.0, False)

    def get_reports(self, trial: Trial) -> List[TrialReport]:
        with self._lock:
            key = str(trial.keys)
            if key not in self._logs:
                return []
            v = self._logs[key]
        return v.reports


class SimpleNonIterativeStopper(NonIterativeStopper):
    def __init__(
        self,
        partition_should_stop: Callable[[TrialReport, bool, List[TrialReport]], bool],
        check_interval: Any = TUNE_STOPPER_DEFAULT_CHECK_INTERVAL,
        log_best_only: bool = False,
    ):
        super().__init__(log_best_only=log_best_only)
        self._partition_should_stop = partition_should_stop
        self._interval = to_timedelta(check_interval)
        self._stopped: Set[str] = set()

    @property
    def check_interval(self) -> timedelta:
        return self._interval

    def should_stop(self, trial: Trial) -> bool:
        key = str(trial.keys)
        with self._lock:
            return key in self._stopped

    def on_report(self, report: TrialReport) -> bool:
        updated = super().on_report(report)
        key = str(report.trial.keys)
        with self._lock:
            if key not in self._stopped:
                if self._partition_should_stop(
                    report, updated, self.get_reports(report.trial)
                ):
                    self._stopped.add(key)
        return updated


def stop_on_n_reports(
    n: int, check_interval: Any = TUNE_STOPPER_DEFAULT_CHECK_INTERVAL
) -> SimpleNonIterativeStopper:
    def func(current: TrialReport, updated: bool, reports: List[TrialReport]):
        return len(reports) >= n

    return SimpleNonIterativeStopper(
        func, check_interval=check_interval, log_best_only=False
    )


def stop_on_no_update_period(
    period: Any, check_interval: Any = TUNE_STOPPER_DEFAULT_CHECK_INTERVAL
) -> SimpleNonIterativeStopper:
    _interval = to_timedelta(period)

    def func(current: TrialReport, updated: bool, reports: List[TrialReport]):
        if updated or len(reports) == 0:
            return False
        return datetime.now() - reports[-1].log_time > _interval

    return SimpleNonIterativeStopper(
        func, check_interval=check_interval, log_best_only=True
    )
