from datetime import datetime
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
        self._stopper_updated = False
        self._lock = RLock()
        self._log_best_only = log_best_only
        self._logs: Dict[str, TrialReportCollection] = {}

    @property
    def updated(self) -> bool:
        return self._stopper_updated

    def should_stop(self, trial: Trial) -> bool:  # pragma: no cover
        return False

    def on_report(self, report: TrialReport) -> bool:
        self._stopper_updated = True
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

    def __and__(self, other: "NonIterativeStopper") -> "NonIterativeStopperCombiner":
        return NonIterativeStopperCombiner(self, other, is_and=True)

    def __or__(self, other: "NonIterativeStopper") -> "NonIterativeStopperCombiner":
        return NonIterativeStopperCombiner(self, other, is_and=False)


class NonIterativeStopperCombiner(NonIterativeStopper):
    def __init__(
        self, left: NonIterativeStopper, right: NonIterativeStopper, is_and: bool
    ):
        super().__init__()
        assert not left.updated, "can't reuse updated stopper"
        assert not right.updated, "can't reuse updated stopper"
        self._left = left
        self._right = right
        self._is_and = is_and

    def should_stop(self, trial: Trial) -> bool:  # pragma: no cover
        if self._is_and:
            return self._left.should_stop(trial) and self._right.should_stop(trial)
        else:
            return self._left.should_stop(trial) or self._right.should_stop(trial)

    def on_report(self, report: TrialReport) -> bool:
        self.monitor.on_report(report)
        left = self._left.on_report(report)
        right = self._right.on_report(report)
        return left or right

    def get_reports(self, trial: Trial) -> List[TrialReport]:  # pragma: no cover
        raise NotImplementedError


class SimpleNonIterativeStopper(NonIterativeStopper):
    def __init__(
        self,
        partition_should_stop: Callable[[TrialReport, bool, List[TrialReport]], bool],
        log_best_only: bool = False,
    ):
        super().__init__(log_best_only=log_best_only)
        self._partition_should_stop = partition_should_stop
        self._stopped: Set[str] = set()

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


def n_samples(n: int) -> SimpleNonIterativeStopper:
    def func(current: TrialReport, updated: bool, reports: List[TrialReport]):
        return len(reports) >= n

    return SimpleNonIterativeStopper(func, log_best_only=False)


def n_updates(n: int) -> SimpleNonIterativeStopper:
    def func(current: TrialReport, updated: bool, reports: List[TrialReport]):
        return len(reports) >= n

    return SimpleNonIterativeStopper(func, log_best_only=True)


def no_update_period(period: Any) -> SimpleNonIterativeStopper:
    _interval = to_timedelta(period)

    def func(current: TrialReport, updated: bool, reports: List[TrialReport]):
        if updated or len(reports) == 0:
            return False
        return datetime.now() - reports[-1].log_time > _interval

    return SimpleNonIterativeStopper(func, log_best_only=True)


def small_improvement(threshold: float, updates: int) -> SimpleNonIterativeStopper:
    assert updates > 0

    def func(current: TrialReport, updated: bool, reports: List[TrialReport]):
        if not updated:
            return False
        if len(reports) <= updates:
            return False
        diff = reports[-updates - 1].sort_metric - current.sort_metric
        return diff < threshold

    return SimpleNonIterativeStopper(func, log_best_only=True)
