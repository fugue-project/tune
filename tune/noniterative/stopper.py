from threading import RLock
from typing import Dict, List, Set

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
    def __init__(self, log_best_only: bool = False):
        super().__init__(log_best_only=log_best_only)
        self._stopped: Set[str] = set()

    def partition_should_stop(
        self, latest_report: TrialReport, updated: bool, reports: List[TrialReport]
    ) -> bool:  # pragma: no cover
        return False

    def should_stop(self, trial: Trial) -> bool:
        key = str(trial.keys)
        with self._lock:
            return key in self._stopped

    def on_report(self, report: TrialReport) -> bool:
        updated = super().on_report(report)
        key = str(report.trial.keys)
        with self._lock:
            if key not in self._stopped:
                if self.partition_should_stop(
                    report, updated, self.get_reports(report.trial)
                ):
                    self._stopped.add(key)
        return updated