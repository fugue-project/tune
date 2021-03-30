from tune.trial import TrialDecision, TrialReport
from tune.iterative.trial import TrialJudge
from typing import List
from threading import RLock


class PerTrialJudge:
    def __init__(self, schedule: List[float]):
        self._schedule = schedule
        self._reports: List[TrialReport] = []

    def report(self, report: TrialReport) -> None:
        self._reports.append(report)


class ASHAJudge(TrialJudge):
    def __init__(self, total_budget: float, per_trial_budget_schedule: List[float]):
        self._lock = RLock()
        self._budget_used = 0.0

    def judge(self, report: TrialReport) -> TrialDecision:
        return super().judge(report)
