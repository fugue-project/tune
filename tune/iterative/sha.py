from typing import Optional

import fsspec
from fsspec.implementations.dirfs import DirFileSystem

from tune.concepts.flow import Trial, TrialDecision, TrialJudge, TrialReport
from tune.iterative.objective import IterativeObjectiveFunc
from tune.noniterative.objective import NonIterativeObjectiveFunc


class _NonIterativeObjectiveWrapper(NonIterativeObjectiveFunc):
    def __init__(
        self,
        func: IterativeObjectiveFunc,
        checkpoint_path: str,
        budget: float,
    ):
        super().__init__()
        self._budget = budget
        self._func = func
        self._checkpoint_path = checkpoint_path

    def generate_sort_metric(self, value: float) -> float:
        return self._func.generate_sort_metric(value)

    def run(self, trial: Trial) -> TrialReport:  # pragma: no cover
        judge = _NonIterativeJudgeWrapper(self._budget)
        base_fs, uri = fsspec.core.url_to_fs(self._checkpoint_path)
        fs = base_fs.makedirs(uri, exist_ok=True)
        cfs = DirFileSystem(path=uri, fs=fs)
        self._func = self._func.copy()
        self._func.run(trial, judge=judge, checkpoint_basedir_fs=cfs)
        return judge.report


class _NonIterativeJudgeWrapper(TrialJudge):
    def __init__(self, budget: float):
        super().__init__()
        self._report: Optional[TrialReport] = None
        self._budget = budget

    @property
    def report(self) -> TrialReport:
        assert self._report is not None
        return self._report

    def can_accept(self, trial: Trial) -> bool:
        return True

    def get_budget(self, trial: Trial, rung: int):
        return self._budget

    def judge(self, report: TrialReport) -> TrialDecision:
        self._report = report
        return TrialDecision(report, budget=0.0, should_checkpoint=True, metadata={})
