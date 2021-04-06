import os
import tempfile
from typing import Callable, List, Optional
from uuid import uuid4

from cloudpickle import pickle
from fs.base import FS as FSBase
from triad.collections.fs import FileSystem
from tune.checkpoint import Checkpoint
from tune.trial import Monitor, Trial, TrialDecision, TrialJudge, TrialReport


class IterativeObjectiveFunc:
    def __init__(self):
        self._rung = 0
        self._current_trial: Optional[Trial] = None

    def copy(self) -> "IterativeObjectiveFunc":  # pragma: no cover
        raise NotImplementedError

    @property
    def current_trial(self) -> Trial:
        assert self._current_trial is not None
        return self._current_trial

    @property
    def rung(self) -> int:
        return self._rung

    def generate_sort_metric(self, value: float) -> float:
        return value

    def load_checkpoint(self, fs: FSBase) -> None:  # pragma: no cover
        return

    def save_checkpoint(self, fs: FSBase) -> None:  # pragma: no cover
        return

    def initialize(self) -> None:  # pragma: no cover
        return

    def finalize(self) -> None:  # pragma: no cover
        return

    def run_single_iteration(self) -> TrialReport:  # pragma: no cover
        raise NotImplementedError

    def run_single_rung(self, budget: float) -> TrialReport:
        used = 0.0
        while True:
            current_report = self.run_single_iteration()
            used += current_report.cost
            if used >= budget:
                return current_report.with_cost(used)

    def run(
        self,
        trial: Trial,
        judge: TrialJudge,
        checkpoint_basedir_fs: FSBase,
    ) -> None:
        checkpoint = Checkpoint(
            checkpoint_basedir_fs.makedir(trial.trial_id, recreate=True)
        )
        if not judge.can_accept(trial):
            return
        self._current_trial = trial
        self.initialize()
        try:
            if len(checkpoint) > 0:
                self._rung = int(checkpoint.latest.readtext("__RUNG__")) + 1
                self.load_checkpoint(checkpoint.latest)
            budget = judge.get_budget(trial, self.rung)
            while budget > 0:
                report = self.run_single_rung(budget)
                report = report.with_rung(self.rung).with_sort_metric(
                    self.generate_sort_metric(report.metric)
                )
                decision = judge.judge(report)
                if decision.should_checkpoint:
                    with checkpoint.create() as fs:
                        fs.writetext("__RUNG__", str(self.rung))
                        self.save_checkpoint(fs)
                budget = decision.budget
                self._rung += 1
        finally:
            self.finalize()


def validate_iterative_objective(
    func: IterativeObjectiveFunc,
    trial: Trial,
    budgets: List[float],
    validator: Callable[[List[TrialReport]], None],
    continuous: bool = False,
    checkpoint_path: str = "",
    monitor: Optional[Monitor] = None,
) -> None:
    path = checkpoint_path if checkpoint_path != "" else tempfile.gettempdir()
    basefs = FileSystem().makedirs(os.path.join(path, str(uuid4())), recreate=True)
    j = _Validator(monitor, budgets, continuous=continuous)
    if continuous:
        f = pickle.loads(pickle.dumps(func)).copy()
        f.run(trial, j, checkpoint_basedir_fs=basefs)
    else:
        for _ in budgets:
            f = pickle.loads(pickle.dumps(func)).copy()
            f.run(trial, j, checkpoint_basedir_fs=basefs)
    validator(j.reports)


class _Validator(TrialJudge):
    def __init__(
        self, monitor: Optional[Monitor], budgets: List[float], continuous: bool
    ):
        super().__init__(monitor)
        self._budgets = budgets
        self._continuous = continuous
        self._reports: List[TrialReport] = []

    @property
    def reports(self) -> List[TrialReport]:
        return self._reports

    def can_accept(self, trial: Trial) -> bool:

        return True

    def get_budget(self, trial: Trial, rung: int) -> float:
        budget = self._budgets[rung] if rung < len(self._budgets) else 0.0
        self.monitor.on_get_budget(trial, rung, budget)
        return budget

    def judge(self, report: TrialReport) -> TrialDecision:
        self.monitor.on_report(report)
        self._reports.append(report)
        decision = TrialDecision(
            report,
            budget=self.get_budget(report.trial, report.rung + 1)
            if self._continuous
            else 0.0,
            should_checkpoint=report.rung >= len(self._budgets)
            if self._continuous
            else True,
        )
        self.monitor.on_judge(decision)
        return decision
