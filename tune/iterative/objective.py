from fs.base import FS as FSBase
from tune.checkpoint import Checkpoint
from tune.iterative.trial import TrialJudge
from tune.trial import Trial, TrialReport


class IterativeObjectiveFunc:
    def __init__(self):
        self._rung = 0

    def copy(self) -> "IterativeObjectiveFunc":  # pragma: no cover
        raise NotImplementedError

    @property
    def rung(self) -> int:
        return self._rung

    def generate_sort_metric(self, value: float) -> float:
        return value

    def load_checkpoint(self, fs: FSBase) -> None:  # pragma: no cover
        return

    def save_checkpoint(self, fs: FSBase) -> None:  # pragma: no cover
        return

    def preprocess(self) -> None:  # pragma: no cover
        return

    def postprocess(self) -> None:  # pragma: no cover
        return

    def run_single_iteration(self, trial: Trial) -> TrialReport:  # pragma: no cover
        raise NotImplementedError

    def run_single_rung(self, trial: Trial, budget: float) -> TrialReport:
        used = 0.0
        while True:
            current_report = self.run_single_iteration(trial)
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
        self.preprocess()
        if len(checkpoint) > 0:
            self._rung = int(checkpoint.latest.readtext("__RUNG__")) + 1
            self.load_checkpoint(checkpoint.latest)
        budget = judge.get_budget(trial, self.rung)
        while budget > 0:
            report = self.run_single_rung(trial, budget)
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
        self.postprocess()
