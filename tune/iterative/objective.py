from fs.base import FS as FSBase
from tune.iterative.trial import IterativeTrial
from tune.trial import Trial, TrialReport


class IterativeObjectiveFunc:
    def generate_sort_metric(self, value: float) -> float:
        return value

    def run(self, trial: IterativeTrial) -> None:
        raise NotImplementedError  # pragma: no cover

    def copy(self) -> "IterativeObjectiveFunc":  # pragma: no cover
        raise NotImplementedError


class MultiRungObjectiveFunc(IterativeObjectiveFunc):
    def __init__(self):
        self._rung = 0

    @property
    def rung(self) -> int:
        return self._rung

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
                return (
                    current_report.with_cost(used)
                    .with_rung(self.rung)
                    .with_sort_metric(
                        self.generate_sort_metric(current_report.sort_metric)
                    )
                )

    def run(self, trial: IterativeTrial) -> None:
        self.preprocess()
        if trial.iteration > 0:
            self._rung = int(trial.checkpoint.latest.readtext("__RUNG__")) + 1
            self.load_checkpoint(trial.checkpoint.latest)
        budget = trial.judge.get_budget(trial, self.rung)
        while budget > 0:
            report = self.run_single_rung(trial, budget)
            decision = trial.judge.judge(report)
            if decision.should_checkpoint and trial.has_checkpoint:
                with trial.checkpoint.create() as fs:
                    fs.writetext("__RUNG__", str(self.rung))
                    self.save_checkpoint(fs)
            budget = decision.budget
            self._rung += 1
        self.postprocess()
