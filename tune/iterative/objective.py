from typing import Optional

from fs.base import FS as FSBase
from tune.iterative.trial import IterativeTrial, TrialJudge
from tune.trial import Trial, TrialReport


class IterativeObjectiveFunc:
    @property
    def min_better(self) -> bool:
        return True  # pragma: no cover

    def run(self, trial: IterativeTrial, budget: float) -> None:
        raise NotImplementedError  # pragma: no cover

    def copy(self) -> "IterativeObjectiveFunc":
        raise NotImplementedError


class MultiIterationObjectiveFunc(IterativeObjectiveFunc):
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

    def run(self, trial: IterativeTrial, budget: float) -> None:
        self.preprocess()
        if trial.iteration > 0:
            self.load_checkpoint(trial.checkpoint.latest)
        while budget > 0:
            report = self.run_single_iteration(trial)
            budget -= report.cost
            decision = trial.report(report.metric, report.params, report.metadata)
            if decision.should_checkpoint and trial.has_checkpoint:
                with trial.checkpoint.create() as fs:
                    self.save_checkpoint(fs)
            if decision.should_stop:
                break
        self.postprocess()


# TODO: remove?
class IterativeObjectiveRunner:
    def __init__(
        self,
        judge: TrialJudge,
        checkpoint_basedir_fs: Optional[FSBase] = None,
    ):
        self._judge = judge
        self._checkpoint_basedir_fs = checkpoint_basedir_fs

    def make_trial(self, trial: Trial) -> IterativeTrial:
        return IterativeTrial(
            trial_id=trial.trial_id,
            params=trial.params,
            metadata=trial.metadata,
            judge=self._judge,
            checkpoint_basedir_fs=self._checkpoint_basedir_fs,
        )

    def run(self, func: IterativeObjectiveFunc, trial: Trial, budget: float) -> None:
        return func.run(self.make_trial(trial), budget=budget)
