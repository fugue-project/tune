from typing import Optional

from fs.base import FS as FSBase
from tune.iterative.trial import IterativeTrial, TrialJudge
from tune.trial import Trial


class IterativeObjectiveFunc:
    def run(self, trial: IterativeTrial) -> None:
        raise NotImplementedError  # pragma: no cover


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

    def run(self, func: IterativeObjectiveFunc, trial: Trial) -> None:
        return func.run(self.make_trial(trial))
