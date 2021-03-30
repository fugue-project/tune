from typing import Any, Optional

from fs.base import FS as FSBase
from triad import assert_or_throw
from tune.checkpoint import Checkpoint
from tune.trial import Trial, TrialDecision, TrialReport


class TrialJudge(object):
    def get_budget(self, trial: Trial, rung: int) -> float:  # pragma: no cover
        return 0.0

    def judge(self, report: TrialReport) -> TrialDecision:
        return TrialDecision(report, budget=0.0, should_checkpoint=False, metadata={})


class IterativeTrial(Trial):
    def __init__(
        self,
        trial: Trial,
        judge: TrialJudge,
        checkpoint_basedir_fs: Optional[FSBase] = None,
    ):
        super().__init__(**trial.jsondict)
        self._judge = judge
        self._checkpoint_basedir_fs = checkpoint_basedir_fs
        self._checkpoint: Any = None

    def with_judge(self, judge: TrialJudge) -> "IterativeTrial":
        return IterativeTrial(self, judge, self._checkpoint_basedir_fs)

    def with_checkpoint_basedir_fs(
        self, checkpoint_basedir_fs: Optional[FSBase]
    ) -> "IterativeTrial":
        return IterativeTrial(self, self._judge, checkpoint_basedir_fs)

    @property
    def has_checkpoint(self) -> bool:
        return self._checkpoint_basedir_fs is not None

    @property
    def checkpoint(self) -> Checkpoint:
        if self._checkpoint is None:
            assert_or_throw(
                self._checkpoint_basedir_fs is not None, "unable to setup checkpoint"
            )
            self._checkpoint = Checkpoint(
                self._checkpoint_basedir_fs.makedir(  # type: ignore
                    self.trial_id, recreate=True
                )
            )
        return self._checkpoint

    @property
    def iteration(self) -> int:
        return len(self.checkpoint) if self.has_checkpoint else 0

    @property
    def judge(self) -> TrialJudge:
        return self._judge
