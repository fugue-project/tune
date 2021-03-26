from typing import Any, Callable, Dict, Optional

from fs.base import FS as FSBase
from triad import assert_or_throw
from tune.checkpoint import Checkpoint
from tune.trial import Trial, TrialDecision, TrialReport


class TrialJudge(object):
    def judge(self, report: TrialReport) -> TrialDecision:
        return TrialDecision(
            report, should_stop=True, should_checkpoint=False, metadata={}
        )


class IterativeTrial(Trial):
    def __init__(
        self,
        trial_id: str,
        params: Dict[str, Any],
        metadata: Dict[str, Any],
        judge: TrialJudge,
        checkpoint_basedir_fs: Optional[FSBase] = None,
    ):
        super().__init__(trial_id=trial_id, params=params, metadata=metadata)
        self._judge = judge
        self._checkpoint_basedir_fs = checkpoint_basedir_fs
        self._checkpoint: Any = None

    def with_judge(self, maker: TrialJudge) -> "IterativeTrial":
        return IterativeTrial(
            self.trial_id,
            self.params,
            self.metadata,
            maker,
            self._checkpoint_basedir_fs,
        )

    def with_checkpoint_basedir_fs(
        self, checkpoint_basedir_fs: Optional[FSBase]
    ) -> "IterativeTrial":
        return IterativeTrial(
            self.trial_id,
            self.params,
            self.metadata,
            self._judge,
            checkpoint_basedir_fs,
        )

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

    def report(
        self,
        metric: Any,
        params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        save_checkpoint: Optional[Callable[[FSBase], None]] = None,
    ) -> TrialDecision:
        _trial = Trial(
            trial_id=self.trial_id, params=self.params, metadata=self.metadata
        )
        _report = TrialReport(
            _trial,
            metric=float(metric),
            params=params or self.params,
            metadata=metadata or {},
        )
        decision = self._judge.judge(_report)
        if decision.should_checkpoint and save_checkpoint is not None:
            with self.checkpoint.create() as fs:
                save_checkpoint(fs)
        return decision
