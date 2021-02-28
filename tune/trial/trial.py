from typing import Any, Callable, Dict, Optional

from fs.base import FS as FSBase
from triad import ParamDict, assert_or_throw

from tune.checkpoint import Checkpoint


class ReportFeedback(object):
    def __init__(self, should_stop: bool, should_checkpoint: bool, **kwargs: Any):
        self._should_stop = should_stop
        self._should_checkpoint = should_checkpoint
        self._metadata = ParamDict(kwargs)

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def should_checkpoint(self) -> bool:
        return self._should_checkpoint

    @property
    def metadata(self) -> ParamDict:
        return self._metadata


class ReportHandler(object):
    def __call__(
        self,
        trial: "Trial",
        params: Dict[str, Any],
        metric: float,
        metadata: Dict[str, Any],
    ) -> ReportFeedback:
        return ReportFeedback(should_stop=True, should_checkpoint=False)


class Trial(object):
    def __init__(
        self,
        trial_id: str,
        params: Dict[str, Any],
        metadata: Dict[str, Any],
        report_handler: ReportHandler,
        checkpoint_basedir_fs: Optional[FSBase] = None,
    ):
        self._trial_id = trial_id
        self._report_handler = report_handler
        self._params = ParamDict(params)
        self._metadata = ParamDict(metadata)
        self._checkpoint_basedir_fs = checkpoint_basedir_fs
        self._checkpoint: Any = None

    @property
    def trial_id(self) -> str:
        return self._trial_id

    @property
    def params(self) -> ParamDict:
        return self._params

    def with_params(self, params: Dict[str, Any]) -> "Trial":
        return Trial(
            self.trial_id,
            params,
            self.metadata,
            self._report_handler,
            self._checkpoint_basedir_fs,
        )

    def with_report_handler(self, handler: ReportHandler) -> "Trial":
        return Trial(
            self.trial_id,
            self.params,
            self.metadata,
            handler,
            self._checkpoint_basedir_fs,
        )

    def with_checkpoint_basedir_fs(
        self, checkpoint_basedir_fs: Optional[FSBase]
    ) -> "Trial":
        return Trial(
            self.trial_id,
            self.params,
            self.metadata,
            self._report_handler,
            checkpoint_basedir_fs,
        )

    @property
    def metadata(self) -> ParamDict:
        return self._metadata

    @property
    def stateful(self) -> bool:
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
        return len(self.checkpoint) if self.stateful else 0

    def report(
        self,
        metric: Any,
        params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        save_checkpoint: Optional[Callable[[FSBase], None]] = None,
    ) -> ReportFeedback:
        _metric = float(metric)
        _metadata = ParamDict(metadata)
        feedback = self._report_handler(self, params or self.params, _metric, _metadata)
        if self.stateful and feedback.should_checkpoint and save_checkpoint is not None:
            with self.checkpoint.create() as fs:
                save_checkpoint(fs)
        return feedback
