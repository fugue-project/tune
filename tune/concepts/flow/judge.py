from typing import Any, Callable, Dict, Optional

from tune.concepts.flow.report import TrialReport
from tune.concepts.flow.trial import Trial


class TrialDecision:
    def __init__(
        self,
        report: TrialReport,
        budget: float,
        should_checkpoint: bool,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._report = report
        self._budget = budget
        self._should_checkpoint = should_checkpoint
        self._reason = reason
        self._metadata = metadata or {}

    def __repr__(self) -> str:
        return repr(
            dict(
                report=self._report,
                budget=self._budget,
                should_checkpoint=self._should_checkpoint,
                reason=self._reason,
                metadata=self._metadata,
            )
        )

    def __copy__(self) -> "TrialDecision":
        return self

    def __deepcopy__(self, memo: Any) -> "TrialDecision":
        return self

    @property
    def report(self) -> TrialReport:
        return self._report

    @property
    def trial(self) -> Trial:
        return self.report.trial

    @property
    def trial_id(self) -> str:
        return self.trial.trial_id

    @property
    def budget(self) -> float:
        return self._budget

    @property
    def should_stop(self) -> bool:
        return self.budget <= 0

    @property
    def should_checkpoint(self) -> bool:
        return self._should_checkpoint

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata


class TrialJudge:
    def __init__(self, monitor: Optional["Monitor"] = None):
        self.reset_monitor(monitor)

    @property
    def monitor(self) -> "Monitor":
        assert self._trial_judge_monitor is not None
        return self._trial_judge_monitor

    def reset_monitor(self, monitor: Optional["Monitor"] = None) -> None:
        self._trial_judge_monitor = monitor or Monitor()

    def can_accept(self, trial: Trial) -> bool:  # pragma: no cover
        raise NotImplementedError

    def get_budget(self, trial: Trial, rung: int) -> float:  # pragma: no cover
        raise NotImplementedError

    def judge(self, report: TrialReport) -> TrialDecision:  # pragma: no cover
        raise NotImplementedError


class RemoteTrialJudge(TrialJudge):
    def __init__(self, entrypoint: Callable[[str, Dict[str, Any]], Any]):
        super().__init__()
        self._entrypoint = entrypoint
        self._report: Optional[TrialReport] = None

    @property
    def report(self) -> Optional[TrialReport]:
        return self._report

    def can_accept(self, trial: Trial) -> bool:
        return self._entrypoint("can_accept", dict(trial=trial))

    def judge(self, report: TrialReport) -> TrialDecision:
        self._report = report
        return self._entrypoint("judge", dict(report=report))

    def get_budget(self, trial: Trial, rung: int) -> float:
        return self._entrypoint("get_budget", dict(trial=trial, rung=rung))


class NoOpTrailJudge(TrialJudge):
    def can_accept(self, trial: Trial) -> bool:  # pragma: no cover
        return True

    def get_budget(self, trial: Trial, rung: int) -> float:  # pragma: no cover
        return 0.0

    def judge(self, report: TrialReport) -> TrialDecision:  # pragma: no cover
        self.monitor.on_report(report)
        return TrialDecision(report, 0.0, False)


class TrialCallback:
    def __init__(self, judge: TrialJudge):
        self._judge = judge

    def entrypoint(self, name, kwargs: Dict[str, Any]) -> Any:
        if name == "can_accept":
            return self._judge.can_accept(kwargs["trial"])
        if name == "judge":
            return self._judge.judge(kwargs["report"])
        if name == "get_budget":
            return self._judge.get_budget(kwargs["trial"], kwargs["rung"])
        raise NotImplementedError  # pragma: no cover


class Monitor:
    def __init__(self):
        self._judge: Optional[TrialJudge] = None

    def on_report(self, report: TrialReport) -> None:  # pragma: no cover
        pass

    def on_get_budget(
        self, trial: Trial, rung: int, budget: float
    ) -> None:  # pragma: no cover
        pass

    def on_judge(self, decision: TrialDecision) -> None:  # pragma: no cover
        pass

    def initialize(self) -> None:  # pragma: no cover
        pass

    def finalize(self) -> None:  # pragma: no cover
        pass
