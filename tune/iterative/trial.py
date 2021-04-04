from typing import Optional

from tune.trial import Trial, TrialDecision, TrialReport


class TrialJudge(object):
    def __init__(self, monitor: Optional["TrialJudgeMonitor"] = None):
        self._trial_judge_monitor = monitor or TrialJudgeMonitor()
        self._trial_judge_monitor.set_judge(self)

    @property
    def monitor(self) -> "TrialJudgeMonitor":
        assert self._trial_judge_monitor is not None
        return self._trial_judge_monitor

    def can_accept(self, trial: Trial) -> bool:
        return True

    def get_budget(self, trial: Trial, rung: int) -> float:  # pragma: no cover
        return 0.0

    def judge(self, report: TrialReport) -> TrialDecision:
        return TrialDecision(report, budget=0.0, should_checkpoint=False, metadata={})


class TrialJudgeMonitor:
    def __init__(self):
        self._judge: Optional[TrialJudge] = None

    def set_judge(self, judge: TrialJudge) -> None:
        self._judge = judge

    @property
    def judge(self) -> TrialJudge:
        assert self._judge is not None
        return self._judge

    def on_report(self, report: TrialReport) -> None:  # pragma: no cover
        pass

    def on_get_budget(
        self, trial: Trial, rung: int, budget: float
    ) -> None:  # pragma: no cover
        pass

    def on_judge(self, decision: TrialDecision) -> None:  # pragma: no cover
        pass
