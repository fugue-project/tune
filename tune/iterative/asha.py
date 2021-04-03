from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Tuple

from triad import to_uuid
from tune.iterative.trial import TrialJudge, TrialJudgeMonitor
from tune.trial import Trial, TrialDecision, TrialReport, TrialReportHeap


class RungHeap:
    def __init__(self, n: int):
        self._lock = RLock()
        self._n = n
        self._heap = TrialReportHeap(min_heap=False)
        self._bests: List[float] = []

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)

    @property
    def capacity(self) -> int:
        return self._n

    @property
    def best(self) -> float:
        with self._lock:
            return self._bests[-1] if len(self._bests) > 0 else float("nan")

    @property
    def bests(self) -> List[float]:
        with self._lock:
            return self._bests

    @property
    def full(self) -> bool:
        with self._lock:
            return self._n == len(self)

    def __contains__(self, tid: str) -> bool:
        with self._lock:
            return tid in self._heap

    def push(self, report: TrialReport) -> bool:
        with self._lock:
            if len(self) == 0:
                best = report.sort_metric
            else:
                best = min(self.best, report.sort_metric)
            self._heap.push(report)
            self._bests.append(best)
            return (
                len(self._heap) <= self._n
                or self._heap.pop().trial_id != report.trial_id
            )


class PerTrialJudge(TrialJudge):
    def __init__(self, parent: "PerPartitionASHAJudge") -> None:
        super().__init__()
        self._history: List[TrialReport] = []
        self._parent = parent

    def get_budget(self, trial: Trial, rung: int) -> float:
        def _get():
            if rung >= len(
                self._parent._parent.schedule
            ) or not self._parent.can_accept(trial):
                return 0.0
            return self._parent._parent.schedule[rung][0]

        res = _get()
        self._parent._parent.monitor.on_get_budget(trial, rung, res)
        return res

    def judge(self, report: TrialReport) -> TrialDecision:
        self._parent._parent.monitor.on_report(report)
        self._history.append(report)
        trial_should_stop = self._parent._parent._trial_should_stop_func(
            self._history, self._parent._rungs[: report.rung]
        )
        promoted = not trial_should_stop and self._parent._rungs[report.rung].push(
            report
        )
        if not promoted or report.rung >= len(self._parent._parent.schedule) - 1:
            return TrialDecision(report, budget=0, should_checkpoint=True)
        next_budget = self.get_budget(report.trial, report.rung + 1)
        decision = TrialDecision(
            report,
            budget=next_budget,
            should_checkpoint=next_budget <= 0
            or self._parent._parent.always_checkpoint,
        )
        self._parent._parent.monitor.on_judge(decision)
        return decision


class PerPartitionASHAJudge(TrialJudge):
    def __init__(self, parent: "ASHAJudge", keys: List[Any]):
        super().__init__()
        self._keys = keys
        self._data: Dict[str, PerTrialJudge] = {}
        self._lock = RLock()
        self._parent = parent
        self._rungs: List[RungHeap] = [RungHeap(x[1]) for x in self._parent.schedule]
        self._active = True

    def can_accept(self, trial: Trial) -> bool:
        if not self._active:
            return False
        self._active = not self._parent._should_deactivate_func(self._keys, self._rungs)
        return self._active

    def get_budget(self, trial: Trial, rung: int) -> float:
        return self._get_judge(trial).get_budget(trial, rung)

    def judge(self, report: TrialReport) -> TrialDecision:
        return self._get_judge(report.trial).judge(report)

    def _get_judge(self, trial: Trial) -> "PerTrialJudge":
        key = to_uuid(trial.keys)
        with self._lock:
            if key not in self._data:
                self._data[key] = PerTrialJudge(self)
            return self._data[key]


def default_should_deactivate(keys: List[Any], rungs: List[RungHeap]) -> bool:
    return all(r.full for r in rungs)


def default_trial_should_stop(
    reports: List[TrialReport], rungs: List[RungHeap]
) -> bool:
    return False


class ASHAJudge(TrialJudge):
    def __init__(
        self,
        schedule: List[Tuple[float, int]],
        always_checkpoint: bool = False,
        should_deactivate_func: Callable[
            [List[Any], List[RungHeap]], bool
        ] = default_should_deactivate,
        trial_should_stop_func: Callable[
            [List[TrialReport], List[RungHeap]], bool
        ] = default_trial_should_stop,
        monitor: Optional[TrialJudgeMonitor] = None,
    ):
        super().__init__(monitor=monitor)
        self._lock = RLock()
        self._data: Dict[str, PerPartitionASHAJudge] = {}
        self._schedule = schedule
        self._always_checkpoint = always_checkpoint
        self._should_deactivate_func = should_deactivate_func
        self._trial_should_stop_func = trial_should_stop_func

    @property
    def schedule(self) -> List[Tuple[float, int]]:
        return self._schedule

    @property
    def always_checkpoint(self) -> bool:
        return self._always_checkpoint

    def can_accept(self, trial: Trial) -> bool:
        return self._get_judge(trial).can_accept(trial)

    def get_budget(self, trial: Trial, rung: int) -> float:
        return self._get_judge(trial).get_budget(trial, rung)

    def judge(self, report: TrialReport) -> TrialDecision:
        return self._get_judge(report.trial).judge(report)

    def _get_judge(self, trial: Trial) -> PerPartitionASHAJudge:
        key = to_uuid(trial.keys)
        with self._lock:
            if key not in self._data:
                self._data[key] = PerPartitionASHAJudge(self, trial.keys)
            return self._data[key]
