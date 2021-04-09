from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from triad import to_uuid
from tune.trial import (
    Monitor,
    Trial,
    TrialDecision,
    TrialJudge,
    TrialReport,
    TrialReportHeap,
)


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
            return self.capacity <= len(self)

    def __contains__(self, tid: str) -> bool:
        with self._lock:
            return tid in self._heap

    def values(self) -> Iterable[TrialReport]:
        return self._heap.values()

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


class _PerTrial:
    def __init__(self, parent: "_PerPartition") -> None:
        self._history: List[TrialReport] = []
        self._parent = parent
        self._active = True

    def can_promote(self, report: TrialReport) -> Tuple[bool, str]:
        reasons: List[str] = []
        if self._active:
            can_accept = self._parent.can_accept(report.trial)
            early_stop = self._parent._parent._trial_early_stop(
                report, self._history, self._parent._rungs
            )
            self._active = can_accept and not early_stop
            if not can_accept:
                reasons.append("can't accept new")
            if early_stop:
                reasons.append("trial early stop")
        if self._active:
            self._history.append(report)
            can_push = self._parent._rungs[report.rung].push(report)
            if not can_push:
                # data = sorted(
                #     (x.jsondict for x in self._parent._rungs[report.rung].values()),
                #     key=lambda x: x["sort_metric"],
                # )
                # reasons.append("not best: " + json.dumps(data))
                reasons.append("not best")
            return can_push, ", ".join(reasons)
        return False, ", ".join(reasons)

    def judge(self, report: TrialReport) -> TrialDecision:
        if report.rung >= len(self._parent._parent.schedule) - 1:
            self._history.append(report)
            self._parent._rungs[report.rung].push(report)
            return TrialDecision(
                report, budget=0, should_checkpoint=True, reason="last"
            )
        promote, reason = self.can_promote(report)
        if not promote:
            return TrialDecision(
                report, budget=0, should_checkpoint=True, reason=reason
            )
        next_budget = self._parent.get_budget(report.trial, report.rung + 1)
        return TrialDecision(
            report,
            budget=next_budget,
            should_checkpoint=next_budget <= 0
            or self._parent._parent.always_checkpoint,
            reason="" if next_budget > 0 else "budget==0",
        )


class _PerPartition:
    def __init__(self, parent: "ASHAJudge", keys: List[Any]):
        self._keys = keys
        self._data: Dict[str, _PerTrial] = {}
        self._lock = RLock()
        self._parent = parent
        self._rungs: List[RungHeap] = [RungHeap(x[1]) for x in self._parent.schedule]
        self._active = True
        self._accepted_ids: Set[str] = set()

    def can_accept(self, trial: Trial) -> bool:
        with self._lock:
            if self._active:
                self._active = not self._parent._study_early_stop(
                    self._keys, self._rungs
                )
                if self._active:
                    self._accepted_ids.add(trial.trial_id)
                    return True
            # if not active, can only accept existing trials
            return trial.trial_id in self._accepted_ids

    def get_budget(self, trial: Trial, rung: int) -> float:
        if rung >= len(self._parent.schedule) or not self.can_accept(trial):
            return 0.0
        return self._parent.schedule[rung][0]

    def judge(self, report: TrialReport) -> TrialDecision:
        return self._get_judge(report.trial).judge(report)

    def _get_judge(self, trial: Trial) -> _PerTrial:
        key = trial.trial_id
        with self._lock:
            if key not in self._data:
                self._data[key] = _PerTrial(self)
            return self._data[key]


class ASHAJudge(TrialJudge):
    def __init__(
        self,
        schedule: List[Tuple[float, int]],
        always_checkpoint: bool = False,
        study_early_stop: Optional[Callable[[List[Any], List[RungHeap]], bool]] = None,
        trial_early_stop: Optional[
            Callable[[TrialReport, List[TrialReport], List[RungHeap]], bool]
        ] = None,
        monitor: Optional[Monitor] = None,
    ):
        super().__init__(monitor=monitor)
        self._lock = RLock()
        self._data: Dict[str, _PerPartition] = {}
        self._schedule = schedule
        self._always_checkpoint = always_checkpoint
        self._study_early_stop = study_early_stop or _default_study_early_stop
        self._trial_early_stop = trial_early_stop or _default_trial_early_stop

    @property
    def schedule(self) -> List[Tuple[float, int]]:
        return self._schedule

    @property
    def always_checkpoint(self) -> bool:
        return self._always_checkpoint

    def can_accept(self, trial: Trial) -> bool:
        return self._get_judge(trial).can_accept(trial)

    def get_budget(self, trial: Trial, rung: int) -> float:
        budget = self._get_judge(trial).get_budget(trial, rung)
        self.monitor.on_get_budget(trial, rung, budget)
        return budget

    def judge(self, report: TrialReport) -> TrialDecision:
        self.monitor.on_report(report)
        decision = self._get_judge(report.trial).judge(report)
        self.monitor.on_judge(decision)
        return decision

    def _get_judge(self, trial: Trial) -> _PerPartition:
        key = to_uuid(trial.keys)
        with self._lock:
            if key not in self._data:
                self._data[key] = _PerPartition(self, trial.keys)
            return self._data[key]


def _default_study_early_stop(keys: List[Any], rungs: List["RungHeap"]) -> bool:
    return all(r.full for r in rungs)


def _default_trial_early_stop(
    report: TrialReport, reports: List[TrialReport], rungs: List["RungHeap"]
) -> bool:
    return False
