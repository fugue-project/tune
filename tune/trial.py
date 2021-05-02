import heapq
import json
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from tune.constants import TUNE_REPORT, TUNE_REPORT_ID, TUNE_REPORT_METRIC
from tune.space.parameters import decode_params, encode_params
from datetime import datetime
from triad.utils.convert import to_datetime


class Trial:
    def __init__(
        self,
        trial_id: str,
        params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        keys: Optional[List[str]] = None,
        dfs: Optional[Dict[str, Any]] = None,
        raw: bool = False,
    ):
        self._trial_id = trial_id
        self._params = params if raw else decode_params(params)
        self._metadata = metadata or {}
        self._keys = keys or []
        self._dfs = dfs or {}

    def copy(self) -> "Trial":
        return Trial(
            trial_id=self._trial_id,
            params=self._params,
            metadata=self._metadata,
            keys=self._keys,
            dfs=self._dfs,
            raw=True,
        )

    def __copy__(self) -> "Trial":
        return self.copy()

    def __deepcopy__(self, memo: Any) -> "Trial":
        return self.copy()

    @property
    def trial_id(self) -> str:
        return self._trial_id

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    @property
    def keys(self) -> List[str]:
        return self._keys

    @property
    def dfs(self) -> Dict[str, Any]:
        return self._dfs

    def with_dfs(self, dfs: Dict[str, Any]) -> "Trial":
        if len(dfs) == 0 and len(self.dfs) == 0:
            return self
        t = self.copy()
        t._dfs = dfs
        return t

    def with_params(self, params: Dict[str, Any], raw: bool = False) -> "Trial":
        t = self.copy()
        t._params = params if raw else decode_params(params)
        return t

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def jsondict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "params": encode_params(self.params),
            "metadata": self.metadata,
            "keys": self.keys,
        }

    @staticmethod
    def from_jsondict(data: Dict[str, Any]) -> "Trial":
        return Trial(**data)


class TrialReport:
    def __init__(
        self,
        trial: Trial,
        metric: Any,
        params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        cost: float = 1.0,
        rung: int = 0,
        sort_metric: Any = None,
        raw: bool = False,
        log_time: Any = None,
    ):
        self._trial = trial.with_dfs({})
        self._metric = float(metric)
        self._cost = float(cost)
        self._rung = rung
        self._sort_metric = self._metric if sort_metric is None else float(sort_metric)
        if params is None:
            self._params = trial.params
        else:
            self._params = params if raw else decode_params(params)
        self._metadata = metadata or {}
        self._log_time = datetime.now() if log_time is None else to_datetime(log_time)

    def copy(self) -> "TrialReport":
        return TrialReport(
            trial=self.trial,
            metric=self.metric,
            params=self._params,
            metadata=self._metadata,
            cost=self._cost,
            rung=self._rung,
            sort_metric=self._sort_metric,
            raw=True,
            log_time=self.log_time,
        )

    def __copy__(self) -> "TrialReport":
        return self.copy()

    def __deepcopy__(self, memo: Any) -> "TrialReport":
        return self.copy()

    @property
    def log_time(self) -> datetime:
        return self._log_time

    def reset_log_time(self) -> "TrialReport":
        res = self.copy()
        res._log_time = datetime.now()
        return res

    @property
    def trial(self) -> Trial:
        return self._trial

    @property
    def trial_id(self) -> str:
        return self.trial.trial_id

    @property
    def metric(self) -> float:
        return self._metric

    @property
    def sort_metric(self) -> float:
        return self._sort_metric

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def rung(self) -> int:
        return self._rung

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def with_cost(self, cost: float) -> "TrialReport":
        t = self.copy()
        t._cost = cost
        return t

    def with_rung(self, rung: int) -> "TrialReport":
        t = self.copy()
        t._rung = rung
        return t

    def with_sort_metric(self, sort_metric: Any) -> "TrialReport":
        t = self.copy()
        t._sort_metric = float(sort_metric)
        return t

    def generate_sort_metric(self, min_better: bool, digits: int) -> "TrialReport":
        t = self.copy()
        t._sort_metric = (
            round(self.metric, digits) if min_better else -round(self.metric, digits)
        )
        return t

    @property
    def jsondict(self) -> Dict[str, Any]:
        return {
            "trial": self.trial.jsondict,
            "metric": self.metric,
            "params": encode_params(self.params),
            "metadata": self.metadata,
            "cost": self.cost,
            "rung": self.rung,
            "sort_metric": self.sort_metric,
            "log_time": str(self.log_time),
        }

    @staticmethod
    def from_jsondict(data: Dict[str, Any]) -> "TrialReport":
        trial = Trial.from_jsondict(data.pop("trial"))
        return TrialReport(trial=trial, **data)

    def fill_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[TUNE_REPORT_ID] = self.trial_id
        data[TUNE_REPORT_METRIC] = self.sort_metric
        data[TUNE_REPORT] = json.dumps(self.jsondict)
        return data


class TrialReportHeap:
    class _Wrapper:
        def __init__(self, report: TrialReport, min_heap: bool):
            self.report = report
            self.min_heap = min_heap

        def __lt__(self, other: "TrialReportHeap._Wrapper") -> bool:
            k1 = (self.report.sort_metric, self.report.cost, self.report.rung)
            k2 = (other.report.sort_metric, other.report.cost, other.report.rung)
            return k1 < k2 if self.min_heap else k2 < k1

        @property
        def trial_id(self) -> str:
            return self.report.trial_id

    def __init__(self, min_heap: bool):
        self._data: List[TrialReportHeap._Wrapper] = []
        self._ids: Set[str] = set()
        self._min_heap = min_heap

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, tid: str) -> bool:
        return tid in self._ids

    def values(self) -> Iterable[TrialReport]:
        for x in self._data:
            yield x.report

    def push(self, report: TrialReport) -> None:
        w = TrialReportHeap._Wrapper(report, self._min_heap)
        if w.trial_id in self._ids:
            self._data = [x if x.trial_id != w.trial_id else w for x in self._data]
            heapq.heapify(self._data)
        else:
            self._ids.add(w.trial_id)
            heapq.heappush(self._data, w)

    def pop(self) -> TrialReport:
        w = heapq.heappop(self._data)
        self._ids.remove(w.trial_id)
        return w.report


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

    @property
    def jsondict(self) -> Dict[str, Any]:
        return {
            "report": self.report.jsondict,
            "budget": self.budget,
            "should_checkpoint": self.should_checkpoint,
            "reason": self.reason,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_jsondict(data: Dict[str, Any]) -> "TrialDecision":
        report = TrialReport.from_jsondict(data.pop("report"))
        return TrialDecision(report=report, **data)


class TrialJudge(object):
    def __init__(self, monitor: Optional["Monitor"] = None):
        self.reset_monitor(monitor)

    @property
    def monitor(self) -> "Monitor":
        assert self._trial_judge_monitor is not None
        return self._trial_judge_monitor

    def reset_monitor(self, monitor: Optional["Monitor"] = None) -> None:
        self._trial_judge_monitor = monitor or Monitor()
        self._trial_judge_monitor.set_judge(self)

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
        return self._entrypoint("can_accept", dict(trial=trial.jsondict))

    def judge(self, report: TrialReport) -> TrialDecision:
        self._report = report
        return TrialDecision.from_jsondict(self._entrypoint("judge", report.jsondict))

    def get_budget(self, trial: Trial, rung: int) -> float:
        return self._entrypoint("get_budget", dict(trial=trial.jsondict, rung=rung))


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
            return self._judge.can_accept(Trial.from_jsondict(kwargs["trial"]))
        if name == "judge":
            return self._judge.judge(TrialReport.from_jsondict(kwargs)).jsondict
        if name == "get_budget":
            return self._judge.get_budget(
                Trial.from_jsondict(kwargs["trial"]), kwargs["rung"]
            )
        raise NotImplementedError  # pragma: no cover


class Monitor:
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


class TrialReportLogger:
    def __init__(self, new_best_only: bool = False):
        self._lock = RLock()
        self._best_report: Optional[TrialReport] = None
        self._new_best_only = new_best_only

    def on_report(self, report: TrialReport) -> bool:
        with self._lock:
            updated = False
            if (
                self._best_report is None
                or report.sort_metric < self._best_report.sort_metric
            ):
                self._best_report = report
                updated = True
            if updated or not self._new_best_only:
                self.log(report)
            return updated

    def log(self, report: TrialReport) -> None:
        pass

    @property
    def best(self) -> Optional[TrialReport]:
        with self._lock:
            return self._best_report
