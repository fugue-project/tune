import heapq
import json
from typing import Any, Dict, Iterable, List, Optional, Set

from tune.constants import TUNE_REPORT, TUNE_REPORT_ID, TUNE_REPORT_METRIC
from tune.space.parameters import decode_params, encode_params


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
    ):
        self._trial = trial
        self._metric = float(metric)
        self._cost = float(cost)
        self._rung = rung
        self._sort_metric = self._metric if sort_metric is None else float(sort_metric)
        if params is None:
            self._params = trial.params
        else:
            self._params = params if raw else decode_params(params)
        self._metadata = metadata or {}

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
        )

    def __copy__(self) -> "TrialReport":
        return self.copy()

    def __deepcopy__(self, memo: Any) -> "TrialReport":
        return self.copy()

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
        self._trial_judge_monitor = monitor or Monitor()
        self._trial_judge_monitor.set_judge(self)

    @property
    def monitor(self) -> "Monitor":
        assert self._trial_judge_monitor is not None
        return self._trial_judge_monitor

    def can_accept(self, trial: Trial) -> bool:  # pragma: no cover
        raise NotImplementedError

    def get_budget(self, trial: Trial, rung: int) -> float:  # pragma: no cover
        raise NotImplementedError

    def judge(self, report: TrialReport) -> TrialDecision:  # pragma: no cover
        raise NotImplementedError


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
