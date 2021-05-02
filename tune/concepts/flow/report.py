import heapq
import json
from datetime import datetime
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Set

from triad.utils.convert import to_datetime
from tune.concepts.space.parameters import decode_params, encode_params
from tune.concepts.flow.trial import Trial
from tune.constants import TUNE_REPORT, TUNE_REPORT_ID, TUNE_REPORT_METRIC


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

    def log(self, report: TrialReport) -> None:  # pragma: no cover
        pass

    @property
    def best(self) -> Optional[TrialReport]:
        with self._lock:
            return self._best_report
