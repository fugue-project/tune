import heapq
from datetime import datetime
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Set

from triad.utils.convert import to_datetime
from tune._utils import to_base64
from tune.concepts.flow.trial import Trial
from tune.concepts.space.parameters import TuningParametersTemplate, to_template
from tune.constants import TUNE_REPORT, TUNE_REPORT_ID, TUNE_REPORT_METRIC


class TrialReport:
    """The result from running the objective. It is immutable.

    :param trial: the original trial sent to the objective
    :param metric: the raw metric from the objective output
    :param params: updated parameters based on the trial input, defaults to None.
      If none, it means the params from the trial was not updated, otherwise
      it is an object convertible to ``TuningParametersTemplate``
      by :func:`~tune.concepts.space.parameters.to_template`
    :param metadata: metadata from the objective output, defaults to None
    :param cost: cost to run the objective, defaults to 1.0
    :param rung: number of rungs in the current objective, defaults to 0. This is
      for iterative problems
    :param sort_metric: the metric for comparison, defaults to None. It must be
      smaller better. If not set, it implies the ``metric`` is ``sort_metric`` and
      it is smaller better
    :param log_time: the time generating this report, defaults to None. If None, current
      time will be used

    .. attention::

        This class is not for users to construct directly.

    """

    def __init__(
        self,
        trial: Trial,
        metric: Any,
        params: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        cost: float = 1.0,
        rung: int = 0,
        sort_metric: Any = None,
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
            self._params = to_template(params)
        self._metadata = metadata or {}
        self._log_time = datetime.now() if log_time is None else to_datetime(log_time)

    def copy(self) -> "TrialReport":
        """Copy the current object.

        :return: the copied object

        .. note::
            This is shallow copy, but it is also used by `__deepcopy__`
            of this object. This is because we disable deepcopy
            of TrialReport.
        """
        return TrialReport(
            trial=self.trial,
            metric=self.metric,
            params=self._params,
            metadata=self._metadata,
            cost=self._cost,
            rung=self._rung,
            sort_metric=self._sort_metric,
            log_time=self.log_time,
        )

    def __repr__(self) -> str:
        return repr(
            dict(
                trial=self.trial,
                metric=self.metric,
                params=self._params,
                metadata=self._metadata,
                cost=self._cost,
                rung=self._rung,
                sort_metric=self._sort_metric,
                log_time=self.log_time,
            )
        )

    def __copy__(self) -> "TrialReport":
        """Shallow copy"""
        return self.copy()

    def __deepcopy__(self, memo: Any) -> "TrialReport":
        """(Enforced) shallow copy"""
        return self.copy()

    @property
    def log_time(self) -> datetime:
        """The time generating this report"""
        return self._log_time

    def reset_log_time(self) -> "TrialReport":
        """Reset :meth:`~.log_time` to now"""
        res = self.copy()
        res._log_time = datetime.now()
        return res

    @property
    def trial(self) -> Trial:
        """The original trial sent to the objective"""
        return self._trial

    @property
    def trial_id(self) -> str:
        """:meth:`tune.concepts.flow.trial.Trial.trial_id`"""
        return self.trial.trial_id

    @property
    def metric(self) -> float:
        """The raw metric from the objective output"""
        return self._metric

    @property
    def sort_metric(self) -> float:
        """The metric for comparison"""
        return self._sort_metric

    @property
    def cost(self) -> float:
        """The cost to run the objective"""
        return self._cost

    @property
    def rung(self) -> int:
        """The number of rungs in the current objective, defaults to 0. This is
        for iterative problems
        """
        return self._rung

    @property
    def params(self) -> TuningParametersTemplate:
        """The parameters used by the objective to generate the
        :meth:`~.metric`
        """
        return self._params

    @property
    def metadata(self) -> Dict[str, Any]:
        """The metadata from the objective output"""
        return self._metadata

    def with_cost(self, cost: float) -> "TrialReport":
        """Construct a new report object with the new ``cost``

        :param cost: new cost
        :return: a new object with the updated value
        """
        t = self.copy()
        t._cost = cost
        return t

    def with_rung(self, rung: int) -> "TrialReport":
        """Construct a new report object with the new ``rung``

        :param rung: new rung
        :return: a new object with the updated value
        """
        t = self.copy()
        t._rung = rung
        return t

    def with_sort_metric(self, sort_metric: Any) -> "TrialReport":
        """Construct a new report object with the new ``sort_metric``

        :param sort_metric: new sort_metric
        :return: a new object with the updated value
        """
        t = self.copy()
        t._sort_metric = float(sort_metric)
        return t

    def generate_sort_metric(self, min_better: bool, digits: int) -> "TrialReport":
        """Construct a new report object with the new derived``sort_metric``

        :param min_better: whether the current :meth:`~.metric` is smaller better
        :param digits: number of digits to keep in ``sort_metric``
        :return: a new object with the updated value
        """
        t = self.copy()
        t._sort_metric = (
            round(self.metric, digits) if min_better else -round(self.metric, digits)
        )
        return t

    def __getstate__(self) -> Dict[str, Any]:
        keys = [
            "_trial",
            "_metric",
            "_params",
            "_metadata",
            "_cost",
            "_rung",
            "_sort_metric",
            "_log_time",
        ]
        return {k: self.__dict__[k] for k in keys}

    def __setstate__(self, d: Dict[str, Any]) -> None:
        for k, v in d.items():
            self.__dict__[k] = v

    def fill_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fill a row of :class:`~tune.concepts.dataset.StudyResult` with
        the report information

        :param data: a row (as dict) from :class:`~tune.concepts.dataset.StudyResult`
        :return: the updated ``data``
        """
        data[TUNE_REPORT_ID] = self.trial_id
        data[TUNE_REPORT_METRIC] = self.sort_metric
        data[TUNE_REPORT] = to_base64(self)
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
