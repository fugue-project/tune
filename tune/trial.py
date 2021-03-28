from typing import Any, Dict, Optional

from tune.space.parameters import decode_params, encode_params


class Trial:
    def __init__(
        self,
        trial_id: str,
        params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        dfs: Optional[Dict[str, Any]] = None,
        raw: bool = False,
    ):
        self._trial_id = trial_id
        self._params = params if raw else decode_params(params)
        self._metadata = metadata or {}
        self._dfs = dfs or {}

    def copy(self) -> "Trial":
        return Trial(
            trial_id=self._trial_id,
            params=self._params,
            metadata=self._metadata,
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
        raw: bool = False,
    ):
        self._trial = trial
        self._metric = float(metric)
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
    def metric(self) -> float:
        return self._metric

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def jsondict(self) -> Dict[str, Any]:
        return {
            "trial": self.trial.jsondict,
            "metric": self.metric,
            "params": encode_params(self.params),
            "metadata": self.metadata,
        }

    @staticmethod
    def from_jsondict(data: Dict[str, Any]) -> "TrialReport":
        trial = Trial.from_jsondict(data.pop("trial"))
        return TrialReport(trial=trial, **data)


class TrialDecision:
    def __init__(
        self,
        report: TrialReport,
        should_stop: bool,
        should_checkpoint: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._report = report
        self._should_stop = should_stop
        self._should_checkpoint = should_checkpoint
        self._metadata = metadata or {}

    @property
    def report(self) -> TrialReport:
        return self._report

    @property
    def trial(self) -> Trial:
        return self.report.trial

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def should_checkpoint(self) -> bool:
        return self._should_checkpoint

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata
