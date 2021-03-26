from typing import Dict, Any


class Trial:
    def __init__(self, trial_id: str, params: Dict[str, Any], metadata: Dict[str, Any]):
        self._trial_id = trial_id
        self._params = params
        self._metadata = metadata

    @property
    def trial_id(self) -> str:
        return self._trial_id

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    def with_params(self, params: Dict[str, Any]) -> "Trial":
        return Trial(
            self.trial_id,
            params,
            self.metadata,
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata


class TrialReport:
    def __init__(
        self,
        trial: Trial,
        metric: Any,
        params: Dict[str, Any],
        metadata: Dict[str, Any],
    ):
        self._trial = trial
        self._metric = float(metric)
        self._params = params
        self._metadata = metadata

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


class TrialDecision:
    def __init__(
        self,
        report: TrialReport,
        should_stop: bool,
        should_checkpoint: bool,
        metadata: Dict[str, Any],
    ):
        self._report = report
        self._should_stop = should_stop
        self._should_checkpoint = should_checkpoint
        self._metadata = metadata

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
