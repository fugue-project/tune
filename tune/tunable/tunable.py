from typing import Any, Dict

from fugue import ExecutionEngine
from triad import ParamDict
from tune.exceptions import TuneRuntimeError


class Tunable(object):
    def run(self, **kwargs: Any) -> None:  # pragma: no cover
        raise NotImplementedError

    def report(self, result: Dict[str, Any]) -> None:
        self._metric = float(result["metric"])
        self._params = ParamDict(result.get("params", None))
        self._metadata = ParamDict(result.get("metadata", None))

    @property
    def metric(self) -> float:
        try:
            return self._metric
        except Exception:
            raise TuneRuntimeError("metric is not set")

    @property
    def params(self) -> ParamDict:
        try:
            return self._params
        except Exception:
            raise TuneRuntimeError("params is not set")

    @property
    def metadata(self) -> ParamDict:
        try:
            return self._metadata
        except Exception:
            raise TuneRuntimeError("metadata is not set")

    @property
    def distributable(self) -> bool:  # pragma: no cover
        return True

    @property
    def execution_engine(self) -> ExecutionEngine:
        # pylint: disable=no-member
        try:
            return self._execution_engine  # type: ignore
        except Exception:
            raise TuneRuntimeError("execution_engine is not set")
