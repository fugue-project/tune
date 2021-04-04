import copy
import inspect
from typing import Any, Callable, Dict, Optional, Tuple, no_type_check

from fugue._utils.interfaceless import (
    FunctionWrapper,
    _FuncParam,
    _OtherParam,
    is_class_method,
)
from triad import assert_or_throw
from triad.utils.convert import get_caller_global_local_vars, to_function
from tune.exceptions import TuneCompileError
from tune.noniterative.objective import NonIterativeObjectiveFunc
from tune.trial import Trial, TrialReport


def noniterative_objective(
    func: Optional[Callable] = None, min_better: bool = True
) -> Callable[[Any], NonIterativeObjectiveFunc]:
    def deco(func: Callable) -> NonIterativeObjectiveFunc:
        assert_or_throw(
            not is_class_method(func),
            NotImplementedError(
                "non_iterative_objective decorator can't be used on class methods"
            ),
        )
        return _NonIterativeObjectiveFuncWrapper.from_func(func, min_better)

    if func is None:
        return deco
    else:
        return deco(func)  # type: ignore


def to_noniterative_objective(
    obj: Any,
    min_better: bool = True,
    global_vars: Optional[Dict[str, Any]] = None,
    local_vars: Optional[Dict[str, Any]] = None,
) -> NonIterativeObjectiveFunc:
    if isinstance(obj, NonIterativeObjectiveFunc):
        return copy.copy(obj)
    global_vars, local_vars = get_caller_global_local_vars(global_vars, local_vars)
    try:
        f = to_function(obj, global_vars=global_vars, local_vars=local_vars)
        # this is for string expression of function with decorator
        if isinstance(f, NonIterativeObjectiveFunc):
            return copy.copy(f)
        # this is for functions without decorator
        return _NonIterativeObjectiveFuncWrapper.from_func(f, min_better)
    except Exception as e:
        exp = e
    raise TuneCompileError(f"{obj} is not a valid tunable function", exp)


class _NonIterativeObjectiveFuncWrapper(NonIterativeObjectiveFunc):
    def __init__(self, min_better: bool):
        self._min_better = min_better

    @property
    def min_better(self) -> bool:
        return self._min_better

    def generate_sort_metric(self, value: float) -> float:
        return float(value) if self._min_better else -float(value)

    @no_type_check
    def run(self, trial: Trial) -> TrialReport:
        if self._orig_input:
            result = self._func(trial)
        else:
            result = self._func(**trial.params, **trial.dfs)
        return self._output_f(result, trial)

    @no_type_check
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)

    @no_type_check
    @staticmethod
    def from_func(
        func: Callable, min_better: bool
    ) -> "_NonIterativeObjectiveFuncWrapper":
        f = _NonIterativeObjectiveFuncWrapper(min_better=min_better)
        w = _NonIterativeObjectiveWrapper(func)
        f._func = w._func
        f._orig_input = w._orig_input
        f._output_f = w._rt.to_report
        return f


class _ReportParam(_FuncParam):
    def __init__(self, param: Optional[inspect.Parameter], annotation: Any, code: str):
        super().__init__(param, annotation, code)

    def to_report(self, v: Any, trial: Trial) -> TrialReport:
        raise NotImplementedError  # pragma: no cover


class _RawReportParam(_ReportParam):
    def __init__(self, param: Optional[inspect.Parameter]):
        super().__init__(param, "TrialReport", "r")

    def to_report(self, v: Any, trial: Trial) -> TrialReport:
        return v


class _MetricParam(_ReportParam):
    def __init__(self, param: Optional[inspect.Parameter]):
        super().__init__(param, "float", "1")

    def to_report(self, v: Any, trial: Trial) -> TrialReport:
        return TrialReport(trial, metric=float(v), params=trial.params, metadata={})


class _MetricMetadataParam(_ReportParam):
    def __init__(self, param: Optional[inspect.Parameter]):
        super().__init__(param, "Tuple[float,Dict[str,Any]]", "2")

    def to_report(self, v: Any, trial: Trial) -> TrialReport:
        return TrialReport(
            trial, metric=float(v[0]), params=trial.params, metadata=v[1]
        )


class _TrialParam(_FuncParam):
    def __init__(self, param: Optional[inspect.Parameter]):
        super().__init__(param, "Trial", "t")


class _NonIterativeObjectiveWrapper(FunctionWrapper):
    def __init__(self, func: Callable):
        self._orig_input = False
        self._orig_output = False
        super().__init__(func, "^(t|([x1]+))$", "^[r12]$")

    def _parse_param(
        self,
        annotation: Any,
        param: Optional[inspect.Parameter],
        none_as_other: bool = True,
    ) -> _FuncParam:
        if annotation is float:
            return _MetricParam(param)
        elif annotation is Tuple[float, Dict[str, Any]]:
            return _MetricMetadataParam(param)
        elif annotation is TrialReport:
            self._orig_output = True
            return _RawReportParam(param)
        elif annotation is Trial:
            self._orig_input = True
            return _TrialParam(param)
        else:
            return _OtherParam(param)
