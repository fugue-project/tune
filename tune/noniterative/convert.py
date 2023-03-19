import copy
from typing import Any, Callable, Dict, Optional, Tuple, no_type_check

from fugue._utils.interfaceless import is_class_method
from triad import assert_or_throw
from triad.collections.function_wrapper import (
    AnnotatedParam,
    FunctionWrapper,
    function_wrapper,
)
from triad.utils.convert import get_caller_global_local_vars, to_function

from tune.concepts.flow import Trial, TrialReport
from tune.exceptions import TuneCompileError
from tune.noniterative.objective import NonIterativeObjectiveFunc


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
            result = self._func(**trial.params.simple_value, **trial.dfs)
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


@function_wrapper(None)
class _NonIterativeObjectiveWrapper(FunctionWrapper):
    def __init__(self, func: Callable):
        super().__init__(func, ".*", "^[r12]$")
        param = self._params.get_value_by_index(0)
        self._orig_input = isinstance(param, _TrialParam)
        self._orig_output = isinstance(self._rt, _RawReportParam)


class _ReportParam(AnnotatedParam):
    def to_report(self, v: Any, trial: Trial) -> TrialReport:
        raise NotImplementedError  # pragma: no cover


@_NonIterativeObjectiveWrapper.annotated_param(TrialReport, "r")
class _RawReportParam(_ReportParam):
    def to_report(self, v: Any, trial: Trial) -> TrialReport:
        return v


@_NonIterativeObjectiveWrapper.annotated_param(float, "1")
class _MetricParam(_ReportParam):
    def to_report(self, v: Any, trial: Trial) -> TrialReport:
        return TrialReport(trial, metric=float(v), params=trial.params, metadata={})


@_NonIterativeObjectiveWrapper.annotated_param(Tuple[float, Dict[str, Any]], "2")
class _MetricMetadataParam(_ReportParam):
    def to_report(self, v: Any, trial: Trial) -> TrialReport:
        return TrialReport(
            trial, metric=float(v[0]), params=trial.params, metadata=v[1]
        )


@_NonIterativeObjectiveWrapper.annotated_param(Trial, "t")
class _TrialParam(AnnotatedParam):
    pass
