from typing import Any, Callable, Dict, Iterable, Optional

from fugue import ArrayDataFrame, DataFrame, ExecutionEngine
from triad import assert_or_throw
from tune.constants import TUNE_REPORT_ADD_SCHEMA
from tune.dataset import StudyResult, TuneDataset, get_trials_from_row
from tune.exceptions import TuneCompileError
from tune.noniterative.objective import (
    NonIterativeObjectiveFunc,
    NonIterativeObjectiveRunner,
)
from tune.trial import Monitor, TrialReport


class NonIterativeStudy:
    def __init__(
        self, objective: NonIterativeObjectiveFunc, runner: NonIterativeObjectiveRunner
    ):
        self._objective = objective
        self._runner = runner

    def optimize(
        self,
        dataset: TuneDataset,
        distributed: Optional[bool] = None,
        monitor: Optional[Monitor] = None,
    ) -> StudyResult:
        _dist = self._get_distributed(distributed)
        on_report: Any = monitor.on_report if monitor is not None else None

        def compute_transformer(
            df: Iterable[Dict[str, Any]],
            _on_report: Optional[Callable[[TrialReport], None]] = None,
        ) -> Iterable[Dict[str, Any]]:
            for row in df:
                for trial in get_trials_from_row(row):
                    report = self._runner.run(self._objective, trial)
                    report = report.with_sort_metric(
                        self._objective.generate_sort_metric(report.metric)
                    )
                    if _on_report is not None:
                        _on_report(report)
                    yield report.fill_dict(dict(row))

        def compute_processor(engine: ExecutionEngine, df: DataFrame) -> DataFrame:
            out_schema = df.schema + TUNE_REPORT_ADD_SCHEMA

            def get_rows() -> Iterable[Any]:
                for row in compute_transformer(
                    df.as_local().as_dict_iterable(), on_report
                ):
                    yield [row[k] for k in out_schema.names]

            # TODO: need to add back execution_engine for engine aware runners
            # t._execution_engine = engine  # type:ignore
            return ArrayDataFrame(get_rows(), out_schema)

        if not _dist:
            res = dataset.data.process(compute_processor)
        else:
            res = dataset.data.per_row().transform(
                compute_transformer,
                schema=f"*,{TUNE_REPORT_ADD_SCHEMA}",
                callback=on_report,
            )

        return StudyResult(dataset=dataset, result=res)

    def _get_distributed(self, distributed: Optional[bool]) -> bool:
        if distributed is None:
            return self._runner.distributable
        if distributed:
            assert_or_throw(
                self._runner.distributable,
                TuneCompileError(
                    f"can't distribute non-distributable runner {self._runner}"
                ),
            )
            return True
        return False
