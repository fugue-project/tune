from typing import Any, Callable, Dict, Iterable, List, Optional

from fugue import ArrayDataFrame, DataFrame, ExecutionEngine
from triad import assert_or_throw
from tune.constants import TUNE_REPORT_ADD_SCHEMA
from tune.dataset import StudyResult, TuneDataset, get_trials_from_row
from tune.exceptions import TuneCompileError
from tune.noniterative.objective import (
    NonIterativeObjectiveFunc,
    NonIterativeObjectiveRunner,
)
from tune.trial import (
    NoOpTrailJudge,
    RemoteTrialJudge,
    TrialCallback,
    TrialJudge,
    TrialReport,
)


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
        judge: Optional[TrialJudge] = None,
    ) -> StudyResult:
        _dist = self._get_distributed(distributed)
        entrypoint: Any = None
        if judge is not None:
            cb = TrialCallback(judge)
            entrypoint = cb.entrypoint

        def compute_processor(engine: ExecutionEngine, df: DataFrame) -> DataFrame:
            out_schema = df.schema + TUNE_REPORT_ADD_SCHEMA

            def get_rows() -> Iterable[Any]:
                for row in self._compute_transformer(
                    df.as_local().as_dict_iterable(), entrypoint=entrypoint
                ):
                    yield [row[k] for k in out_schema.names]

            # TODO: need to add back execution_engine for engine aware runners
            # t._execution_engine = engine  # type:ignore
            return ArrayDataFrame(get_rows(), out_schema)

        if not _dist:
            res = dataset.data.process(compute_processor)
        else:
            res = dataset.data.per_row().transform(
                self._compute_transformer,
                schema=f"*,{TUNE_REPORT_ADD_SCHEMA}",
                callback=entrypoint,
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

    def _compute_transformer(
        self,
        df: Iterable[Dict[str, Any]],
        entrypoint: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    ) -> Iterable[Dict[str, Any]]:
        j: TrialJudge = (
            NoOpTrailJudge() if entrypoint is None else RemoteTrialJudge(entrypoint)
        )
        for row in df:
            reports = self._local_process_row(row, j)
            for report in reports:
                j.judge(report)
                yield report.fill_dict(dict(row))

    def _local_process_row(
        self, row: Dict[str, Any], judge: TrialJudge
    ) -> List[TrialReport]:
        reports: List[TrialReport] = []
        for trial in get_trials_from_row(row):
            if judge.can_accept(trial):
                report = self._runner.run(self._objective, trial)
                report = report.with_sort_metric(
                    self._objective.generate_sort_metric(report.metric)
                )
                reports.append(report)
        return reports
