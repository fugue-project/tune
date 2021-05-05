from typing import Any, Callable, Dict, Iterable, Optional

from fugue import ArrayDataFrame, DataFrame, ExecutionEngine
from triad import assert_or_throw
from tune._utils import run_monitored_process
from tune.concepts.dataset import StudyResult, TuneDataset, get_trials_from_row
from tune.concepts.flow import RemoteTrialJudge, TrialCallback, TrialJudge, TrialReport
from tune.constants import TUNE_REPORT_ADD_SCHEMA
from tune.exceptions import TuneCompileError, TuneInterrupted
from tune.noniterative.objective import (
    NonIterativeObjectiveFunc,
    NonIterativeObjectiveRunner,
)


class NonIterativeStudy:
    def __init__(
        self, objective: NonIterativeObjectiveFunc, runner: NonIterativeObjectiveRunner
    ):
        self._objective = objective
        self._runner = runner

    def optimize(  # noqa: C901
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

        def preprocess(df: DataFrame) -> DataFrame:
            if judge is not None:
                judge.monitor.initialize()
            return df

        def postprocess(df: DataFrame) -> None:
            if judge is not None:
                judge.monitor.finalize()

        if not _dist:
            res = dataset.data.process(preprocess).process(compute_processor)
        else:
            res = (
                dataset.data.process(preprocess)
                .per_row()
                .transform(
                    self._compute_transformer,
                    schema=f"*,{TUNE_REPORT_ADD_SCHEMA}",
                    callback=entrypoint,
                )
            )

        res.persist().output(postprocess)

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
        j: Optional[RemoteTrialJudge] = (
            None if entrypoint is None else RemoteTrialJudge(entrypoint)
        )
        for row in df:
            for n, trial in enumerate(get_trials_from_row(row, with_dfs=False)):
                if j is not None:
                    if j.can_accept(trial):
                        try:
                            report = run_monitored_process(
                                self._local_process_trial,
                                [row, n],
                                {},
                                lambda: j.can_accept(trial),  # type: ignore
                                "60sec",
                            )
                        except TuneInterrupted:
                            continue
                        except Exception:
                            raise
                        j.judge(report)
                        yield report.fill_dict(dict(row))
                else:
                    report = self._local_process_trial(row, n)
                    yield report.fill_dict(dict(row))

    def _local_process_trial(self, row: Dict[str, Any], idx: int) -> TrialReport:
        trial = list(get_trials_from_row(row))[idx]
        report = self._runner.run(self._objective, trial)
        return report.with_sort_metric(
            self._objective.generate_sort_metric(report.metric)
        )
