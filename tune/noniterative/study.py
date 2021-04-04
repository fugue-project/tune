import json
from typing import Any, Dict, Iterable, Optional

from fugue import ArrayDataFrame, DataFrame, ExecutionEngine
from triad import assert_or_throw
from tune.constants import TUNE_REPORT, TUNE_REPORT_METRIC
from tune.dataset import StudyResult, TuneDataset, get_trials_from_row
from tune.exceptions import TuneCompileError
from tune.noniterative.objective import (
    NonIterativeObjectiveFunc,
    NonIterativeObjectiveRunner,
)


def run_noniterative_study(
    objective: NonIterativeObjectiveFunc,
    dataset: TuneDataset,
    runner: Optional[NonIterativeObjectiveRunner] = None,
    distributed: Optional[bool] = None,
) -> StudyResult:
    study = NonIterativeStudy(objective, runner or NonIterativeObjectiveRunner())
    return study.optimize(dataset, distributed=distributed)


class NonIterativeStudy:
    def __init__(
        self, objective: NonIterativeObjectiveFunc, runner: NonIterativeObjectiveRunner
    ):
        self._objective = objective
        self._runner = runner

    def optimize(
        self, dataset: TuneDataset, distributed: Optional[bool] = None
    ) -> StudyResult:
        _dist = self._get_distributed(distributed)
        add_schema = f"{TUNE_REPORT_METRIC}:double,{TUNE_REPORT}:str"

        def compute_transformer(
            df: Iterable[Dict[str, Any]]
        ) -> Iterable[Dict[str, Any]]:
            for row in df:
                for trial in get_trials_from_row(row):
                    report = self._runner.run(self._objective, trial)
                    report = report.with_sort_metric(
                        self._objective.generate_sort_metric(report.metric)
                    )
                    res = dict(row)
                    res[TUNE_REPORT_METRIC] = report.sort_metric
                    res[TUNE_REPORT] = json.dumps(report.jsondict)
                    yield res

        def compute_processor(engine: ExecutionEngine, df: DataFrame) -> DataFrame:
            out_schema = df.schema + add_schema

            def get_rows() -> Iterable[Any]:
                for row in compute_transformer(df.as_local().as_dict_iterable()):
                    yield [row[k] for k in out_schema.names]

            # TODO: need to add back execution_engine for engine aware runners
            # t._execution_engine = engine  # type:ignore
            return ArrayDataFrame(get_rows(), out_schema)

        if not _dist:
            res = dataset.data.process(compute_processor)
        else:
            res = dataset.data.per_row().transform(
                compute_transformer, schema=f"*,{add_schema}"
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
