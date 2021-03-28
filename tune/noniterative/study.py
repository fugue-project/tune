import json
from typing import Any, Dict, Iterable, Optional, Set

import pandas as pd
from fugue import ArrayDataFrame, DataFrame, ExecutionEngine, WorkflowDataFrame
from triad import assert_or_throw
from tune.constants import (
    TUNE_DATASET_DF_PREFIX,
    TUNE_DATASET_TRIALS,
    TUNE_REPORT,
    TUNE_REPORT_METRIC,
)
from tune.dataset import TuneDataset
from tune.exceptions import TuneCompileError
from tune.noniterative.objective import (
    NonIterativeObjectiveFunc,
    NonIterativeObjectiveRunner,
)
from tune.trial import Trial


class StudyResult:
    def __init__(
        self, dataset: TuneDataset, result: WorkflowDataFrame, min_better: bool
    ):
        self._dataset = dataset
        self._result = result.persist()
        self._min_better = min_better

    def result(self, best_n: int = 0) -> WorkflowDataFrame:
        if best_n <= 0:
            return self._result
        if self._min_better:
            presort = TUNE_REPORT_METRIC
        else:
            presort = TUNE_REPORT_METRIC + " desc"
        if len(self._dataset.keys) == 0:
            return self._result.take(n=best_n, presort=presort)
        else:
            return self._result.partition(by=self._dataset.keys, presort=presort).take(
                best_n
            )


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
                yield from self._process_row(row)

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

        return StudyResult(
            dataset=dataset, result=res, min_better=self._objective.min_better
        )

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

    def _process_row(self, row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        dfs: Dict[str, Any] = {}
        dfs_keys: Set[str] = set()
        for k, v in row.items():
            if k.startswith(TUNE_DATASET_DF_PREFIX):
                key = k[len(TUNE_DATASET_DF_PREFIX) :]
                if v is not None:
                    dfs[key] = pd.read_parquet(v)
                dfs_keys.add(key)
        for params in json.loads(row[TUNE_DATASET_TRIALS]):
            trial = Trial.from_jsondict(params).with_dfs(dfs)
            report = self._runner.run(self._objective, trial)
            res = dict(row)
            res[TUNE_REPORT_METRIC] = report.metric
            res[TUNE_REPORT] = json.dumps(report.jsondict)
            yield res
