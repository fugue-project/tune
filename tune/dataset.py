import json
import os
import pickle
import random
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from uuid import uuid4

import pandas as pd
from fugue import (
    ArrayDataFrame,
    DataFrame,
    ExecutionEngine,
    FugueWorkflow,
    IterableDataFrame,
    LocalDataFrame,
    Transformer,
    WorkflowDataFrame,
    WorkflowDataFrames,
)
from triad import ParamDict, assert_or_throw, to_uuid

from tune.constants import (
    TUNE_DATASET_DF_PREFIX,
    TUNE_DATASET_PARAMS_PREFIX,
    TUNE_DATASET_TRIALS,
    TUNE_REPORT_METRIC,
    TUNE_TEMP_PATH,
)
from tune.exceptions import TuneCompileError
from tune.space import Space
from tune.trial import Trial


class TuneDataset:
    def __init__(self, data: WorkflowDataFrame, dfs: List[str], keys: List[str]):
        self._data = data.persist()
        self._dfs = dfs
        self._keys = keys

    @property
    def data(self) -> WorkflowDataFrame:
        return self._data

    @property
    def dfs(self) -> List[str]:
        return self._dfs

    @property
    def keys(self) -> List[str]:
        return self._keys


def get_trials_from_row(row: Dict[str, Any]) -> Iterable[Trial]:
    dfs: Dict[str, Any] = {}
    dfs_keys: Set[str] = set()
    for k, v in row.items():
        if k.startswith(TUNE_DATASET_DF_PREFIX):
            key = k[len(TUNE_DATASET_DF_PREFIX) :]
            if v is not None:
                dfs[key] = pd.read_parquet(v)
            dfs_keys.add(key)
    for params in json.loads(row[TUNE_DATASET_TRIALS]):
        yield Trial.from_jsondict(params).with_dfs(dfs)


class TuneDatasetBuilder:
    def __init__(self, space: Space, path: str = ""):
        self._dfs_spec: List[Tuple[str, WorkflowDataFrame, str]] = []
        self._space = space
        self._path = path

    def add_df(
        self, name: str, df: WorkflowDataFrame, how: str = ""
    ) -> "TuneDatasetBuilder":
        assert_or_throw(
            not any(r[0] == name for r in self._dfs_spec),
            TuneCompileError(name + " already exists"),
        )
        if len(self._dfs_spec) == 0:
            assert_or_throw(
                how == "", TuneCompileError("first dataframe can't specify how to join")
            )
        else:
            assert_or_throw(
                how != "",
                TuneCompileError("must specify how to join after first dataframe"),
            )
        self._dfs_spec.append((name, df, how))
        return self

    def add_dfs(self, dfs: WorkflowDataFrames, how: str = "") -> "TuneDatasetBuilder":
        assert_or_throw(dfs.has_key, "all datarames must be named")
        for k, v in dfs.items():
            if len(self._dfs_spec) == 0:
                self.add_df(k, v)
            else:
                self.add_df(k, v, how=how)
        return self

    def build(
        self,
        wf: FugueWorkflow,
        batch_size: int = 1,
        shuffle: bool = True,
        trial_metadata: Optional[Dict[str, Any]] = None,
    ) -> TuneDataset:
        space = self._space_to_df(wf=wf, batch_size=batch_size, shuffle=shuffle)
        if len(self._dfs_spec) == 0:
            res = space
            keys: List[str] = []
        else:
            dfs, keys = self._serialize_dfs()
            res = dfs.cross_join(space)

        def postprocess(df: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
            for row in df:
                yield _to_trail_row(row, trial_metadata or {})

        data = res.transform(
            postprocess,
            schema=f"*,{TUNE_DATASET_TRIALS}:str-{TUNE_DATASET_PARAMS_PREFIX}",
        )
        return TuneDataset(data, [x[0] for x in self._dfs_spec], keys)

    def _serialize_df(self, df: WorkflowDataFrame, name: str) -> WorkflowDataFrame:
        pre_partition = df.partition_spec
        path = self._path

        def _get_temp_path(p: str, conf: ParamDict) -> str:
            if p is not None and p != "":
                return p
            return conf.get_or_throw(TUNE_TEMP_PATH, str)

        if len(pre_partition.partition_by) == 0:

            def save_single_file(e: ExecutionEngine, _input: DataFrame) -> DataFrame:
                p = _get_temp_path(path, e.conf)
                fp = os.path.join(p, str(uuid4()) + ".parquet")
                e.save_df(_input, fp, force_single=True)
                return ArrayDataFrame([[fp]], f"{TUNE_DATASET_DF_PREFIX}{name}:str")

            return df.process(save_single_file)
        else:

            class SavePartition(Transformer):
                def get_output_schema(self, df: DataFrame) -> Any:
                    dfn = self.params.get_or_throw("name", str)
                    return self.key_schema + f"{TUNE_DATASET_DF_PREFIX}{dfn}:str"

                def transform(self, df: LocalDataFrame) -> LocalDataFrame:
                    p = _get_temp_path(self.params.get("path", ""), self.workflow_conf)
                    fp = os.path.join(p, str(uuid4()) + ".parquet")
                    df.as_pandas().to_parquet(fp)
                    return ArrayDataFrame(
                        [self.cursor.key_value_array + [fp]], self.output_schema
                    )

            return df.transform(SavePartition, params={"path": path, "name": name})

    def _serialize_dfs(self) -> Tuple[WorkflowDataFrame, List[str]]:
        df = self._serialize_df(self._dfs_spec[0][1], self._dfs_spec[0][0])
        keys = list(self._dfs_spec[0][1].partition_spec.partition_by)
        for i in range(1, len(self._dfs_spec)):
            how = self._dfs_spec[i][2]
            new_keys = set(self._dfs_spec[i][1].partition_spec.partition_by)
            next_df = self._serialize_df(self._dfs_spec[i][1], self._dfs_spec[i][0])
            df = df.join(next_df, how=how)
            if how != "cross":
                keys = [k for k in keys if k in new_keys]
        return df, keys

    def _space_to_df(
        self, wf: FugueWorkflow, batch_size: int = 1, shuffle: bool = True
    ) -> WorkflowDataFrame:
        def get_data() -> Iterable[List[Any]]:
            it = list(self._space)  # type: ignore
            if shuffle:
                random.seed(0)
                random.shuffle(it)
            res: List[Any] = []
            for a in it:
                res.append(a)
                if batch_size == len(res):
                    yield [pickle.dumps(res)]
                    res = []
            if len(res) > 0:
                yield [pickle.dumps(res)]

        return wf.df(
            IterableDataFrame(get_data(), f"{TUNE_DATASET_PARAMS_PREFIX}:binary")
        )


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


def _to_trail_row(data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    key_names = sorted(
        k
        for k in data.keys()
        if not k.startswith(TUNE_DATASET_DF_PREFIX)
        and not k.startswith(TUNE_DATASET_PARAMS_PREFIX)
    )
    keys = [data[k] for k in key_names]
    trials: Dict[str, Dict[str, Any]] = {}
    for param in pickle.loads(data[TUNE_DATASET_PARAMS_PREFIX]):
        p = ParamDict(sorted(((k, v) for k, v in param.items()), key=lambda x: x[0]))
        tid = to_uuid(keys, p)
        trials[tid] = Trial(trial_id=tid, params=p, metadata=metadata).jsondict
    data[TUNE_DATASET_TRIALS] = json.dumps(list(trials.values()))
    del data[TUNE_DATASET_PARAMS_PREFIX]
    return data
