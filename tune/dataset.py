import json
import os
import random
from typing import Any, Iterable, List, Tuple
from uuid import uuid4

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
from triad import ParamDict, assert_or_throw

from tune.constants import (
    TUNE_DATASET_DF_PREFIX,
    TUNE_DATASET_PARAMS_PREFIX,
    TUNE_TEMP_PATH,
)
from tune.exceptions import TuneCompileError
from tune.space import Space


class TuneDatasetBuilder:
    def __init__(self, space: Space, path: str = ""):
        self._dfs_spec: List[Tuple[str, WorkflowDataFrame, str]] = []
        self._space = space
        self._path = path

    def add_df(self, name: str, df: WorkflowDataFrame, how: str = "") -> None:
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

    def add_dfs(self, dfs: WorkflowDataFrames, how: str = "") -> None:
        assert_or_throw(dfs.has_key, "all datarames must be named")
        for k, v in dfs.items():
            if len(self._dfs_spec) == 0:
                self.add_df(k, v)
            else:
                self.add_df(k, v, how=how)

    def build(
        self, wf: FugueWorkflow, batch_size: int = 1, shuffle: bool = True
    ) -> WorkflowDataFrame:
        space = self._space_to_df(wf=wf, batch_size=batch_size, shuffle=shuffle)
        if len(self._dfs_spec) == 0:
            return space
        else:
            return self._serialize_dfs().cross_join(space)

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

    def _serialize_dfs(self) -> WorkflowDataFrame:
        df = self._serialize_df(self._dfs_spec[0][1], self._dfs_spec[0][0])
        for i in range(1, len(self._dfs_spec)):
            next_df = self._serialize_df(self._dfs_spec[i][1], self._dfs_spec[i][0])
            df = df.join(next_df, how=self._dfs_spec[i][2])
        return df

    def _space_to_df(
        self, wf: FugueWorkflow, batch_size: int = 1, shuffle: bool = True
    ) -> WorkflowDataFrame:
        def get_data() -> Iterable[List[Any]]:
            it = list(self._space.encode())  # type: ignore
            if shuffle:
                random.seed(0)
                random.shuffle(it)
            res: List[Any] = []
            for a in it:
                res.append(a)
                if batch_size == len(res):
                    yield [json.dumps(res)]
                    res = []
            if len(res) > 0:
                yield [json.dumps(res)]

        return wf.df(IterableDataFrame(get_data(), f"{TUNE_DATASET_PARAMS_PREFIX}:str"))
