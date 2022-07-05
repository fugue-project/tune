import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import cloudpickle
import numpy as np
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
from triad import FileSystem, ParamDict, assert_or_throw, to_uuid
from tune._utils import from_base64, to_base64
from tune.concepts.flow import Trial
from tune.concepts.space import Space
from tune.constants import (
    TUNE_DATASET_DF_PREFIX,
    TUNE_DATASET_PARAMS_PREFIX,
    TUNE_DATASET_TRIALS,
    TUNE_PREFIX,
    TUNE_REPORT,
    TUNE_REPORT_ID,
    TUNE_REPORT_METRIC,
    TUNE_TEMP_PATH,
)
from tune.exceptions import TuneCompileError


class TuneDataset:
    """A Fugue :class:`~fugue.workflow.workflow.WorkflowDataFrame` with metadata
    representing all dataframes required for a tuning task.

    :param data: the Fugue :class:`~fugue.workflow.workflow.WorkflowDataFrame`
      containing all required dataframes
    :param dfs: the names of the dataframes
    :param keys: the common partition keys of all dataframes

    .. attention::

        Do not construct this class directly, please read
        :ref:`TuneDataset Tutorial </notebooks/tune_dataset.ipynb>`
        to find the right way
    """

    def __init__(self, data: WorkflowDataFrame, dfs: List[str], keys: List[str]):
        self._data = data.persist()
        self._dfs = dfs
        self._keys = keys

    @property
    def data(self) -> WorkflowDataFrame:
        """the Fugue :class:`~fugue.workflow.workflow.WorkflowDataFrame`
        containing all required dataframes
        """
        return self._data

    @property
    def dfs(self) -> List[str]:
        """All dataframe names (you can also find them part of the
        column names of :meth:`.data` )
        """
        return self._dfs

    @property
    def keys(self) -> List[str]:
        """Partition keys (columns) of :meth:`.data`"""
        return self._keys

    def split(self, weights: List[float], seed: Any) -> List["TuneDataset"]:
        """Split the dataset randomly to small partitions. This is useful for
        some algorithms such as Hyperband, because it needs different subset to
        run successive halvings with different parameters.

        :param weights: a list of numeric values. The length represents the number
          of splitd partitions, and the values represents the proportion of each
          partition
        :param seed: random seed for the split

        :returns: a list of sub-datasets

        .. code-block:: python

            # randomly split the data to two partitions 25% and 75%
            dataset.split([1, 3], seed=0)
            # same because weights will be normalized
            dataset.split([10, 30], seed=0)

        """

        def label(df: pd.DataFrame) -> pd.DataFrame:
            if seed is not None:
                np.random.seed(seed)
            w = np.array(weights)
            p = w / np.sum(w)
            df["__tune_split_id_"] = np.random.choice(len(weights), df.shape[0], p=p)
            return df.reset_index(drop=True)

        def select(df: pd.DataFrame, n: int) -> pd.DataFrame:
            return (
                df[df["__tune_split_id_"] == n]
                .drop(["__tune_split_id_"], axis=1)
                .reset_index(drop=True)
            )

        temp = self._data.process(label).persist()
        datasets: List["TuneDataset"] = []
        for i in range(len(weights)):
            datasets.append(
                TuneDataset(
                    temp.transform(
                        select, schema="*-__tune_split_id_", params=dict(n=i)
                    ),
                    self.dfs,
                    self.keys,
                )
            )
        return datasets


class TuneDatasetBuilder:
    """Builder of :class:`~.TuneDataset`, for details please read
    :ref:`TuneDataset Tutorial </notebooks/tune_dataset.ipynb>`

    :param space: searching space, see |SpaceTutorial|
    :param path: temp path to store searialized dataframe partitions
      , defaults to ""
    """

    def __init__(self, space: Space, path: str = ""):
        self._dfs_spec: List[Tuple[str, WorkflowDataFrame, str]] = []
        self._space = space
        self._path = path

    def add_df(
        self, name: str, df: WorkflowDataFrame, how: str = ""
    ) -> "TuneDatasetBuilder":
        """Add a dataframe to the dataset

        :param name: name of the dataframe, it will also create a
          ``__tune_df__<name>`` column in the dataset dataframe
        :param df: the dataframe to add.
        :param how: join type, can accept ``semi``, ``left_semi``,
          ``anti``, ``left_anti``, ``inner``, ``left_outer``,
          ``right_outer``, ``full_outer``, ``cross``
        :returns: the builder itself

        .. note::

            For the first dataframe you add, ``how`` should be empty.
            From the second dataframe you add, ``how`` must be set.

        .. note::

            If ``df`` is prepartitioned, the partition key will be used to
            join with the added dataframes. Read
            :ref:`TuneDataset Tutorial </notebooks/tune_dataset.ipynb>`
            for more details
        """
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
        """Add multiple dataframes with the same join type

        :param dfs: dictionary like dataframe collection. The keys
          will be used as the dataframe names
        :param how: join type, can accept ``semi``, ``left_semi``,
          ``anti``, ``left_anti``, ``inner``, ``left_outer``,
          ``right_outer``, ``full_outer``, ``cross``
        :returns: the builder itself
        """
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
        """Build :class:`~.TuneDataset`, for details please read
        :ref:`TuneDataset Tutorial </notebooks/tune_dataset.ipynb>`

        :param wf: the workflow associated with the dataset
        :param batch_size: how many configurations as a batch, defaults to 1
        :param shuffle: whether to shuffle the entire dataset, defaults to True.
          This is to make the tuning process more even, it will look better. It
          should have slight benefit on speed, no effect on result.
        :param trial_metadata: metadata to pass to each |Trial|, defaults to None
        :return: the dataset for tuning
        """
        space = self._space_to_df(wf=wf, batch_size=batch_size, shuffle=shuffle)
        if len(self._dfs_spec) == 0:
            res = space
            keys: List[str] = []
        else:
            dfs, keys = self._serialize_dfs()
            res = dfs.cross_join(space)

        def finalize(df: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
            for row in df:
                yield _to_trail_row(row, trial_metadata or {})

        data = res.transform(
            finalize,
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
                    first = df.peek_dict()
                    keys = [first[x] for x in self.key_schema.names]
                    fs = FileSystem()
                    with fs.open(fp, "wb") as handler:
                        df.as_pandas().to_parquet(handler)
                    return ArrayDataFrame([keys + [fp]], self.output_schema)

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
                    yield [cloudpickle.dumps(res)]
                    res = []
            if len(res) > 0:
                yield [cloudpickle.dumps(res)]

        return wf.df(
            IterableDataFrame(get_data(), f"{TUNE_DATASET_PARAMS_PREFIX}:binary")
        )


class StudyResult:
    """A collection of the input :class:`~.TuneDataset` and the tuning result

    :param dataset: input dataset for tuning
    :param result: tuning result as a dataframe

    .. attention::

        Do not construct this class directly.
    """

    def __init__(self, dataset: TuneDataset, result: WorkflowDataFrame):
        self._dataset = dataset
        self._result = (
            result.persist()
            .partition_by(TUNE_REPORT_ID, presort=TUNE_REPORT_METRIC)
            .take(1)
            .persist()
        )

    def result(self, best_n: int = 0) -> WorkflowDataFrame:
        """Get the top n results sorted by |SortMetric|

        :param best_n: number of result to get, defaults to 0.
          if `<=0` then it will return the entire result
        :return: result subset
        """
        if best_n <= 0:
            return self._result
        if len(self._dataset.keys) == 0:
            return self._result.take(n=best_n, presort=TUNE_REPORT_METRIC)
        else:
            return self._result.partition(
                by=self._dataset.keys, presort=TUNE_REPORT_METRIC
            ).take(best_n)

    def next_tune_dataset(self, best_n: int = 0) -> TuneDataset:
        """Convert the result back to a new :class:`~.TuneDataset` to be
        used by the next steps.

        :param best_n: top n result to extract, defaults to 0 (entire result)
        :return: a new dataset for tuning
        """
        data = self.result(best_n).drop(
            [TUNE_REPORT_ID, TUNE_REPORT_METRIC, TUNE_REPORT], if_exists=True
        )
        return TuneDataset(data, dfs=self._dataset.dfs, keys=self._dataset.keys)

    def union_with(self, other: "StudyResult") -> None:
        """Union with another result set and update itself

        :param other: the other result dataset

        .. note::
            This method also removes duplicated reports based on
            :meth:`tune.concepts.flow.trial.Trial.trial_id`. Each
            trial will have only the best report in the updated
            result
        """
        self._result = (
            self._result.union(other._result)
            .partition_by(TUNE_REPORT_ID, presort=TUNE_REPORT_METRIC)
            .take(1)
            .persist()
        )


def _to_trail_row(data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    key_names = sorted(k for k in data.keys() if not k.startswith(TUNE_PREFIX))
    keys = [data[k] for k in key_names]
    trials: Dict[str, Trial] = {}
    for params in cloudpickle.loads(data[TUNE_DATASET_PARAMS_PREFIX]):
        tid = to_uuid(keys, params)
        trials[tid] = Trial(trial_id=tid, params=params, metadata=metadata, keys=keys)
    data[TUNE_DATASET_TRIALS] = to_base64(list(trials.values()))
    del data[TUNE_DATASET_PARAMS_PREFIX]
    return data


def _get_trials_from_row(row: Dict[str, Any], with_dfs: bool = True) -> Iterable[Trial]:
    if not with_dfs:
        yield from from_base64(row[TUNE_DATASET_TRIALS])
    else:
        dfs: Dict[str, Any] = {}
        fs = FileSystem()
        for k, v in row.items():
            if k.startswith(TUNE_DATASET_DF_PREFIX):
                key = k[len(TUNE_DATASET_DF_PREFIX) :]
                if v is not None:
                    with fs.open(v, "rb") as handler:
                        dfs[key] = pd.read_parquet(handler)
        for params in from_base64(row[TUNE_DATASET_TRIALS]):
            yield params.with_dfs(dfs)
