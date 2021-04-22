from tune.constants import (
    TUNE_DATASET_DF_DEFAULT_NAME,
    TUNE_DATASET_VALIDATION_DF_DEFAULT_NAME,
)
from tune.space import Space
from tune.trial import Monitor
from typing import Any, Callable, List, Optional

from triad import assert_or_throw
from fugue import FugueWorkflow
from tune.exceptions import TuneCompileError
from tune.iterative.objective import IterativeObjectiveFunc
from tune.noniterative.objective import (
    NonIterativeObjectiveFunc,
    NonIterativeObjectiveRunner,
)
from tune.dataset import TuneDataset, TuneDatasetBuilder


class TuneObjectFactory:
    def __init__(self):
        self._iterative_objective_converter = self._object_to_iterative_objective
        self._noniterative_objective_converter = self._object_to_noniterative_objective
        self._noniterative_objective_runner_converter = (
            self._object_to_noniterative_objective_runner
        )
        self._monitor_converter = self._object_to_monitor
        self._tmp = ""

    def set_iterative_objective_converter(
        self, func: Callable[[Any], IterativeObjectiveFunc]
    ) -> None:
        self._iterative_objective_converter = func

    def set_noniterative_objective_converter(
        self, func: Callable[[Any], NonIterativeObjectiveFunc]
    ) -> None:
        self._noniterative_objective_converter = func

    def set_noniterative_objective_runner_converter(
        self, func: Callable[[Any], NonIterativeObjectiveRunner]
    ) -> None:
        self._noniterative_objective_runner_converter = func

    def set_monitor_converter(self, func: Callable[[Any], Optional[Monitor]]) -> None:
        self._monitor_converter = func

    def set_temp_path(self, path: str) -> None:
        self._tmp = path

    def make_iterative_objective(self, obj: Any) -> IterativeObjectiveFunc:
        assert_or_throw(obj is not None, TuneCompileError("objective can't be None"))
        if isinstance(obj, IterativeObjectiveFunc):
            return obj
        return self._iterative_objective_converter(obj)

    def make_noniterative_objective(self, obj: Any) -> NonIterativeObjectiveFunc:
        assert_or_throw(obj is not None, TuneCompileError("objective can't be None"))
        if isinstance(obj, NonIterativeObjectiveFunc):
            return obj
        return self._noniterative_objective_converter(obj)

    def make_noniterative_objective_runner(
        self, obj: Any
    ) -> NonIterativeObjectiveRunner:
        if isinstance(obj, NonIterativeObjectiveRunner):
            return obj
        return self._noniterative_objective_runner_converter(obj)

    def make_monitor(self, obj: Any) -> Optional[Monitor]:
        if isinstance(obj, Monitor):
            return obj
        return self._monitor_converter(obj)

    def make_dataset(
        self,
        dag: FugueWorkflow,
        dataset: Any,
        df: Any = None,
        df_name: str = TUNE_DATASET_DF_DEFAULT_NAME,
        test_df: Any = None,
        test_df_name: str = TUNE_DATASET_VALIDATION_DF_DEFAULT_NAME,
        partition_keys: Optional[List[str]] = None,
        temp_path: str = "",
    ) -> TuneDataset:
        assert_or_throw(dataset is not None, TuneCompileError("dataset can't be None"))
        if isinstance(dataset, TuneDataset):
            assert_or_throw(
                df is None, TuneCompileError("can't set df when dataset is TuneDataset")
            )
            return dataset
        if isinstance(dataset, Space):
            path = self.get_path_or_temp(temp_path)
            builder = TuneDatasetBuilder(dataset, path)
            if df is not None:
                wdf = dag.df(df)
                if partition_keys is not None and len(partition_keys) > 0:
                    wdf = wdf.partition_by(*partition_keys)
                builder.add_df(df_name, wdf)
            if test_df is not None:
                wdf = dag.df(test_df)
                how = "cross"
                if partition_keys is not None and len(partition_keys) > 0:
                    wdf = wdf.partition_by(*partition_keys)
                    how = "inner"
                builder.add_df(test_df_name, wdf, how=how)
            return builder.build(dag, batch_size=1, shuffle=True)
        raise TuneCompileError(f"{dataset} can't be converted to TuneDataset")

    def get_path_or_temp(self, path: str) -> str:
        if path is None or path == "":
            path = self._tmp
        assert_or_throw(path != "", TuneCompileError("path or temp path must be set"))
        return path

    def _object_to_iterative_objective(self, obj: Any) -> IterativeObjectiveFunc:
        raise TuneCompileError(
            f"{obj} can't be converted to iterative objective function"
        )

    def _object_to_noniterative_objective(self, obj: Any) -> NonIterativeObjectiveFunc:
        raise TuneCompileError(
            f"{obj} can't be converted to non iterative objective function"
        )

    def _object_to_noniterative_objective_runner(
        self, obj: Any
    ) -> NonIterativeObjectiveRunner:
        if obj is None:
            return NonIterativeObjectiveRunner()
        raise TuneCompileError(
            f"{obj} can't be converted to non iterative objective runner"
        )

    def _object_to_monitor(self, obj: Any) -> Optional[Monitor]:
        if obj is None:
            return None
        raise TuneCompileError(f"{obj} can't be converted to Monitor")


TUNE_OBJECT_FACTORY = TuneObjectFactory()
