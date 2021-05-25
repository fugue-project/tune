from typing import Any, Callable, List, Optional

from fugue import FugueWorkflow
from triad import assert_or_throw
from tune.concepts.dataset import TuneDataset, TuneDatasetBuilder
from tune.concepts.flow import Monitor
from tune.concepts.space import Space
from tune.constants import (
    TUNE_DATASET_DF_DEFAULT_NAME,
    TUNE_DATASET_VALIDATION_DF_DEFAULT_NAME,
)
from tune.exceptions import TuneCompileError
from tune.iterative.objective import IterativeObjectiveFunc
from tune.noniterative.convert import to_noniterative_objective
from tune.noniterative.objective import (
    NonIterativeObjectiveFunc,
    NonIterativeObjectiveLocalOptimizer,
)
from tune.noniterative.stopper import NonIterativeStopper


class TuneObjectFactory:
    def __init__(self):
        self._iterative_objective_converter = self._object_to_iterative_objective
        self._noniterative_objective_converter = self._object_to_noniterative_objective
        self._noniterative_local_optimizer_converter = (
            self._object_to_noniterative_local_optimizer
        )
        self._monitor_converter = self._object_to_monitor
        self._stopper_converter = self._object_to_stopper
        self._tmp = ""

    def set_iterative_objective_converter(
        self, func: Callable[[Any], IterativeObjectiveFunc]
    ) -> None:
        self._iterative_objective_converter = func

    def set_noniterative_objective_converter(
        self, func: Callable[[Any], NonIterativeObjectiveFunc]
    ) -> None:
        self._noniterative_objective_converter = func

    def set_noniterative_local_optimizer_converter(
        self, func: Callable[[Any], NonIterativeObjectiveLocalOptimizer]
    ) -> None:
        self._noniterative_local_optimizer_converter = func

    def set_noniterative_stopper_converter(
        self, func: Callable[[Any], NonIterativeStopper]
    ) -> None:
        self._stopper_converter = func

    def set_monitor_converter(self, func: Callable[[Any], Optional[Monitor]]) -> None:
        self._monitor_converter = func

    def set_temp_path(self, path: str) -> None:
        self._tmp = path

    def make_iterative_objective(self, obj: Any) -> IterativeObjectiveFunc:
        return self._iterative_objective_converter(obj)

    def make_noniterative_objective(self, obj: Any) -> NonIterativeObjectiveFunc:
        return self._noniterative_objective_converter(obj)

    def make_noniterative_local_optimizer(
        self, obj: Any
    ) -> NonIterativeObjectiveLocalOptimizer:
        return self._noniterative_local_optimizer_converter(obj)

    def make_monitor(self, obj: Any) -> Optional[Monitor]:
        return self._monitor_converter(obj)

    def make_noniterative_stopper(self, obj: Any) -> Optional[NonIterativeStopper]:
        return self._stopper_converter(obj)

    def make_dataset(
        self,
        dag: FugueWorkflow,
        dataset: Any,
        df: Any = None,
        df_name: str = TUNE_DATASET_DF_DEFAULT_NAME,
        test_df: Any = None,
        test_df_name: str = TUNE_DATASET_VALIDATION_DF_DEFAULT_NAME,
        partition_keys: Optional[List[str]] = None,
        shuffle: bool = True,
        temp_path: str = "",
    ) -> TuneDataset:
        assert_or_throw(dataset is not None, TuneCompileError("dataset can't be None"))
        if isinstance(dataset, TuneDataset):
            assert_or_throw(
                df is None, TuneCompileError("can't set df when dataset is TuneDataset")
            )
            return dataset
        if isinstance(dataset, Space):
            if df is None and test_df is None:
                path = ""
            else:
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
            return builder.build(dag, batch_size=1, shuffle=shuffle)
        raise TuneCompileError(f"{dataset} can't be converted to TuneDataset")

    def get_path_or_temp(self, path: str) -> str:
        if path is None or path == "":
            path = self._tmp
        assert_or_throw(path != "", TuneCompileError("path or temp path must be set"))
        return path

    def _object_to_iterative_objective(self, obj: Any) -> IterativeObjectiveFunc:
        assert_or_throw(obj is not None, TuneCompileError("objective can't be None"))
        if isinstance(obj, IterativeObjectiveFunc):
            return obj
        raise TuneCompileError(
            f"{obj} can't be converted to iterative objective function"
        )

    def _object_to_noniterative_objective(self, obj: Any) -> NonIterativeObjectiveFunc:
        assert_or_throw(obj is not None, TuneCompileError("objective can't be None"))
        if isinstance(obj, NonIterativeObjectiveFunc):
            return obj
        if callable(obj):
            return to_noniterative_objective(obj)
        raise TuneCompileError(
            f"{obj} can't be converted to non iterative objective function"
        )

    def _object_to_noniterative_local_optimizer(
        self, obj: Any
    ) -> NonIterativeObjectiveLocalOptimizer:
        if isinstance(obj, NonIterativeObjectiveLocalOptimizer):
            return obj
        if obj is None:
            return NonIterativeObjectiveLocalOptimizer()
        raise TuneCompileError(
            f"{obj} can't be converted to non iterative objective optimizer"
        )

    def _object_to_monitor(self, obj: Any) -> Optional[Monitor]:
        if isinstance(obj, Monitor):
            return obj
        if obj is None:
            return None
        raise TuneCompileError(f"{obj} can't be converted to Monitor")

    def _object_to_stopper(self, obj: Any) -> Optional[NonIterativeStopper]:
        if isinstance(obj, NonIterativeStopper):
            return obj
        if obj is None:
            return None
        raise TuneCompileError(f"{obj} can't be converted to NonIterativeStopper")


TUNE_OBJECT_FACTORY = TuneObjectFactory()
