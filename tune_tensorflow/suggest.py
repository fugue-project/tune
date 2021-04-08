from typing import Any, List, Optional, Set, Tuple, Type

from tune import (
    Space,
    TrialReport,
    suggest_by_continuous_asha,
    suggest_by_hyperband,
    suggest_by_sha,
)

from tune_tensorflow.objective import KerasObjective
from tune_tensorflow.spec import KerasTrainingSpec
from tune_tensorflow.utils import extract_keras_spec


def suggest_keras_models_by_sha(
    space: Space,
    plan: List[Tuple[float, int]],
    train_df: Any = None,
    temp_path: str = "",
    partition_keys: Optional[List[str]] = None,
    top_n: int = 1,
    monitor: Any = None,
    distributed: Optional[bool] = None,
    execution_engine: Any = None,
    execution_engine_conf: Any = None,
) -> List[TrialReport]:
    return suggest_by_sha(
        objective=_get_objective(space),
        space=space,
        plan=plan,
        train_df=train_df,
        temp_path=temp_path,
        partition_keys=partition_keys,
        top_n=top_n,
        monitor=monitor,
        distributed=distributed,
        execution_engine=execution_engine,
        execution_engine_conf=execution_engine_conf,
    )


def suggest_keras_models_by_hyperband(
    space: Space,
    plans: List[List[Tuple[float, int]]],
    train_df: Any = None,
    temp_path: str = "",
    partition_keys: Optional[List[str]] = None,
    top_n: int = 1,
    monitor: Any = None,
    distributed: Optional[bool] = None,
    execution_engine: Any = None,
    execution_engine_conf: Any = None,
) -> List[TrialReport]:
    return suggest_by_hyperband(
        objective=_get_objective(space),
        space=space,
        plans=plans,
        train_df=train_df,
        temp_path=temp_path,
        partition_keys=partition_keys,
        top_n=top_n,
        monitor=monitor,
        distributed=distributed,
        execution_engine=execution_engine,
        execution_engine_conf=execution_engine_conf,
    )


def suggest_keras_models_by_continuous_asha(
    space: Space,
    plan: List[Tuple[float, int]],
    train_df: Any = None,
    temp_path: str = "",
    partition_keys: Optional[List[str]] = None,
    top_n: int = 1,
    monitor: Any = None,
    execution_engine: Any = None,
    execution_engine_conf: Any = None,
) -> List[TrialReport]:
    return suggest_by_continuous_asha(
        objective=_get_objective(space),
        space=space,
        plan=plan,
        train_df=train_df,
        temp_path=temp_path,
        partition_keys=partition_keys,
        top_n=top_n,
        monitor=monitor,
        execution_engine=execution_engine,
        execution_engine_conf=execution_engine_conf,
    )


def _get_objective(space: Space) -> KerasObjective:
    types: Set[Type[KerasTrainingSpec]] = set()
    for d in space:  # type: ignore
        types.add(extract_keras_spec(d))
    return KerasObjective(types)
