from typing import Any, List, Optional, Tuple

from tune import (
    Space,
    TrialReport,
    suggest_by_continuous_asha,
    suggest_by_hyperband,
    suggest_by_sha,
)

from tune_tensorflow.objective import KerasObjective
from tune_tensorflow.utils import _TYPE_DICT


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
        objective=KerasObjective(_TYPE_DICT),
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
        objective=KerasObjective(_TYPE_DICT),
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
        objective=KerasObjective(_TYPE_DICT),
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
