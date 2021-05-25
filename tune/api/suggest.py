import json
from typing import Any, List, Optional, Tuple

from fugue import FugueWorkflow
from fugue.exceptions import FugueDataFrameError
from triad import assert_or_throw
from tune.api.factory import TUNE_OBJECT_FACTORY
from tune.api.optimize import (
    optimize_by_continuous_asha,
    optimize_by_hyperband,
    optimize_by_sha,
    optimize_noniterative,
)
from tune.concepts.flow import TrialReport
from tune.concepts.space import Space
from tune.constants import TUNE_DATASET_DF_DEFAULT_NAME, TUNE_REPORT, TUNE_REPORT_METRIC
from tune.exceptions import TuneCompileError


def suggest_for_noniterative_objective(
    objective: Any,
    space: Space,
    df: Any = None,
    df_name: str = TUNE_DATASET_DF_DEFAULT_NAME,
    temp_path: str = "",
    partition_keys: Optional[List[str]] = None,
    top_n: int = 1,
    local_optimizer: Any = None,
    monitor: Any = None,
    stopper: Any = None,
    stop_check_interval: Any = None,
    distributed: Optional[bool] = None,
    shuffle_candidates: bool = True,
    execution_engine: Any = None,
    execution_engine_conf: Any = None,
) -> List[TrialReport]:
    dag = FugueWorkflow()
    dataset = TUNE_OBJECT_FACTORY.make_dataset(
        dag,
        space,
        df=df,
        df_name=df_name,
        partition_keys=partition_keys,
        temp_path=temp_path,
        shuffle=shuffle_candidates,
    )
    study = optimize_noniterative(
        objective=objective,
        dataset=dataset,
        optimizer=local_optimizer,
        distributed=distributed,
        monitor=monitor,
        stopper=stopper,
        stop_check_interval=stop_check_interval,
    )
    study.result(top_n).yield_dataframe_as("result")

    return _run(
        dag=dag,
        execution_engine=execution_engine,
        execution_engine_conf=execution_engine_conf,
    )


def suggest_by_sha(
    objective: Any,
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
    assert_or_throw(
        not space.has_random_parameter,
        TuneCompileError(
            "space can't contain random parameters, "
            "use sample method before calling this function"
        ),
    )
    dag = FugueWorkflow()
    dataset = TUNE_OBJECT_FACTORY.make_dataset(
        dag,
        space,
        df=train_df,
        partition_keys=partition_keys,
        temp_path=temp_path,
    )
    study = optimize_by_sha(
        objective=objective,
        dataset=dataset,
        plan=plan,
        checkpoint_path=temp_path,
        distributed=distributed,
        monitor=monitor,
    )
    study.result(top_n).yield_dataframe_as("result")

    return _run(
        dag=dag,
        execution_engine=execution_engine,
        execution_engine_conf=execution_engine_conf,
    )


def suggest_by_hyperband(
    objective: Any,
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
    assert_or_throw(
        not space.has_random_parameter,
        TuneCompileError(
            "space can't contain random parameters, "
            "use sample method before calling this function"
        ),
    )
    dag = FugueWorkflow()
    dataset = TUNE_OBJECT_FACTORY.make_dataset(
        dag,
        space,
        df=train_df,
        partition_keys=partition_keys,
        temp_path=temp_path,
    )
    study = optimize_by_hyperband(
        objective=objective,
        dataset=dataset,
        plans=plans,
        checkpoint_path=temp_path,
        distributed=distributed,
        monitor=monitor,
    )
    study.result(top_n).yield_dataframe_as("result")

    return _run(
        dag=dag,
        execution_engine=execution_engine,
        execution_engine_conf=execution_engine_conf,
    )


def suggest_by_continuous_asha(
    objective: Any,
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
    assert_or_throw(
        not space.has_random_parameter,
        TuneCompileError(
            "space can't contain random parameters, "
            "use sample method before calling this function"
        ),
    )
    dag = FugueWorkflow()
    dataset = TUNE_OBJECT_FACTORY.make_dataset(
        dag,
        space,
        df=train_df,
        partition_keys=partition_keys,
        temp_path=temp_path,
    )
    study = optimize_by_continuous_asha(
        objective=objective,
        dataset=dataset,
        plan=plan,
        checkpoint_path=temp_path,
        monitor=monitor,
    )
    study.result(top_n).yield_dataframe_as("result")

    return _run(
        dag=dag,
        execution_engine=execution_engine,
        execution_engine_conf=execution_engine_conf,
    )


def _run(
    dag: FugueWorkflow, execution_engine: Any, execution_engine_conf: Any
) -> List[TrialReport]:
    try:
        rows = list(
            dag.run(
                execution_engine,
                conf=execution_engine_conf,
            )["result"].as_dict_iterable()
        )
        return [
            TrialReport.from_jsondict(json.loads(r[TUNE_REPORT]))
            for r in sorted(rows, key=lambda r: r[TUNE_REPORT_METRIC])
        ]
    except FugueDataFrameError as e:
        raise e.__cause__ or e.__context__ or e
