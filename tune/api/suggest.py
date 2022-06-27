from typing import Any, List, Optional, Tuple

from fugue import FugueWorkflow
from fugue.exceptions import FugueDataFrameError
from triad import assert_or_throw
from tune._utils import from_base64
from tune.api.factory import TUNE_OBJECT_FACTORY
from tune.api.optimize import (
    optimize_by_continuous_asha,
    optimize_by_hyperband,
    optimize_by_sha,
    optimize_noniterative,
)
from tune.concepts.flow import TrialReport
from tune.concepts.logger import make_logger, parse_logger
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
    logger: Any = None,
    monitor: Any = None,
    stopper: Any = None,
    stop_check_interval: Any = None,
    distributed: Optional[bool] = None,
    shuffle_candidates: bool = True,
    execution_engine: Any = None,
    execution_engine_conf: Any = None,
) -> List[TrialReport]:
    """Given non-iterative ``objective``, ``space`` and (optional) dataframe,
    suggest the best parameter combinations.

    .. important::
        Please read
        :ref:`Non-Iterative Tuning Guide </notebooks/noniterative.ipynb>`

    :param objective: |NonIterativeObjective|
    :param space: search space, please read |SpaceTutorial|
    :param df: |DataFrameLike|, defaults to None
    :param df_name: dataframe name, defaults to the value of
      ``TUNE_DATASET_DF_DEFAULT_NAME``
    :param temp_path: |TempPath|, defaults to ""
    :param partition_keys: partition keys for ``df``, defaults to None.
      For details, please read |DatasetTutorial|
    :param top_n: number of best results to return, defaults to 1.
      If `<=0` all results will be returned
    :param local_optimizer: |NonIterativeOptimizer|, defaults to None
    :param logger: |LoggerLikeObject|, defaults to None
    :param monitor: realtime monitor, defaults to None. Read
      :ref:`Monitoring Guide </notebooks/noniterative.ipynb#Realtime-Monitoring>`
    :param stopper: early stopper, defaults to None. Read
      :ref:`Early Stopping Guide </notebooks/noniterative.ipynb#Early-Stopping>`
    :param stop_check_interval: an object that can be converted to timedelta,
      defaults to None.
      For details, read :func:`~triad.utils.convert.to_timedelta`
    :param distributed: whether to use the exeuction engine to run different trials
      distributedly, defaults to None. If None, it's equal to True.
    :param shuffle_candidates: whether to shuffle the candidate configurations,
      defaults to True. This is no effect on final result.
    :param execution_engine: Fugue |ExecutionEngine| like object, defaults to None.
      If None, :class:`~fugue.execution.native_execution_engine.NativeExecutionEngine`
      will be used, the task will be running on local machine.
    :param execution_engine_conf: |ParamsLikeObject|, defaults to None
    :return: a list of best results
    """
    logger = parse_logger(logger)
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
        logger=logger,
        stopper=stopper,
        stop_check_interval=stop_check_interval,
    )
    study.result(top_n).yield_dataframe_as("result")

    return _run(
        dag=dag,
        execution_engine=execution_engine,
        execution_engine_conf=execution_engine_conf,
        logger=logger,
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
        not space.has_stochastic,
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
        not space.has_stochastic,
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
        not space.has_stochastic,
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
    dag: FugueWorkflow,
    execution_engine: Any,
    execution_engine_conf: Any,
    logger: Any = None,
) -> List[TrialReport]:
    def _safe(xx):
        return float("inf") if xx is None else float(xx)

    try:
        rows = list(
            dag.run(
                execution_engine,
                conf=execution_engine_conf,
            )["result"].as_dict_iterable()
        )
        result = [
            from_base64(r[TUNE_REPORT])
            for r in sorted(rows, key=lambda r: _safe(r[TUNE_REPORT_METRIC]))
        ]
        if len(result) > 0 and logger is not None:
            try:
                with make_logger(logger) as p_logger:
                    p_logger.log_report(
                        result[0],
                        log_params=True,
                        extract_metrics=True,
                        log_metadata=True,
                    )
            except Exception:
                pass
        return result
    except FugueDataFrameError as e:  # pragma: no cover
        raise e.__cause__ or e.__context__ or e
