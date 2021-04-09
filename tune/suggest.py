import json
from tune.exceptions import TuneCompileError
from typing import Any, List, Optional, Tuple

from fugue import FugueWorkflow
from triad.utils.assertion import assert_or_throw
from tune.factory import TUNE_OBJECT_FACTORY
from tune.space import Space
from tune.trial import TrialReport
from tune.api import (
    optimize_by_continuous_asha,
    optimize_by_sha,
    optimize_by_hyperband,
)
from tune.constants import TUNE_REPORT, TUNE_REPORT_METRIC


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
