import json
from typing import Any, List, Optional

from fugue import FugueWorkflow
from tune import (
    TUNE_OBJECT_FACTORY,
    NonIterativeObjectiveRunner,
    Space,
    TrialReport,
    optimize_noniterative,
)
from tune.constants import TUNE_REPORT, TUNE_REPORT_METRIC

from tune_sklearn.objective import SKCVObjective, SKObjective


def suggest_sk_models(
    space: Space,
    train_df: Any,
    test_df: Any,
    scoring: str,
    temp_path: str = "",
    feature_prefix: str = "",
    label_col: str = "label",
    save_model: bool = False,
    partition_keys: Optional[List[str]] = None,
    top_n: int = 1,
    objective_runner: Optional[NonIterativeObjectiveRunner] = None,
    monitor: Any = None,
    distributed: Optional[bool] = None,
    execution_engine: Any = None,
    execution_engine_conf: Any = None,
) -> List[TrialReport]:
    dag = FugueWorkflow()
    dataset = TUNE_OBJECT_FACTORY.make_dataset(
        dag,
        space,
        df=train_df,
        test_df=test_df,
        partition_keys=partition_keys,
        temp_path=temp_path,
    )
    objective = SKObjective(
        scoring=scoring,
        feature_prefix=feature_prefix,
        label_col=label_col,
        checkpoint_path=temp_path if save_model else None,
    )
    study = optimize_noniterative(
        objective=objective,
        dataset=dataset,
        runner=objective_runner,
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


def suggest_sk_models_by_cv(
    space: Space,
    train_df: Any,
    scoring: str,
    cv: int = 5,
    temp_path: str = "",
    feature_prefix: str = "",
    label_col: str = "label",
    save_model: bool = False,
    partition_keys: Optional[List[str]] = None,
    top_n: int = 1,
    objective_runner: Optional[NonIterativeObjectiveRunner] = None,
    monitor: Any = None,
    distributed: Optional[bool] = None,
    execution_engine: Any = None,
    execution_engine_conf: Any = None,
) -> List[TrialReport]:
    dag = FugueWorkflow()
    dataset = TUNE_OBJECT_FACTORY.make_dataset(
        dag,
        space,
        df=train_df,
        partition_keys=partition_keys,
        temp_path=temp_path,
    )
    objective = SKCVObjective(
        scoring=scoring,
        cv=cv,
        feature_prefix=feature_prefix,
        label_col=label_col,
        checkpoint_path=temp_path if save_model else None,
    )
    study = optimize_noniterative(
        objective=objective,
        dataset=dataset,
        runner=objective_runner,
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
