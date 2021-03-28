import pandas as pd
from fugue import FugueWorkflow
from tune.constants import TUNE_REPORT, TUNE_REPORT_METRIC
from tune.dataset import TuneDatasetBuilder
from tune.noniterative.convert import to_noniterative_objective
from tune.noniterative.objective import NonIterativeObjectiveRunner
from tune.noniterative.study import NonIterativeStudy
from tune.space import Grid, Space
from typing import List


def objective(a: float, b: pd.DataFrame) -> float:
    return a ** 2 + b.shape[0]


def objective2(a: float, b: pd.DataFrame) -> float:
    return -(a ** 2 + b.shape[0])


def assert_metric(df: pd.DataFrame, metrics: List[float]) -> None:
    assert len(metrics) == df.shape[0]
    assert set(metrics) == set(df[TUNE_REPORT_METRIC].tolist())


def test_study(tmpdir):
    study = NonIterativeStudy(
        to_noniterative_objective(objective), NonIterativeObjectiveRunner()
    )
    study2 = NonIterativeStudy(
        to_noniterative_objective(objective2, min_better=False),
        NonIterativeObjectiveRunner(),
    )
    space = Space(a=Grid(-2, 0, 1))
    input_df = pd.DataFrame([[0, 1], [1, 1], [0, 2]], columns=["a", "b"])
    dag = FugueWorkflow()

    # no data partition
    builder = TuneDatasetBuilder(space, str(tmpdir)).add_df("b", dag.df(input_df))
    dataset = builder.build(dag, 1)
    for distributed in [True, False, None]:
        # min_better = True
        result = study.optimize(dataset, distributed=distributed)
        result.result()[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(
            assert_metric, params=dict(metrics=[3.0, 4.0, 7.0])
        )
        result.result(2)[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(
            assert_metric, params=dict(metrics=[3.0, 4.0])
        )

        # min_better = False
        result = study2.optimize(dataset, distributed=distributed)
        result.result()[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(
            assert_metric, params=dict(metrics=[-3.0, -4.0, -7.0])
        )
        result.result(2)[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(
            assert_metric, params=dict(metrics=[-3.0, -4.0])
        )

    # with data partition
    builder = TuneDatasetBuilder(space, str(tmpdir)).add_df(
        "b", dag.df(input_df).partition_by("a")
    )
    dataset = builder.build(dag, 1)
    for distributed in [True, False, None]:
        result = study.optimize(dataset, distributed=distributed)
        result.result()[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(
            assert_metric, params=dict(metrics=[2.0, 3.0, 6.0, 1.0, 2.0, 5.0])
        )
        result.result(1)[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(
            assert_metric, params=dict(metrics=[1.0, 2.0])
        )

    dag.run()
