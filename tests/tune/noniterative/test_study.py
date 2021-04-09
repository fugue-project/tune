from typing import List

import pandas as pd
from fugue import FugueWorkflow
from tune.constants import TUNE_REPORT, TUNE_REPORT_METRIC
from tune.dataset import TuneDatasetBuilder
from tune.noniterative.convert import to_noniterative_objective
from tune import optimize_noniterative
from tune.space import Grid, Space
from tune.trial import Monitor


def objective(a: float, b: pd.DataFrame) -> float:
    return a ** 2 + b.shape[0]


def objective2(a: float, b: pd.DataFrame) -> float:
    return -(a ** 2 + b.shape[0])


def assert_metric(df: pd.DataFrame, metrics: List[float]) -> None:
    assert len(metrics) == df.shape[0]
    assert set(metrics) == set(df[TUNE_REPORT_METRIC].tolist())


class M(Monitor):
    def __init__(self):
        super().__init__()
        self._reports = []

    def on_report(self, report) -> None:
        self._reports.append(report)


def test_study(tmpdir):
    space = Space(a=Grid(-2, 0, 1))
    input_df = pd.DataFrame([[0, 1], [1, 1], [0, 2]], columns=["a", "b"])
    dag = FugueWorkflow()
    monitor = M()

    # no data partition
    builder = TuneDatasetBuilder(space, str(tmpdir)).add_df("b", dag.df(input_df))
    dataset = builder.build(dag, 1)
    for distributed in [True, False, None]:
        # min_better = True
        result = optimize_noniterative(
            objective=to_noniterative_objective(objective),
            dataset=dataset,
            distributed=distributed,
        )
        result.result()[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(
            assert_metric, params=dict(metrics=[3.0, 4.0, 7.0])
        )
        result.result(2)[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(
            assert_metric, params=dict(metrics=[3.0, 4.0])
        )

        # min_better = False
        result = optimize_noniterative(
            objective=to_noniterative_objective(objective, min_better=False),
            dataset=dataset,
            distributed=distributed,
        )
        result.result()[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(
            assert_metric, params=dict(metrics=[-7.0, -4.0, -3.0])
        )
        result.result(2)[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(
            assert_metric, params=dict(metrics=[-7.0, -4.0])
        )

    # with data partition
    builder = TuneDatasetBuilder(space, str(tmpdir)).add_df(
        "b", dag.df(input_df).partition_by("a")
    )
    dataset = builder.build(dag, 1)
    for distributed in [True, False, None]:
        result = optimize_noniterative(
            objective=to_noniterative_objective(objective),
            dataset=dataset,
            distributed=distributed,
            monitor=monitor,
        )
        result.result()[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(
            assert_metric, params=dict(metrics=[2.0, 3.0, 6.0, 1.0, 2.0, 5.0])
        )
        result.result(1)[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(
            assert_metric, params=dict(metrics=[1.0, 2.0])
        )

    dag.run()

    assert 3 * 3 * 2 == len(monitor._reports)
