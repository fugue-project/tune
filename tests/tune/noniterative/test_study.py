from typing import List

import pandas as pd
from fugue import FugueWorkflow
from pytest import raises
from tune import optimize_noniterative, suggest_for_noniterative_objective
from tune.concepts.dataset import TuneDatasetBuilder
from tune.concepts.flow import Monitor
from tune.concepts.space import Grid, Space
from tune.constants import TUNE_REPORT, TUNE_REPORT_METRIC
from tune.exceptions import TuneInterrupted
from tune.noniterative.convert import to_noniterative_objective
from tune.noniterative.stopper import n_samples


def objective(a: float, b: pd.DataFrame) -> float:
    return a ** 2 + b.shape[0]


def objective2(a: float, b: pd.DataFrame) -> float:
    return -(a ** 2 + b.shape[0])


def objective3(a: float, b: pd.DataFrame) -> float:
    if a == -2:
        raise TuneInterrupted()
    return a ** 2 + b.shape[0]


def objective4(a: float, b: pd.DataFrame) -> float:
    if a == -2:
        raise ValueError("expected")
    return a ** 2 + b.shape[0]


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
            objective=objective,
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


def test_study_with_stopper(tmpdir):
    space = Space(a=Grid(-2, 0, 1))
    input_df = pd.DataFrame([[0, 1], [1, 1], [0, 2]], columns=["a", "b"])

    result = suggest_for_noniterative_objective(
        objective=objective,
        space=space,
        df=input_df,
        df_name="b",
        stopper=n_samples(2),
        top_n=0,
        shuffle_candidates=False,
        temp_path=str(tmpdir),
    )
    assert [3.0, 7.0] == [x.metric for x in result]

    monitor = M()
    result = suggest_for_noniterative_objective(
        objective=objective,
        space=space,
        df=input_df,
        df_name="b",
        stopper=n_samples(2),
        monitor=monitor,
        top_n=0,
        shuffle_candidates=False,
        temp_path=str(tmpdir),
    )
    assert [3.0, 7.0] == [x.metric for x in result]

    assert 2 == len(monitor._reports)

    # Tune interrupted
    monitor = M()
    result = suggest_for_noniterative_objective(
        objective=objective3,
        space=space,
        df=input_df,
        df_name="b",
        stopper=n_samples(2),
        monitor=monitor,
        top_n=0,
        shuffle_candidates=False,
        temp_path=str(tmpdir),
    )
    assert [3.0, 4.0] == [x.metric for x in result]

    assert 2 == len(monitor._reports)


def test_study_with_exceptions(tmpdir):
    space = Space(a=Grid(-2, 0, 1))
    input_df = pd.DataFrame([[0, 1], [1, 1], [0, 2]], columns=["a", "b"])

    with raises(ValueError):
        suggest_for_noniterative_objective(
            objective=objective4,
            space=space,
            df=input_df,
            df_name="b",
            temp_path=str(tmpdir),
        )

    with raises(ValueError):
        monitor = M()
        suggest_for_noniterative_objective(
            objective=objective4,
            space=space,
            df=input_df,
            df_name="b",
            stopper=n_samples(2),
            monitor=monitor,
            temp_path=str(tmpdir),
        )
