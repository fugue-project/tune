from typing import Any, Dict, Iterable

from fugue import FugueWorkflow
from tune.dataset import TuneDatasetBuilder
from tune.iterative.objective import IterativeObjectiveFunc
from tune import optimize_by_sha, optimize_by_hyperband
from tune.space import Grid, Space
from tune.trial import TrialReport
from tune.constants import TUNE_REPORT_METRIC


class F(IterativeObjectiveFunc):
    def __init__(self):
        super().__init__()
        self._it = 0
        self._all = [
            [9, 3, 1, 1],
            [8, 6, 5, 5],
            [8, 5, 4, 3],
            [7, 4, 3, 4],
        ]

    def save_checkpoint(self, fs):
        fs.writetext("x", str(self._it))

    def load_checkpoint(self, fs):
        self._it = int(fs.readtext("x"))

    def run_single_iteration(self):
        trial = self.current_trial
        metric = self._all[trial.params["a"]][self._it]
        self._it += 1
        return TrialReport(trial, metric=metric)

    def copy(self):
        return F()


def test_sha(tmpdir):
    def assert_metric(df: Iterable[Dict[str, Any]], metric: float, ct: int) -> None:
        n = 0
        for row in df:
            assert row[TUNE_REPORT_METRIC] == metric
            n += 1
        assert n == ct

    space = Space(a=Grid(0, 1, 2, 3))
    dag = FugueWorkflow()
    dataset = TuneDatasetBuilder(space, str(tmpdir)).build(dag)
    obj = F()
    res = optimize_by_sha(
        obj,
        dataset,
        plan=[[1.0, 3], [1.0, 2], [1.0, 1], [1.0, 1]],
        checkpoint_path=str(tmpdir),
    )
    res.result().output(assert_metric, dict(metric=4.0, ct=1))

    res = optimize_by_sha(
        obj,
        dataset,
        plan=[[2.0, 2], [1.0, 1], [1.0, 1]],
        checkpoint_path=str(tmpdir),
    )
    res.result().output(assert_metric, dict(metric=1.0, ct=1))
    dag.run()


def test_hyperband(tmpdir):
    def assert_metric(df: Iterable[Dict[str, Any]], metric: float, ct: int) -> None:
        n = 0
        for row in df:
            if metric > 0:
                assert row[TUNE_REPORT_METRIC] == metric
            n += 1
        assert n == ct

    space = Space(a=Grid(0, 1, 2, 3))
    dag = FugueWorkflow()
    dataset = TuneDatasetBuilder(space, str(tmpdir)).build(dag)
    obj = F()
    res = optimize_by_hyperband(
        obj,
        dataset,
        plans=[
            [[1.0, 3], [1.0, 2], [1.0, 1], [1.0, 1]],
            [[2.0, 2], [1.0, 1], [1.0, 1]],
        ],
        checkpoint_path=str(tmpdir),
    )
    res.result().output(assert_metric, dict(metric=0.0, ct=2))
    res.result(1).output(assert_metric, dict(metric=1.0, ct=1))
    dag.run()
