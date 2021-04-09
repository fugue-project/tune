import math
from typing import Any, Dict, Iterable

from fugue import FugueWorkflow
from tune import optimize_by_continuous_asha
from tune.constants import TUNE_REPORT_METRIC
from tune.dataset import TuneDatasetBuilder
from tune.iterative.asha import ASHAJudge, RungHeap
from tune.iterative.objective import IterativeObjectiveFunc
from tune.space import Grid, Space
from tune.trial import Monitor, Trial, TrialReport


def test_rung_heap():
    h = RungHeap(2)
    assert 2 == h.capacity
    assert not h.full
    assert math.isnan(h.best)
    assert 0 == len(h)
    assert [] == h.bests
    assert h.push(rp("a", 1))
    assert h.push(rp("b", 0.1))
    assert h.push(rp("a", 2))
    assert h.push(rp("c", 0.5))
    assert not h.push(rp("d", 0.6))
    assert 0.1 == h.best
    assert 2 == len(h)
    assert h.full
    assert "a" not in h
    assert "d" not in h
    assert "b" in h
    assert "c" in h
    assert [1.0, 0.1, 0.1, 0.1, 0.1] == h.bests
    assert h.push(rp("e", 0.01))
    assert [1.0, 0.1, 0.1, 0.1, 0.1, 0.01] == h.bests
    assert "b" in h and "e" in h and 2 == len(h)
    assert h.push(rp("e", 5))
    assert [1.0, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01] == h.bests
    values = {x.trial_id: x for x in h.values()}
    assert "b" in values and "e" in values and 2 == len(values)
    assert 5.0 == values["e"].sort_metric


# TODO: test multiple partitions


def test_asha_judge_simple_happy_path():
    j = ASHAJudge(schedule=[(1.0, 2), (2.0, 1)])
    d = j.judge(rp("a", 0.5, 0))
    assert 2.0 == d.budget
    assert not d.should_checkpoint
    d = j.judge(rp("b", 0.6, 0))
    assert 2.0 == d.budget
    assert not d.should_checkpoint
    d = j.judge(rp("a", 0.4, 1))
    assert d.should_stop
    assert d.should_checkpoint
    # stop criteria met, so other jobs won't get more budget
    d = j.judge(rp("c", 0.2, 0))
    assert d.should_stop
    assert d.should_checkpoint


def test_asha_stop():
    def should_stop(keys, rungs):
        metrics = []
        for r in rungs:
            if len(r) == 0:
                break
            if not r.full:
                return False
            metrics.append(r.best)
        if len(metrics) < 2:
            return False
        return metrics[-2] - metrics[-1] < 0.2

    j = ASHAJudge(
        schedule=[(1.0, 2), (2.0, 2), (3.0, 1)],
        always_checkpoint=True,
        study_early_stop=should_stop,
    )
    d = j.judge(rp("a", 0.6, 0))
    assert 2.0 == d.budget
    assert d.should_checkpoint
    d = j.judge(rp("b", 0.5, 0))
    assert 2.0 == d.budget
    assert d.should_checkpoint
    d = j.judge(rp("c", 0.4, 0))
    assert 2.0 == d.budget
    assert d.should_checkpoint
    d = j.judge(rp("b", 0.45, 1))
    assert 3.0 == d.budget
    assert d.should_checkpoint
    d = j.judge(rp("c", 0.39, 1))
    assert 3.0 == d.budget
    assert d.should_checkpoint
    d = j.judge(rp("x", 0.45, 0))
    # rungs[1] and rungs[0] diff so somall
    # no longer accept new trials
    assert 0.0 == d.budget
    assert d.should_checkpoint
    d = j.judge(rp("b", 0.45, 1))
    assert 3.0 == d.budget  # existed ids can still be accepted
    assert d.should_checkpoint
    d = j.judge(rp("a", 0.44, 2))
    assert 0.0 == d.budget  # already stopped
    assert d.should_checkpoint


def test_trial_stop():
    def should_stop(report, history, rungs):
        return not all(report.trial_id in x for x in rungs[: report.rung])

    j = ASHAJudge(
        schedule=[(1.0, 2), (2.0, 2), (3.0, 1)],
        always_checkpoint=True,
        trial_early_stop=should_stop,
    )
    d = j.judge(rp("a", 0.6, 0))
    assert 2.0 == d.budget
    assert d.should_checkpoint
    d = j.judge(rp("b", 0.5, 0))
    assert 2.0 == d.budget
    assert d.should_checkpoint
    d = j.judge(rp("c", 0.4, 0))
    assert 2.0 == d.budget
    assert d.should_checkpoint
    d = j.judge(rp("a", 0.1, 1))
    assert d.should_stop  # kicked out by c
    assert d.should_checkpoint


def test_run_asha(tmpdir):
    class M(Monitor):
        def on_report(self, report: TrialReport) -> None:
            print(report.jsondict)

    def assert_metric(df: Iterable[Dict[str, Any]], metric: float, ct: int) -> None:
        n = 0
        for row in df:
            assert row[TUNE_REPORT_METRIC] == metric
            n += 1
        assert n == ct

    space = Space(a=Grid(0, 1, 2, 3))
    dag = FugueWorkflow()
    dataset = TuneDatasetBuilder(space, str(tmpdir)).build(dag, shuffle=False)
    obj = F()
    res = optimize_by_continuous_asha(
        obj,
        dataset,
        plan=[[1.0, 3], [1.0, 2], [1.0, 1], [1.0, 1]],
        checkpoint_path=str(tmpdir),
    )
    res.result(1).output(assert_metric, dict(metric=1.0, ct=1))

    res = optimize_by_continuous_asha(
        obj,
        dataset,
        plan=[[2.0, 2], [1.0, 1], [1.0, 1]],
        checkpoint_path=str(tmpdir),
        monitor=M(),
    )
    res.result(1).output(assert_metric, dict(metric=1.0, ct=1))
    dag.run()


def rp(tid, metric, rung=0, keys=[]):
    t = Trial(tid, {}, keys=keys)
    return TrialReport(t, metric=metric, rung=rung)


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
