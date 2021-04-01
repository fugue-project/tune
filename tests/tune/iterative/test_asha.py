from tune.iterative.asha import RungHeap, ASHAJudge
from tune.trial import TrialReport, Trial
import math


def test_run_heap():
    h = RungHeap(2)
    assert 2 == h.capacity
    assert not h.full
    assert math.isnan(h.best)
    assert 0 == len(h)
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
        asha_should_stop_func=should_stop,
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
    d = j.judge(rp("a", 0.5, 1))
    assert 3.0 == d.budget
    assert d.should_checkpoint
    d = j.judge(rp("b", 0.45, 1))
    assert 0.0 == d.budget  # rungs[1] and rungs[0] diff so somall
    assert d.should_checkpoint
    d = j.judge(rp("a", 0.44, 2))
    assert 0.0 == d.budget  # already stopped
    assert d.should_checkpoint


def test_trial_stop():
    def should_stop(history, rungs):
        if len(history) == 0:
            return False
        tid = history[-1].trial_id
        return not all(tid in x for x in rungs)

    j = ASHAJudge(
        schedule=[(1.0, 2), (2.0, 2), (3.0, 1)],
        always_checkpoint=True,
        trial_should_stop_func=should_stop,
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


def rp(tid, metric, rung=0, keys=[]):
    t = Trial(tid, {}, keys=keys)
    return TrialReport(t, metric=metric, rung=rung)
