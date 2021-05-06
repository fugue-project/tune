from datetime import datetime
from time import sleep

from pytest import raises
from tune.concepts.flow.report import TrialReport
from tune.concepts.flow.trial import Trial
from tune.noniterative.stopper import (
    SimpleNonIterativeStopper,
    n_samples,
    n_updates,
    no_update_period,
    small_improvement,
)


def test_simple_stopper():
    r1 = mr([], 0.2)
    r2 = mr(["x"], 0.4)
    r3 = mr([], 0.1)
    r4 = mr(["x"], 0.5)

    s = MockSimpleStopper(lambda latest, updated, reports: len(reports) >= 1)
    assert 0 == len(s.get_reports(r1.trial))
    assert s.can_accept(r1.trial)
    s.judge(r1)
    assert s._last is r1

    assert s.can_accept(r2.trial)
    s.judge(r2)
    assert s._last is r2

    assert not s.can_accept(r3.trial)
    assert not s.can_accept(r4.trial)


def test_logical_ops():
    r1 = mr([], 0.5)
    r2 = mr([], 0.4)
    r3 = mr([], 0.3)
    r4 = mr([], 0.2)
    take_two = MockSimpleStopper(lambda latest, updated, reports: len(reports) >= 2)
    ends_small = MockSimpleStopper(
        lambda latest, updated, reports: reports[-1].sort_metric <= 0.3
    )

    s = take_two & ends_small
    assert s.can_accept(r1.trial)
    s.judge(r1)
    assert s.can_accept(r2.trial)
    s.judge(r2)
    assert s.can_accept(r3.trial)
    s.judge(r3)
    assert not s.can_accept(r4.trial)

    with raises(AssertionError):
        take_two | ends_small  # can't reuse updated stoppers

    take_two = MockSimpleStopper(lambda latest, updated, reports: len(reports) >= 2)
    ends_small = MockSimpleStopper(
        lambda latest, updated, reports: reports[-1].sort_metric <= 0.3
    )

    s = take_two | ends_small
    assert s.can_accept(r1.trial)
    s.judge(r1)
    assert s.can_accept(r2.trial)
    s.judge(r2)
    assert not s.can_accept(r3.trial)


def test_n_samples():
    r1 = mr([], 0.1)
    r2 = mr([], 0.4)
    r3 = mr([], 0.3)

    s = n_samples(2)
    assert s.can_accept(r1.trial)
    s.judge(r1)
    assert s.can_accept(r2.trial)
    s.judge(r2)
    assert not s.can_accept(r3.trial)


def test_n_updates():
    r1 = mr([], 0.4)
    r2 = mr([], 0.5)
    r3 = mr([], 0.3)
    r4 = mr([], 0.2)

    s = n_updates(2)
    assert s.can_accept(r1.trial)
    s.judge(r1)
    assert s.can_accept(r2.trial)
    s.judge(r2)
    assert s.can_accept(r3.trial)
    s.judge(r3)  # updates satisfied
    assert not s.can_accept(r4.trial)


def test_no_update_period():
    r1 = mr([], 0.1)
    r2 = mr([], 0.4)
    r3 = mr([], 0.3)
    r4 = mr([], 0.3)

    s = no_update_period("0.2sec")
    assert s.can_accept(r1.trial)
    s.judge(r1)
    sleep(0.5)
    assert s.can_accept(r2.trial)
    s.judge(r2)  # not an update, and exceeded timeout, so trigger the shut down
    assert not s.can_accept(r3.trial)

    s = no_update_period("0.2sec")
    assert s.can_accept(r2.trial)
    s.judge(r2)
    sleep(0.5)
    assert s.can_accept(r3.trial)
    s.judge(r3)  # an update will not trigger shut down
    assert s.can_accept(r1.trial)
    s.judge(r1)
    assert s.can_accept(r4.trial)
    s.judge(r4)


def test_small_improvement():
    r1 = mr([], 0.5)
    r2 = mr([], 0.4)
    r22 = mr([], 0.51)
    r3 = mr([], 0.39)
    r4 = mr([], 0.2)

    s = small_improvement(0.09, 1)
    assert s.can_accept(r1.trial)
    s.judge(r1)
    assert s.can_accept(r2.trial)
    s.judge(r2)
    assert s.can_accept(r3.trial)
    s.judge(r3)
    assert not s.can_accept(r4.trial)

    s = small_improvement(0.25, 2)
    assert s.can_accept(r1.trial)
    s.judge(r1)
    assert s.can_accept(r2.trial)
    s.judge(r2)
    assert s.can_accept(r22.trial)
    s.judge(r22)  # if not an update, it doesn't count
    assert s.can_accept(r3.trial)
    s.judge(r3)
    assert not s.can_accept(r4.trial)


class MockSimpleStopper(SimpleNonIterativeStopper):
    def __init__(self, func):
        super().__init__(
            partition_should_stop=self.partition_should_stop, log_best_only=False
        )
        self._last = None
        self._func = func

    def partition_should_stop(self, latest_report, updated, reports) -> bool:
        self._last = latest_report
        return self._func(latest_report, updated, reports)


def mr(keys, metric):
    t = Trial(str(metric), dict(a=1), keys=keys)
    return TrialReport(t, metric, log_time=datetime.now())
