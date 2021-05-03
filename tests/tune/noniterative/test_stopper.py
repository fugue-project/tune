from tune.concepts.flow.report import TrialReport
from tune.concepts.flow.trial import Trial
from tune.noniterative.stopper import SimpleNonIterativeStopper


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


class MockSimpleStopper(SimpleNonIterativeStopper):
    def __init__(self, func):
        super().__init__(log_best_only=False)
        self._func = func
        self._last = None

    def partition_should_stop(self, latest_report, updated, reports) -> bool:
        self._last = latest_report
        return self._func(latest_report, updated, reports)


def mr(keys, metric):
    t = Trial(str(metric), dict(a=1), keys=keys)
    return TrialReport(t, metric)
