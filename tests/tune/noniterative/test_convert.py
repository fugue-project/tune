from typing import Any, Dict, Tuple

from pytest import raises
from tune.exceptions import TuneCompileError
from tune.trial import Trial, TrialReport
from tune.noniterative.convert import noniterative_objective, to_noniterative_objective
from tune.noniterative.objective import NonIterativeObjectiveFunc


def test_to_noniterative_objective():
    def f1(a, b: int) -> float:
        return a - b

    func = to_noniterative_objective(f1)
    assert func.min_better
    trial = Trial("abc", dict(b=20, a=10), dict(c=3))
    report = func.run(trial)
    assert report.trial is trial
    assert report.metric == -10
    assert report.params == trial.params
    assert report.metadata == {}

    func = to_noniterative_objective("f1", min_better=False)
    assert not func.min_better
    trial = Trial("abc", dict(b=20, a=10), dict(c=3))
    report = func.run(trial)
    assert report.trial is trial
    assert report.metric == -10
    assert report.params == trial.params
    assert report.metadata == {}

    assert -1 == func(1, 2)

    def f2(a, b: int) -> Tuple[float, Dict[str, Any]]:
        return a - b, dict(c=5)

    func = to_noniterative_objective(f2)
    trial = Trial("abc", dict(b=20, a=10), dict(c=3))
    report = func.run(trial)
    assert report.trial is trial
    assert report.metric == -10
    assert report.params == trial.params
    assert report.metadata == dict(c=5)

    def f3(t: Trial) -> TrialReport:
        return TrialReport(
            t, t.params["a"] - t.params["b"], params=dict(a=1), metadata=dict(c=6)
        )

    func = to_noniterative_objective(f3)
    trial = Trial("abc", dict(b=20, a=10), dict(c=3))
    report = func.run(trial)
    assert report.trial is trial
    assert report.metric == -10
    assert report.params == dict(a=1)
    assert report.metadata == dict(c=6)

    class F4(NonIterativeObjectiveFunc):
        def run(self, t: Trial) -> TrialReport:
            return TrialReport(
                t, t.params["a"] - t.params["b"], params=dict(a=1), metadata=dict(c=6)
            )

    f4 = F4()
    f4_ = to_noniterative_objective(f4)
    assert isinstance(f4_, F4)
    assert f4 is not f4_

    raises(TuneCompileError, lambda: to_noniterative_objective("abc"))


def test_noniterative_objective():
    @noniterative_objective
    def f1(a, b: int) -> float:
        return a - b

    assert isinstance(f1, NonIterativeObjectiveFunc)
    trial = Trial("abc", dict(b=20, a=10), dict(c=3))
    report = f1.run(trial)
    assert report.trial is trial
    assert report.metric == -10
    assert report.params == trial.params
    assert report.metadata == {}

    func = to_noniterative_objective("f1")
    report = func.run(trial)
    assert report.trial is trial
    assert report.metric == -10
    assert report.params == trial.params
    assert report.metadata == {}

    @noniterative_objective(min_better=False)
    def f2(a, b: int) -> float:
        return a - b

    assert isinstance(f2, NonIterativeObjectiveFunc)
    assert not f2.min_better
