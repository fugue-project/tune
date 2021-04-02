from pytest import raises
from triad import FileSystem
from tune.iterative.trial import (
    IterativeTrial,
    TrialDecision,
    TrialJudge,
    TrialJudgeMonitor,
    TrialReport,
)
from tune.trial import Trial


def test_iterative_trial(tmpdir):
    class J(TrialJudge):
        def judge(self, report: TrialReport) -> TrialDecision:
            self.report = report
            return TrialDecision(report, False, True, metadata={})

    class M(TrialJudgeMonitor):
        def __init__(self):
            self._gb = 0
            self._j = 0
            self._r = 0

        def on_get_budget(self, trial, rung, budget):
            self._gb += 1

        def on_judge(self, decision):
            self._j += 1

        def on_report(self, report):
            self._r += 1

    tr = Trial("abc", {"a": 1}, {"b": 2}, keys=["a", "b"])
    m = M()
    t1 = IterativeTrial(tr, TrialJudge(monitor=m))
    assert "abc" == t1.trial_id
    assert {"a": 1} == t1.params
    assert {"b": 2} == t1.metadata
    assert not t1.has_checkpoint
    raises(AssertionError, lambda: t1.checkpoint)
    assert 0 == t1.iteration
    report = TrialReport(t1, 1.1, {"c": 3}, metadata={"d": 4})
    decision = t1.judge.judge(report)
    t1.judge.monitor.on_report(report)
    t1.judge.monitor.on_judge(decision)
    t1.judge.monitor.on_get_budget(report.trial, report.rung, 0.0)
    assert 1 == m._gb and 1 == m._r and 1 == m._j
    assert 1.1 == decision.report.metric
    assert {"c": 3} == decision.report.params
    assert {"d": 4} == decision.report.metadata
    assert decision.should_stop
    assert not decision.should_checkpoint

    judge = J()
    fs = FileSystem().opendir(str(tmpdir))
    t2 = t1.with_checkpoint_basedir_fs(fs).with_judge(judge)
    assert "abc" == t2.trial_id
    assert {"a": 1} == t2.params
    assert {"b": 2} == t2.metadata
    assert t2.has_checkpoint
    report = TrialReport(t2, 1.2, {"c": 30}, metadata={"d": 40})
    decision = t2.judge.judge(report)
    with t2.checkpoint.create() as ffs:
        ffs.touch("x")
    assert 1.2 == decision.report.metric
    assert {"c": 30} == decision.report.params
    assert {"d": 40} == decision.report.metadata
    assert 1 == t2.iteration
