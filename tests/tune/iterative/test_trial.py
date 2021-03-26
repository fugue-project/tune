from tune.iterative.trial import IterativeTrial, TrialJudge, TrialReport, TrialDecision
from triad import FileSystem
from pytest import raises


def test_iterative_trial(tmpdir):
    class J(TrialJudge):
        def judge(self, report: TrialReport) -> TrialDecision:
            self.report = report
            return TrialDecision(report, False, True, metadata={})

    t1 = IterativeTrial("abc", {"a": 1}, {"b": 2}, TrialJudge())
    assert "abc" == t1.trial_id
    assert {"a": 1} == t1.params
    assert {"b": 2} == t1.metadata
    assert not t1.has_checkpoint
    raises(AssertionError, lambda: t1.checkpoint)
    assert 0 == t1.iteration
    decision = t1.report(1.1, {"c": 3}, metadata={"d": 4})
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
    decision = t2.report(
        1.2, {"c": 30}, metadata={"d": 40}, save_checkpoint=lambda f: f.touch("x")
    )
    assert 1.2 == decision.report.metric
    assert {"c": 30} == decision.report.params
    assert {"d": 40} == decision.report.metadata
    assert t2.checkpoint.latest.exists("x")
    assert 1 == t2.iteration
