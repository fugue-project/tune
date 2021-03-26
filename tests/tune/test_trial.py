from tune.trial import Trial, TrialDecision, TrialReport
import numpy as np


def test_trial():
    trial = Trial("abc", {"a": 1}, {"b": 2})
    assert "abc" == trial.trial_id
    assert {"a": 1} == trial.params
    assert {"b": 2} == trial.metadata

    t2 = trial.with_params({"c": 3})
    assert "abc" == t2.trial_id
    assert {"c": 3} == t2.params
    assert {"b": 2} == t2.metadata


def test_trial_report():
    trial = Trial("abc", {"a": 1}, {"b": 2})
    report = TrialReport(
        trial, metric=np.float(0.1), params={"c": 3}, metadata={"d": 4}
    )
    assert trial is report.trial
    assert 0.1 == report.metric
    assert type(report.metric) == float
    assert {"c": 3} == report.params
    assert {"d": 4} == report.metadata


def test_trial_decision():
    trial = Trial("abc", {"a": 1}, {"b": 2})
    report = TrialReport(
        trial, metric=np.float(0.1), params={"c": 3}, metadata={"d": 4}
    )
    decision = TrialDecision(
        report, should_stop=True, should_checkpoint=True, metadata={"x": 1}
    )
    assert trial is decision.trial
    assert report is decision.report
    assert decision.should_stop
    assert decision.should_checkpoint
    assert {"x": 1} == decision.metadata
