import numpy as np

from tune.space import Rand
from tune.trial import Trial, TrialDecision, TrialReport
import pandas as pd
import copy


def test_trial():
    trial = Trial("abc", {"a": 1}, {"b": 2})
    assert "abc" == trial.trial_id
    assert {"a": 1} == trial.params
    assert {"b": 2} == trial.metadata

    t2 = trial.with_params({"c": 3})
    assert "abc" == t2.trial_id
    assert {"c": 3} == t2.params
    assert {"b": 2} == t2.metadata


def test_copy():
    trial = Trial("abc", {"a": 1})
    t1 = trial.with_dfs({"c": pd.DataFrame([[0]])})

    t2 = copy.copy(t1)
    t3 = copy.deepcopy(t1)
    assert trial.trial_id == t2.trial_id == t3.trial_id
    assert t1.dfs is t2.dfs is t3.dfs
    assert 0 == len(trial.dfs)
    assert 1 == len(t1.dfs)


def test_encode_decode():
    p = {"a": 1, "b": Rand(1, 2)}
    trial = Trial("abc", p, {})
    d = trial.jsondict
    assert isinstance(d["params"]["b"], dict)
    t = Trial.from_jsondict(d)
    assert isinstance(t.params["b"], Rand)


def test_trial_report():
    trial = Trial("abc", {"a": Rand(3, 4)}, {"b": 2})
    report = copy.copy(
        TrialReport(
            trial,
            metric=np.float(0.1),
            params={"c": Rand(1, 2)},
            metadata={"d": 4},
            cost=2.0,
        )
    )
    assert trial is report.trial
    assert 0.1 == report.metric
    assert type(report.metric) == float
    assert {"c": Rand(1, 2)} == report.params
    assert {"d": 4} == report.metadata
    assert 2.0 == report.cost

    report = copy.deepcopy(TrialReport(trial, metric=np.float(0.1), cost=2.0))
    assert trial is report.trial
    assert 0.1 == report.metric
    assert type(report.metric) == float
    assert {"a": Rand(3, 4)} == report.params
    assert {} == report.metadata
    assert 2.0 == report.cost

    report = TrialReport.from_jsondict(report.jsondict)
    assert trial.trial_id == report.trial_id
    assert 0.1 == report.metric
    assert type(report.metric) == float
    assert {"a": Rand(3, 4)} == report.params
    assert {} == report.metadata
    assert 2.0 == report.cost


def test_trial_decision():
    trial = Trial("abc", {"a": 1}, {"b": Rand(0, 2)})
    report = TrialReport(
        trial, metric=np.float(0.1), params={"c": Rand(0, 3)}, metadata={"d": 4}
    )
    decision = TrialDecision(
        report, should_stop=True, should_checkpoint=True, metadata={"x": 1}
    )
    assert trial is decision.trial
    assert report is decision.report
    assert decision.should_stop
    assert decision.should_checkpoint
    assert {"x": 1} == decision.metadata

    assert copy.copy(decision) is decision
    assert copy.deepcopy(decision) is decision

    d2 = TrialDecision.from_jsondict(decision.jsondict)
    assert d2.trial_id == trial.trial_id
    assert Rand(0, 3) == d2.report.params["c"]
    assert decision.should_stop
    assert decision.should_checkpoint
    assert {"x": 1} == decision.metadata
