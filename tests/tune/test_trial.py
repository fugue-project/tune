import copy

import numpy as np
import pandas as pd

from tune.space import Rand
from tune.trial import Trial, TrialDecision, TrialReport, TrialReportHeap


def test_trial():
    trial = Trial("abc", {"a": 1}, {"b": 2}, keys=["a", "b"])
    assert "abc" == trial.trial_id
    assert {"a": 1} == trial.params
    assert {"b": 2} == trial.metadata
    assert ["a", "b"] == trial.keys

    t2 = trial.with_params({"c": 3})
    assert "abc" == t2.trial_id
    assert {"c": 3} == t2.params
    assert {"b": 2} == t2.metadata
    assert ["a", "b"] == t2.keys


def test_copy():
    trial = Trial("abc", {"a": 1})
    t1 = trial.with_dfs({"c": pd.DataFrame([[0]])})

    t2 = copy.copy(t1)
    t3 = copy.deepcopy(t1)
    assert trial.trial_id == t2.trial_id == t3.trial_id
    assert t1.dfs is t2.dfs is t3.dfs
    assert 0 == len(trial.dfs)
    assert 1 == len(t1.dfs)
    assert [] == t3.keys


def test_encode_decode():
    p = {"a": 1, "b": Rand(1, 2)}
    trial = Trial("abc", p, {}, keys=["x", "y"], dfs={"v": ""})
    d = trial.jsondict
    assert isinstance(d["params"]["b"], dict)
    t = Trial.from_jsondict(d)
    assert isinstance(t.params["b"], Rand)
    assert ["x", "y"] == t.keys
    assert {} == t.dfs  # dfs will not be serialized


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
    assert 0 == report.rung
    assert 0.1 == report.sort_metric

    report = copy.deepcopy(
        TrialReport(trial, metric=np.float(0.111), cost=2.0, rung=4, sort_metric=1.23)
    )
    assert trial is report.trial
    assert 0.111 == report.metric
    assert type(report.metric) == float
    assert {"a": Rand(3, 4)} == report.params
    assert {} == report.metadata
    assert 2.0 == report.cost
    assert 4 == report.rung

    r1 = report.generate_sort_metric(True, 2)
    r2 = report.generate_sort_metric(False, 1)
    r3 = report.with_sort_metric(0.234)
    assert 1.23 == report.sort_metric
    assert 0.11 == r1.sort_metric
    assert -0.1 == r2.sort_metric
    assert 0.234 == r3.sort_metric

    report = TrialReport.from_jsondict(report.jsondict)
    assert trial.trial_id == report.trial_id
    assert 0.111 == report.metric
    assert type(report.metric) == float
    assert {"a": Rand(3, 4)} == report.params
    assert {} == report.metadata
    assert 2.0 == report.cost

    assert 3.0 == report.with_cost(3.0).cost
    assert 5 == report.with_rung(5).rung


def test_trial_report_heap():
    t1 = Trial("a", {})
    r1 = TrialReport(t1, 0.1)
    t2 = Trial("b", {})
    r2 = TrialReport(t2, 0.2)
    t3 = Trial("c", {})
    r3 = TrialReport(t3, 0.3)
    r4 = TrialReport(t3, -0.3)
    h = TrialReportHeap(min_heap=True)
    for r in [r1, r2, r3, r4]:
        h.push(r)
    assert "a" in h
    assert "x" not in h
    for r in [r4, r1, r2]:
        assert h.pop() is r
    assert 0 == len(h)

    h = TrialReportHeap(min_heap=False)
    for r in [r1, r2, r3, r4]:
        h.push(r)
    for r in [r1, r2, r4]:
        assert r in list(h.values())
    for r in [r2, r1, r4]:
        assert h.pop() is r
    assert 0 == len(h)

    # test __lt__, the sort key is sort_metric!
    r5 = TrialReport(t1, metric=0.1, sort_metric=-0.1)
    r6 = TrialReport(t2, metric=0.2, sort_metric=-0.2)
    r7 = TrialReport(t3, metric=0.3, sort_metric=-0.3)

    h = TrialReportHeap(min_heap=True)
    for r in [r7, r6, r5]:
        h.push(r)
    for r in [r7, r6, r5]:
        assert h.pop() is r
    assert 0 == len(h)

    r5 = TrialReport(t1, metric=0.1, cost=0.2, rung=5)
    r6 = TrialReport(t2, metric=0.1, cost=0.3, rung=5)
    r7 = TrialReport(t3, metric=0.1, cost=0.3, rung=6)

    h = TrialReportHeap(min_heap=True)
    for r in [r7, r6, r5]:
        h.push(r)
    for r in [r5, r6, r7]:
        assert h.pop() is r
    assert 0 == len(h)

    # equal case
    r8 = TrialReport(t1, metric=0.1, cost=0.3, rung=6)
    r9 = TrialReport(t2, metric=0.1, cost=0.3, rung=6)

    h = TrialReportHeap(min_heap=False)
    for r in [r8, r9]:
        h.push(r)
    for r in [r8, r9]:
        assert h.pop() is r
    assert 0 == len(h)


def test_trial_decision():
    trial = Trial("abc", {"a": 1}, {"b": Rand(0, 2)})
    report = TrialReport(
        trial, metric=np.float(0.1), params={"c": Rand(0, 3)}, metadata={"d": 4}
    )
    decision = TrialDecision(
        report, budget=0.0, should_checkpoint=True, metadata={"x": 1}, reason="p"
    )
    assert trial is decision.trial
    assert report is decision.report
    assert decision.should_stop
    assert decision.should_checkpoint
    assert {"x": 1} == decision.metadata
    assert "p" == decision.reason
    assert 0.0 == decision.budget

    assert copy.copy(decision) is decision
    assert copy.deepcopy(decision) is decision

    d2 = TrialDecision.from_jsondict(decision.jsondict)
    assert d2.trial_id == trial.trial_id
    assert Rand(0, 3) == d2.report.params["c"]
    assert decision.should_stop
    assert decision.should_checkpoint
    assert {"x": 1} == decision.metadata
    assert "p" == decision.reason

    decision = TrialDecision(
        report, budget=1.0, should_checkpoint=True, metadata={"x": 1}
    )
    assert 1.0 == decision.budget
    assert not decision.should_stop
