from tune import Trial, TrialReport

from tune_notebook import (
    NotebookSimpleChart,
    NotebookSimpleHist,
    NotebookSimpleRungs,
    NotebookSimpleTimeSeries,
    PrintBest,
)
from tune_notebook.monitors import _ReportBin


def test_report_bin():
    t1 = Trial("a", dict(a=1, b=2), keys=["x", "y"])
    r1 = TrialReport(t1, 0.8, sort_metric=-0.8)
    t2 = Trial("b", dict(a=11, b=12), keys=["xx", "y"])
    r2 = TrialReport(t2, 0.7, sort_metric=-0.7)
    t3 = Trial("c", dict(a=10, b=20), keys=["x", "y"])
    r3 = TrialReport(t3, 0.9, sort_metric=-0.9)

    b = _ReportBin()
    assert 0 == len(b.records)
    assert b.best is None
    assert b.on_report(r1)
    assert b.on_report(r3)
    assert r3 is b.best
    assert 2 == len(b.records)

    b = _ReportBin()
    assert b.on_report(r3)
    assert not b.on_report(r1)
    assert r3 is b.best
    assert 2 == len(b.records)

    b = _ReportBin(new_best_only=True)
    assert b.on_report(r3)
    assert not b.on_report(r1)
    assert r3 is b.best
    assert 1 == len(b.records)


def test_print_best():
    t1 = Trial("a", dict(a=1, b=2), keys=["x", "y"])
    r1 = TrialReport(t1, 0.8, sort_metric=-0.8)
    t2 = Trial("b", dict(a=11, b=12), keys=["xx", "y"])
    r2 = TrialReport(t2, 0.7, sort_metric=-0.7)
    t3 = Trial("c", dict(a=10, b=20), keys=["x", "y"])
    r3 = TrialReport(t3, 0.9, sort_metric=-0.9)

    b = PrintBest()
    b.on_report(r3)
    b.on_report(r2)
    b.on_report(r1)


def _test_charts():
    t1 = Trial("a", dict(a=1, b=2), keys=["x", "y"])
    r1 = TrialReport(t1, 0.8, sort_metric=-0.8)
    t2 = Trial("b", dict(a=11, b=12), keys=["xx", "y"])
    r2 = TrialReport(t2, 0.7, sort_metric=-0.7)
    t3 = Trial("c", dict(a=10, b=20), keys=["x", "y"])
    r3 = TrialReport(t3, 0.9, sort_metric=-0.9)

    b = NotebookSimpleTimeSeries("1sec")
    b.on_report(r3)
    b.on_report(r2)
    b.on_report(r1)

    b = NotebookSimpleHist("2sec")
    b.on_report(r3)
    b.on_report(r2)
    b.on_report(r1)

    b = NotebookSimpleRungs("3sec")
    b.on_report(r3)
    b.on_report(r2)
    b.on_report(r1)
