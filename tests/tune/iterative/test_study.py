from typing import Any, Dict, Iterable

import numpy as np
from fugue.workflow.workflow import FugueWorkflow
from tune.constants import TUNE_REPORT_METRIC
from tune.dataset import TuneDatasetBuilder
from tune.iterative.objective import IterativeObjectiveFunc
from tune.iterative.study import IterativeStudy
from tune.space import Grid, Space
from tune.trial import Trial, TrialDecision, TrialJudge, TrialReport


def f(x, a, b):
    return -np.log(x + 0.01) * a + b


class F(IterativeObjectiveFunc):
    def __init__(self) -> None:
        self.step = 0
        super().__init__()

    def copy(self) -> "F":
        return F()

    def initialize(self) -> None:
        assert self.step == 0  # because of copy

    def run_single_iteration(self) -> TrialReport:
        self.step += 1
        trial = self.current_trial
        return TrialReport(
            trial=trial, metric=f(self.step, trial.params["a"], trial.params["b"])
        )


class J(TrialJudge):
    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule

    def can_accept(self, trial: Trial) -> bool:
        return True

    def get_budget(self, trial: Trial, rung: int) -> float:
        return float(self.schedule[rung]) if rung < len(self.schedule) else 0.0

    def judge(self, report: TrialReport) -> TrialDecision:
        return TrialDecision(
            report,
            budget=self.get_budget(report.trial, report.rung + 1),
            should_checkpoint=False,
            metadata={},
        )


def test_iterative_study(tmpdir):
    def assert_metric(df: Iterable[Dict[str, Any]], metric: float) -> None:
        for row in df:
            assert row[TUNE_REPORT_METRIC] < metric

    study = IterativeStudy(F(), str(tmpdir))
    space = sum(
        Space(a=a, b=b)
        for a, b in [(1.1, 0.2), (0.8, -0.2), (1.2, -0.1), (0.7, 0.3), (1.0, 1.5)]
    )
    dag = FugueWorkflow()
    dataset = TuneDatasetBuilder(space, str(tmpdir)).build(dag)
    result = study.optimize(
        dataset,
        J([1, 2, 3, 4]),
    )
    result.result(1).show()
    result.result(1).output(assert_metric, params=dict(metric=-2.8))

    dag.run()
