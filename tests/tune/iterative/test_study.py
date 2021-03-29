from fugue.workflow.workflow import FugueWorkflow
from tune.trial import Trial, TrialDecision, TrialReport
from tune.iterative.trial import IterativeTrial, TrialJudge
from tune.iterative.objective import MultiIterationObjectiveFunc
from tune.iterative.study import IterativeStudy
from tune.dataset import TuneDatasetBuilder
from tune.space import Space, Grid
from collections import defaultdict
import numpy as np


def f(x, a, b):
    return -np.log(x + 0.01) * a + b


class F(MultiIterationObjectiveFunc):
    def __init__(self) -> None:
        self.step = 0
        super().__init__()

    def copy(self) -> "F":
        return F()

    def preprocess(self) -> None:
        assert self.step == 0  # because of copy

    def run_single_iteration(self, trial: Trial) -> TrialReport:
        self.step += 1
        return TrialReport(
            trial=trial, metric=f(self.step, trial.params["a"], trial.params["b"])
        )


class J(TrialJudge):
    def __init__(self, n):
        self.n = n
        self.ct = defaultdict(int)

    def judge(self, report: TrialReport) -> TrialDecision:
        self.ct[report.trial_id] += 1
        return TrialDecision(
            report,
            should_stop=self.ct[report.trial_id] == self.n,
            should_checkpoint=False,
            metadata={},
        )


def test_iterative_study(tmpdir):
    study = IterativeStudy(F(), str(tmpdir))
    space = sum(
        Space(a=a, b=b)
        for a, b in [(1.1, 0.2), (0.8, -0.2), (1.2, -0.1), (0.7, 0.3), (1.0, 1.5)]
    )
    dag = FugueWorkflow()
    dataset = TuneDatasetBuilder(space, str(tmpdir)).build(dag)
    result = study.optimize(dataset, J(10))
    result.result(1).show()

    dag.run()
