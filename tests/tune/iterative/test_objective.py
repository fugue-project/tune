from fs.base import FS as FSBase
from triad import FileSystem
from tune.iterative.objective import (
    IterativeObjectiveRunner,
    MultiIterationObjectiveFunc,
)
from tune.iterative.trial import TrialJudge
from tune.trial import Trial, TrialDecision, TrialReport


class F(MultiIterationObjectiveFunc):
    def __init__(self):
        self.v = -10

    def preprocess(self) -> None:
        self.v = 0

    def postprocess(self) -> None:
        self.v = -10

    def load_checkpoint(self, fs: FSBase) -> None:
        self.v = int(fs.readtext("x"))

    def save_checkpoint(self, fs: FSBase) -> None:
        fs.writetext("x", str(self.v))

    def run_single_iteration(self, trial: Trial) -> TrialReport:
        self.v += 1
        return TrialReport(trial, self.v, metadata={"d": 4})


class J(TrialJudge):
    def judge(self, report: TrialReport) -> TrialDecision:
        self.report = report
        return TrialDecision(
            report,
            should_stop=int(report.metric) in [3, 6, 9],
            should_checkpoint=int(report.metric) in [2, 3, 6],
            metadata={"x": 1},
        )


def test_objective_runner(tmpdir):
    fs = FileSystem().opendir(str(tmpdir))
    j = J()
    runner = IterativeObjectiveRunner(j, fs)
    f = F()
    runner.run(f, Trial("abc", {"a": 1}), 100)
    assert -10 == f.v
    runner.run(f, Trial("abc", {"a": 1}), 100)
    assert -10 == f.v
    assert 6.0 == j.report.metric
    runner.run(f, Trial("abc", {"a": 1}), 2)
    assert -10 == f.v
    assert 8.0 == j.report.metric
