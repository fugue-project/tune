from fs.base import FS as FSBase
from triad import FileSystem
from tune.iterative.objective import IterativeObjectiveFunc
from tune.iterative.trial import TrialJudge
from tune.trial import Trial, TrialDecision, TrialReport


class F(IterativeObjectiveFunc):
    def __init__(self):
        self.v = -10
        super().__init__()

    def copy(self) -> "F":
        return F()

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

    def generate_sort_metric(self, value: float) -> float:
        return -value


class J(TrialJudge):
    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule

    def get_budget(self, trial: Trial, rung: int) -> float:
        return float(self.schedule[rung]) if rung < len(self.schedule) else 0.0

    def judge(self, report: TrialReport) -> TrialDecision:
        self.report = report
        return TrialDecision(
            report,
            budget=0,  # stop at each rung
            should_checkpoint=True,
            metadata={"x": 1},
        )


def test_objective_func(tmpdir):
    fs = FileSystem().opendir(str(tmpdir))
    j = J([3, 3, 2])
    f = F().copy()
    t = Trial("abc", {"a": 1})
    f.run(t, judge=j, checkpoint_basedir_fs=fs)
    assert -10 == f.v
    f.run(t, judge=j, checkpoint_basedir_fs=fs)
    assert -10 == f.v
    assert 6.0 == j.report.metric
    assert -6.0 == j.report.sort_metric
    f.run(t, judge=j, checkpoint_basedir_fs=fs)
    assert -10 == f.v
    assert 8.0 == j.report.metric
    assert -8.0 == j.report.sort_metric
