from fs.base import FS as FSBase
from triad import FileSystem
from tune.iterative.objective import (
    IterativeObjectiveFunc,
    validate_iterative_objective,
)
from tune.trial import Trial, TrialDecision, TrialJudge, TrialReport, Monitor


class F(IterativeObjectiveFunc):
    def __init__(self):
        self.v = -10
        super().__init__()

    def copy(self) -> "F":
        return F()

    def initialize(self) -> None:
        self.v = 0

    def finalize(self) -> None:
        self.v = -10

    def load_checkpoint(self, fs: FSBase) -> None:
        self.v = int(fs.readtext("x"))

    def save_checkpoint(self, fs: FSBase) -> None:
        fs.writetext("x", str(self.v))

    def run_single_iteration(self) -> TrialReport:
        self.v += 1
        return TrialReport(self.current_trial, self.v, metadata={"d": 4})

    def generate_sort_metric(self, value: float) -> float:
        return -value


class J(TrialJudge):
    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule

    def can_accept(self, trial: Trial) -> bool:
        return True

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


class M(Monitor):
    def __init__(self):
        self._reports = []

    def on_report(self, report: TrialReport) -> None:
        return self._reports.append(report)


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


def test_validator():
    m = M()
    for cont in [True, False]:
        validate_iterative_objective(
            F(),
            Trial("abc", {"a": 1}),
            [3, 3, 2],
            lambda reports: [-3.0, -6.0, -8.0] == [x.sort_metric for x in reports],
            continuous=cont,
            monitor=m,
        )
    assert 6 == len(m._reports)
