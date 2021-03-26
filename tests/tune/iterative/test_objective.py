from tune.iterative.objective import IterativeObjectiveFunc, IterativeObjectiveRunner
from tune.iterative.trial import IterativeTrial, TrialJudge
from tune.trial import TrialDecision, TrialReport, Trial
from triad import FileSystem
from tune.checkpoint import Checkpoint


class F(IterativeObjectiveFunc):
    def run(self, trial: IterativeTrial) -> None:
        v = 0
        if trial.iteration > 0:
            v = int(trial.checkpoint.latest.readtext("x"))
        while True:
            v += 1
            res = trial.report(
                v, metadata={"d": 4}, save_checkpoint=lambda f: f.writetext("x", str(v))
            )
            if res.should_stop:
                return


class J(TrialJudge):
    def judge(self, report: TrialReport) -> TrialDecision:
        self.report = report
        return TrialDecision(
            report,
            should_stop=int(report.metric) % 3 == 0,
            should_checkpoint=int(report.metric) % 2 == 1,
            metadata={"x": 1},
        )


def test_objective_runner(tmpdir):
    fs = FileSystem().opendir(str(tmpdir))
    j = J()
    runner = IterativeObjectiveRunner(j, fs)
    f = F()
    runner.run(f, Trial("abc", {"a": 1}, {"b": 2}))
    runner.run(f, Trial("abc", {"a": 1}, {"b": 2}))
    assert 6.0 == j.report.metric
