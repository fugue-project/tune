from typing import List, Optional, Tuple

from fs.base import FS as FSBase
from triad import FileSystem
from tune.dataset import TuneDataset
from tune.iterative.objective import IterativeObjectiveFunc
from tune.iterative.trial import TrialDecision, TrialJudge
from tune.noniterative.objective import NonIterativeObjectiveFunc
from tune.trial import Trial, TrialReport


def run_sha(
    objective: IterativeObjectiveFunc,
    dataset: TuneDataset,
    plan: List[Tuple[float, int]],
    checkpoint_path: str,
) -> None:
    obj = _to_noniterative_objective(
        objective, checkpoint_path=checkpoint_path, plan=plan
    )
    print(obj)
    # run_noniterative_study()


def _to_noniterative_objective(
    func: IterativeObjectiveFunc,
    checkpoint_path: FSBase,
    plan: List[Tuple[float, int]],
) -> NonIterativeObjectiveFunc:
    return _NonIterativeObjectiveWrapper(func, checkpoint_path, plan)


class _NonIterativeObjectiveWrapper(NonIterativeObjectiveFunc):
    def __init__(
        self,
        func: IterativeObjectiveFunc,
        checkpoint_path: str,
        plan: List[Tuple[float, int]],
    ):
        super().__init__()
        self._plan = plan
        self._func = func
        self._checkpoint_path = checkpoint_path

    def generate_sort_metric(self, value: float) -> float:
        return self._func.generate_sort_metric(value)

    def run(self, trial: Trial) -> TrialReport:  # pragma: no cover
        judge = _NonIterativeJudgeWrapper(self._plan)
        fs = FileSystem().makedirs(self._checkpoint_path, recreate=True)
        self._func.run(trial, judge=judge, checkpoint_basedir_fs=fs)
        return judge.report


class _NonIterativeJudgeWrapper(TrialJudge):
    def __init__(self, plan: List[Tuple[float, int]]):
        super().__init__()
        self._report: Optional[TrialReport] = None
        self._plan = plan

    @property
    def report(self) -> TrialReport:
        assert self._report is not None
        return self._report

    def get_budget(self, trial: Trial, rung: int):
        if rung >= len(self._plan):
            return 0
        return self._plan[rung][0]

    def judge(self, report: TrialReport) -> TrialDecision:
        self._report = report
        return TrialDecision(report, budget=0.0, should_checkpoint=True, metadata={})
