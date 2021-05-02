from typing import Any, Callable, Dict, Iterable

from triad import FileSystem
from tune.constants import TUNE_REPORT_ADD_SCHEMA
from tune.dataset import StudyResult, TuneDataset, get_trials_from_row
from tune.iterative.objective import IterativeObjectiveFunc
from tune.trial import RemoteTrialJudge, Trial, TrialJudge, TrialReport


class TrialCallback:
    def __init__(self, judge: TrialJudge):
        self._judge = judge

    def entrypoint(self, name, kwargs: Dict[str, Any]) -> Any:
        if name == "can_accept":
            return self._judge.can_accept(Trial.from_jsondict(kwargs["trial"]))
        if name == "judge":
            return self._judge.judge(TrialReport.from_jsondict(kwargs)).jsondict
        if name == "get_budget":
            return self._judge.get_budget(
                Trial.from_jsondict(kwargs["trial"]), kwargs["rung"]
            )
        raise NotImplementedError  # pragma: no cover


class IterativeStudy:
    def __init__(self, objective: IterativeObjectiveFunc, checkpoint_path: str):
        self._objective = objective
        self._checkpoint_path = checkpoint_path

    def optimize(self, dataset: TuneDataset, judge: TrialJudge) -> StudyResult:
        callback = TrialCallback(judge)

        res = dataset.data.per_row().transform(
            self._compute,
            schema=f"*,{TUNE_REPORT_ADD_SCHEMA}",
            callback=callback.entrypoint,
        )

        return StudyResult(dataset=dataset, result=res)

    def _compute(
        self,
        df: Iterable[Dict[str, Any]],
        entrypoint: Callable[[str, Dict[str, Any]], Any],
    ) -> Iterable[Dict[str, Any]]:
        ck_fs = FileSystem().makedirs(self._checkpoint_path, recreate=True)
        for row in df:
            for trial in get_trials_from_row(row):
                rjudge = RemoteTrialJudge(entrypoint)
                self._objective.copy().run(trial, rjudge, ck_fs)
                if rjudge.report is not None:
                    yield rjudge.report.fill_dict(dict(row))
