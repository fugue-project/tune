import json
from typing import Any, Callable, Dict, Iterable, Optional

from triad.collections.fs import FileSystem
from tune.constants import (
    TUNE_REPORT,
    TUNE_REPORT_METRIC,
)
from tune.dataset import StudyResult, TuneDataset, get_trials_from_row
from tune.iterative.objective import (
    IterativeObjectiveFunc,
    IterativeObjectiveRunner,
    TrialJudge,
)
from tune.trial import TrialDecision, TrialReport


class TrialCallback:
    def __init__(self, judge: TrialJudge):
        self._judge = judge

    def entrypoint(self, name, kwargs: Dict[str, Any]) -> Any:
        if name == "judge":
            return self.run_judge(kwargs)
        raise NotImplementedError  # pragma: no cover

    def run_judge(self, kwargs: Dict[str, Any]):
        return self._judge.judge(TrialReport.from_jsondict(kwargs)).jsondict


class RemoteTrialJudge(TrialJudge):
    def __init__(self, entrypoint: Callable[[str, Dict[str, Any]], Any]):
        self._entrypoint = entrypoint
        self._report: Optional[TrialReport] = None

    @property
    def report(self) -> Optional[TrialReport]:
        return self._report

    def judge(self, report: TrialReport) -> TrialDecision:
        self._report = report
        return TrialDecision.from_jsondict(self._entrypoint("judge", report.jsondict))


class IterativeStudy:
    def __init__(self, objective: IterativeObjectiveFunc, checkpoint_path: str):
        self._objective = objective
        self._checkpoint_path = checkpoint_path

    def optimize(self, dataset: TuneDataset, judge: TrialJudge) -> StudyResult:
        add_schema = f"{TUNE_REPORT_METRIC}:double,{TUNE_REPORT}:str"
        callback = TrialCallback(judge)

        res = dataset.data.per_row().transform(
            self._compute,
            schema=f"*,{add_schema}",
            params=dict(per_config_budget=100.0),
            callback=callback.entrypoint,
        )

        return StudyResult(
            dataset=dataset, result=res, min_better=self._objective.min_better
        )

    def _compute(
        self,
        df: Iterable[Dict[str, Any]],
        entrypoint: Callable[[str, Dict[str, Any]], Any],
        per_config_budget: float,
    ) -> Iterable[Dict[str, Any]]:
        ck_fs = FileSystem().makedirs(self._checkpoint_path, recreate=True)
        for row in df:
            for trial in get_trials_from_row(row):
                rjudge = RemoteTrialJudge(entrypoint)
                runner = IterativeObjectiveRunner(rjudge, ck_fs)
                runner.run(self._objective.copy(), trial, budget=per_config_budget)
                if rjudge.report is not None:
                    res = dict(row)
                    res[TUNE_REPORT_METRIC] = rjudge.report.metric
                    res[TUNE_REPORT] = json.dumps(rjudge.report.jsondict)
                    yield res
