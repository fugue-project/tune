from typing import List, Optional, Tuple

from tune.constants import TUNE_REPORT, TUNE_REPORT_METRIC
from tune.dataset import StudyResult, TuneDataset
from tune.iterative.asha import ASHAJudge
from tune.iterative.objective import IterativeObjectiveFunc
from tune.iterative.study import IterativeStudy, TrialCallback
from tune.iterative.trial import TrialJudgeMonitor


class SHAStudy(IterativeStudy):
    def __init__(
        self,
        objective: IterativeObjectiveFunc,
        checkpoint_path: str,
    ):
        super().__init__(objective, checkpoint_path)

    def sha(
        self,
        dataset: TuneDataset,
        plan: List[Tuple[float, int]],
        monitor: Optional[TrialJudgeMonitor] = None,
    ) -> StudyResult:
        add_schema = f"{TUNE_REPORT_METRIC}:double,{TUNE_REPORT}:str"

        for budget, keep in plan:
            judge = ASHAJudge(
                [(budget, keep)],
                always_checkpoint=True,
                should_stop_func=lambda *ars, **kwargs: False,
                trial_should_stop_func=lambda *args, **kwargs: False,
                monitor=monitor,
            )
            callback = TrialCallback(judge)

            res = dataset.data.per_row().transform(
                self._compute,
                schema=f"*,{add_schema}",
                callback=callback.entrypoint,
            )
            study_res = StudyResult(dataset=dataset, result=res)
            dataset = study_res.next_tune_dataset(keep)

        return study_res
