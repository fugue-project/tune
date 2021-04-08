import os
from typing import Any, Callable, List, Optional, Tuple
from uuid import uuid4

from triad import FileSystem

from tune.dataset import StudyResult, TuneDataset
from tune.iterative.asha import ASHAJudge, RungHeap
from tune.iterative.objective import IterativeObjectiveFunc
from tune.iterative.sha import _NonIterativeObjectiveWrapper
from tune.iterative.study import IterativeStudy
from tune.noniterative.objective import (
    NonIterativeObjectiveFunc,
    NonIterativeObjectiveRunner,
)
from tune.noniterative.study import NonIterativeStudy
from tune.trial import Monitor, TrialReport
from tune.factory import TUNE_OBJECT_FACTORY


def optimize_noniterative(
    objective: NonIterativeObjectiveFunc,
    dataset: TuneDataset,
    runner: Optional[NonIterativeObjectiveRunner] = None,
    distributed: Optional[bool] = None,
    monitor: Optional[Monitor] = None,
) -> StudyResult:
    _runner = TUNE_OBJECT_FACTORY.make_noniterative_objective_runner(runner)
    study = NonIterativeStudy(objective, _runner)
    return study.optimize(dataset, distributed=distributed, monitor=monitor)


def optimize_by_sha(
    objective: IterativeObjectiveFunc,
    dataset: TuneDataset,
    plan: List[Tuple[float, int]],
    checkpoint_path: str = "",
    distributed: Optional[bool] = None,
    monitor: Optional[Monitor] = None,
) -> StudyResult:
    checkpoint_path = TUNE_OBJECT_FACTORY.get_path_or_temp(checkpoint_path)
    path = os.path.join(checkpoint_path, str(uuid4()))
    for budget, keep in plan:
        obj = _NonIterativeObjectiveWrapper(
            objective, checkpoint_path=path, budget=budget
        )
        result = optimize_noniterative(
            obj, dataset, distributed=distributed, monitor=monitor
        )
        dataset = result.next_tune_dataset(keep)
    return result


def optimize_by_hyperband(
    objective: IterativeObjectiveFunc,
    dataset: TuneDataset,
    plans: List[List[Tuple[float, int]]],
    checkpoint_path: str = "",
    distributed: Optional[bool] = None,
    monitor: Optional[Monitor] = None,
) -> StudyResult:
    weights = [float(p[0][1]) for p in plans]
    datasets = dataset.divide(weights, seed=0)
    result: Any = None
    for d, plan in zip(datasets, plans):
        r = optimize_by_sha(
            objective=objective,
            dataset=d,
            plan=plan,
            checkpoint_path=checkpoint_path,
            distributed=distributed,
            monitor=monitor,
        )
        if result is None:
            result = r
        else:
            result.union_with(r)
    return result


def optimize_by_continuous_asha(
    objective: IterativeObjectiveFunc,
    dataset: TuneDataset,
    plan: List[Tuple[float, int]],
    checkpoint_path: str = "",
    always_checkpoint: bool = False,
    study_early_stop: Optional[Callable[[List[Any], List[RungHeap]], bool]] = None,
    trial_early_stop: Optional[
        Callable[[TrialReport, List[TrialReport], List[RungHeap]], bool]
    ] = None,
    monitor: Optional[Monitor] = None,
) -> StudyResult:
    checkpoint_path = TUNE_OBJECT_FACTORY.get_path_or_temp(checkpoint_path)
    judge = ASHAJudge(
        schedule=plan,
        always_checkpoint=always_checkpoint,
        study_early_stop=study_early_stop,
        trial_early_stop=trial_early_stop,
        monitor=monitor,
    )
    path = os.path.join(checkpoint_path, str(uuid4()))
    FileSystem().makedirs(path, recreate=True)
    study = IterativeStudy(objective, checkpoint_path=path)
    return study.optimize(dataset, judge=judge)
