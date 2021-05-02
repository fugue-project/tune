import os
from typing import Any, Callable, List, Optional, Tuple
from uuid import uuid4

from triad import FileSystem

from tune.dataset import StudyResult, TuneDataset
from tune.factory import TUNE_OBJECT_FACTORY
from tune.iterative.asha import ASHAJudge, RungHeap
from tune.iterative.sha import _NonIterativeObjectiveWrapper
from tune.iterative.study import IterativeStudy
from tune.noniterative.study import NonIterativeStudy
from tune.trial import TrialJudge, TrialReport, NoOpTrailJudge


def make_stopper(monitor: Any, stopper: Any) -> Optional[TrialJudge]:
    _stopper = TUNE_OBJECT_FACTORY.make_stopper(stopper)
    _monitor = TUNE_OBJECT_FACTORY.make_monitor(monitor)
    if _monitor is None and _stopper is None:
        return None
    if _stopper is None and _monitor is not None:
        return NoOpTrailJudge(_monitor)
    if _stopper is not None and _monitor is None:
        return _stopper
    if _stopper is not None and _monitor is not None:
        _stopper.reset_monitor(_monitor)
        return _stopper
    raise NotImplementedError  # pragma: no cover


def optimize_noniterative(
    objective: Any,
    dataset: TuneDataset,
    runner: Any = None,
    distributed: Optional[bool] = None,
    monitor: Any = None,
    stopper: Any = None,
) -> StudyResult:
    _objective = TUNE_OBJECT_FACTORY.make_noniterative_objective(objective)
    _runner = TUNE_OBJECT_FACTORY.make_noniterative_objective_runner(runner)
    study = NonIterativeStudy(_objective, _runner)
    judge = make_stopper(monitor, stopper)
    return study.optimize(dataset, distributed=distributed, judge=judge)


def optimize_by_sha(
    objective: Any,
    dataset: TuneDataset,
    plan: List[Tuple[float, int]],
    checkpoint_path: str = "",
    distributed: Optional[bool] = None,
    monitor: Any = None,
) -> StudyResult:
    _objective = TUNE_OBJECT_FACTORY.make_iterative_objective(objective)
    _monitor = TUNE_OBJECT_FACTORY.make_monitor(monitor)
    checkpoint_path = TUNE_OBJECT_FACTORY.get_path_or_temp(checkpoint_path)
    path = os.path.join(checkpoint_path, str(uuid4()))
    for budget, keep in plan:
        obj = _NonIterativeObjectiveWrapper(
            _objective, checkpoint_path=path, budget=budget
        )
        result = optimize_noniterative(
            obj, dataset, distributed=distributed, monitor=_monitor
        )
        dataset = result.next_tune_dataset(keep)
    return result


def optimize_by_hyperband(
    objective: Any,
    dataset: TuneDataset,
    plans: List[List[Tuple[float, int]]],
    checkpoint_path: str = "",
    distributed: Optional[bool] = None,
    monitor: Any = None,
) -> StudyResult:
    _monitor = TUNE_OBJECT_FACTORY.make_monitor(monitor)
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
            monitor=_monitor,
        )
        if result is None:
            result = r
        else:
            result.union_with(r)
    return result


def optimize_by_continuous_asha(
    objective: Any,
    dataset: TuneDataset,
    plan: List[Tuple[float, int]],
    checkpoint_path: str = "",
    always_checkpoint: bool = False,
    study_early_stop: Optional[Callable[[List[Any], List[RungHeap]], bool]] = None,
    trial_early_stop: Optional[
        Callable[[TrialReport, List[TrialReport], List[RungHeap]], bool]
    ] = None,
    monitor: Any = None,
) -> StudyResult:
    _objective = TUNE_OBJECT_FACTORY.make_iterative_objective(objective)
    _monitor = TUNE_OBJECT_FACTORY.make_monitor(monitor)
    checkpoint_path = TUNE_OBJECT_FACTORY.get_path_or_temp(checkpoint_path)
    judge = ASHAJudge(
        schedule=plan,
        always_checkpoint=always_checkpoint,
        study_early_stop=study_early_stop,
        trial_early_stop=trial_early_stop,
        monitor=_monitor,
    )
    path = os.path.join(checkpoint_path, str(uuid4()))
    FileSystem().makedirs(path, recreate=True)
    study = IterativeStudy(_objective, checkpoint_path=path)
    return study.optimize(dataset, judge=judge)
