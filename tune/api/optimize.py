import os
from typing import Any, Callable, List, Optional, Tuple
from uuid import uuid4

from triad import FileSystem
from tune.api.factory import (
    TUNE_OBJECT_FACTORY,
    parse_iterative_objective,
    parse_monitor,
    parse_noniterative_local_optimizer,
    parse_noniterative_objective,
    parse_noniterative_stopper,
)
from tune.concepts.dataset import StudyResult, TuneDataset
from tune.concepts.flow import TrialReport
from tune.iterative.asha import ASHAJudge, RungHeap
from tune.iterative.sha import _NonIterativeObjectiveWrapper
from tune.iterative.study import IterativeStudy
from tune.noniterative.study import NonIterativeStudy


def optimize_noniterative(
    objective: Any,
    dataset: TuneDataset,
    optimizer: Any = None,
    distributed: Optional[bool] = None,
    logger: Any = None,
    monitor: Any = None,
    stopper: Any = None,
    stop_check_interval: Any = None,
) -> StudyResult:
    _objective = parse_noniterative_objective(objective)
    _optimizer = parse_noniterative_local_optimizer(optimizer)
    _stopper = parse_noniterative_stopper(stopper)
    _monitor = parse_monitor(monitor)
    study = NonIterativeStudy(_objective, _optimizer)
    return study.optimize(
        dataset,
        distributed=distributed,
        monitor=_monitor,
        stopper=_stopper,
        stop_check_interval=stop_check_interval,
        logger=logger,
    )


def optimize_by_sha(
    objective: Any,
    dataset: TuneDataset,
    plan: List[Tuple[float, int]],
    checkpoint_path: str = "",
    distributed: Optional[bool] = None,
    monitor: Any = None,
) -> StudyResult:
    _objective = parse_iterative_objective(objective)
    _monitor = parse_monitor(monitor)
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
    _monitor = parse_monitor(monitor)
    weights = [float(p[0][1]) for p in plans]
    datasets = dataset.split(weights, seed=0)
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
    _objective = parse_iterative_objective(objective)
    _monitor = parse_monitor(monitor)
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
