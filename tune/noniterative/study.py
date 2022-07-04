from copy import copy
from typing import Any, Callable, Dict, Iterable, Optional

from fugue import ArrayDataFrame, DataFrame, ExecutionEngine
from triad import assert_or_throw
from tune._utils import run_monitored_process
from tune.concepts.dataset import StudyResult, TuneDataset, _get_trials_from_row
from tune.concepts.flow import RemoteTrialJudge, TrialCallback, TrialJudge, TrialReport
from tune.concepts.flow.judge import Monitor, NoOpTrailJudge
from tune.constants import TUNE_REPORT_ADD_SCHEMA, TUNE_STOPPER_DEFAULT_CHECK_INTERVAL
from tune.exceptions import TuneCompileError, TuneInterrupted
from tune.noniterative.objective import (
    NonIterativeObjectiveFunc,
    NonIterativeObjectiveLocalOptimizer,
)
from tune.noniterative.stopper import NonIterativeStopper


def _make_judge(
    monitor: Optional[Monitor] = None, stopper: Optional[NonIterativeStopper] = None
) -> Optional[TrialJudge]:
    if monitor is None and stopper is None:
        return None
    if stopper is None and monitor is not None:
        return NoOpTrailJudge(monitor)
    if stopper is not None and monitor is None:
        return stopper
    if stopper is not None and monitor is not None:
        stopper.reset_monitor(monitor)
        return stopper
    raise NotImplementedError  # pragma: no cover


class NonIterativeStudy:
    def __init__(
        self,
        objective: NonIterativeObjectiveFunc,
        optimizer: NonIterativeObjectiveLocalOptimizer,
    ):
        self._objective = objective
        self._optimizer = optimizer

    def optimize(  # noqa: C901
        self,
        dataset: TuneDataset,
        distributed: Optional[bool] = None,
        monitor: Optional[Monitor] = None,
        stopper: Optional[NonIterativeStopper] = None,
        stop_check_interval: Any = None,
        logger: Any = None,
    ) -> StudyResult:
        _dist = self._get_distributed(distributed)
        entrypoint: Any = None
        judge = _make_judge(monitor, stopper)
        if judge is not None:
            cb = TrialCallback(judge)
            entrypoint = cb.entrypoint
        if stopper is None:
            _interval: Any = None
        else:
            _interval = stop_check_interval or TUNE_STOPPER_DEFAULT_CHECK_INTERVAL

        def compute_processor(engine: ExecutionEngine, df: DataFrame) -> DataFrame:
            out_schema = df.schema + TUNE_REPORT_ADD_SCHEMA

            def get_rows() -> Iterable[Any]:
                for row in self._compute_transformer(
                    df.as_local().as_dict_iterable(),
                    entrypoint=entrypoint,
                    stop_check_interval=_interval,
                    logger=logger,
                ):
                    yield [row[k] for k in out_schema.names]

            # TODO: need to add back execution_engine for engine aware optimizers
            # t._execution_engine = engine  # type:ignore
            return ArrayDataFrame(get_rows(), out_schema)

        def preprocess(df: DataFrame) -> DataFrame:
            if judge is not None:
                judge.monitor.initialize()
            return df

        def postprocess(df: DataFrame) -> None:
            if judge is not None:
                judge.monitor.finalize()

        if not _dist:
            res = dataset.data.process(preprocess).process(compute_processor)
        else:
            res = (
                dataset.data.process(preprocess)
                .per_row()
                .transform(
                    self._compute_transformer,
                    schema=f"*,{TUNE_REPORT_ADD_SCHEMA}",
                    callback=entrypoint,
                    params=dict(stop_check_interval=_interval, logger=logger),
                )
            )

        res.persist().output(postprocess)

        return StudyResult(dataset=dataset, result=res)

    def _get_distributed(self, distributed: Optional[bool]) -> bool:
        if distributed is None:
            return self._optimizer.distributable
        if distributed:
            assert_or_throw(
                self._optimizer.distributable,
                TuneCompileError(
                    f"can't distribute non-distributable optimizer {self._optimizer}"
                ),
            )
            return True
        return False

    def _compute_transformer(
        self,
        df: Iterable[Dict[str, Any]],
        entrypoint: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        stop_check_interval: Any = None,
        logger: Any = None,
    ) -> Iterable[Dict[str, Any]]:
        j: Optional[RemoteTrialJudge] = (
            None if entrypoint is None else RemoteTrialJudge(entrypoint)
        )
        for row in df:
            for n, trial in enumerate(_get_trials_from_row(row, with_dfs=False)):
                if j is not None:
                    if stop_check_interval is None:
                        # monitor only
                        report = self._local_process_trial(row, n, logger)
                        j.judge(report)
                        yield report.fill_dict(dict(row))
                    elif j.can_accept(trial):
                        try:
                            report = run_monitored_process(
                                self._local_process_trial,
                                [row, n],
                                {"logger": logger},
                                lambda: not j.can_accept(trial),  # type: ignore  # noqa
                                stop_check_interval,
                            )
                        except TuneInterrupted:
                            continue
                        j.judge(report)
                        yield report.fill_dict(dict(row))
                else:
                    report = self._local_process_trial(row, n, logger)
                    yield report.fill_dict(dict(row))

    def _local_process_trial(
        self, row: Dict[str, Any], idx: int, logger: Any
    ) -> TrialReport:
        trial = list(_get_trials_from_row(row))[idx]
        # This copy is to prevent statful objective to mess up state
        # between runs or between threads
        objective = copy(self._objective)
        return self._optimizer.run(objective, trial, logger=logger)
