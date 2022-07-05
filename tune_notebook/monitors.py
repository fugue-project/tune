from datetime import datetime
from threading import RLock
from typing import Any, Dict, List

import pandas as pd
from triad.utils.convert import to_timedelta
from tune import Monitor, TrialReport, TrialReportLogger, parse_monitor


@parse_monitor.candidate(
    lambda obj: isinstance(obj, str) and obj == "hist", priority=0.0
)
def _hist(obj: str) -> Monitor:  # pragma: no cover
    return NotebookSimpleHist()


@parse_monitor.candidate(
    lambda obj: isinstance(obj, str) and obj == "rungs", priority=0.0
)
def _rungs(obj: str) -> Monitor:  # pragma: no cover
    return NotebookSimpleRungs()


@parse_monitor.candidate(lambda obj: isinstance(obj, str) and obj == "ts", priority=0.0)
def _ts(obj: str) -> Monitor:  # pragma: no cover
    return NotebookSimpleTimeSeries()


@parse_monitor.candidate(
    lambda obj: isinstance(obj, str) and obj == "text", priority=0.0
)
def _text(obj: str) -> Monitor:  # pragma: no cover
    return PrintBest()


class PrintBest(Monitor):
    def __init__(self):
        super().__init__()
        self._lock = RLock()
        self._bins: Dict[str, "_ReportBin"] = {}

    def on_report(self, report: TrialReport) -> None:
        with self._lock:
            key = str(report.trial.keys)
            if key not in self._bins:
                self._bins[key] = _ReportBin(new_best_only=True)
            rbin = self._bins[key]
        if rbin.on_report(report):
            print(report.trial.keys, report.metric, report)


class NotebookSimpleChart(Monitor):
    def __init__(
        self,
        interval: Any = "1sec",
        best_only: bool = True,
        always_update: bool = False,
    ):
        super().__init__()
        self._lock = RLock()
        self._last: Any = None
        self._bins: Dict[str, "_ReportBin"] = {}
        self._interval = to_timedelta(interval)
        self._best_only = best_only
        self._always_update = always_update

    def on_report(self, report: TrialReport) -> None:
        now = datetime.now()
        with self._lock:
            key = str(report.trial.keys)
            if key not in self._bins:
                self._bins[key] = _ReportBin(new_best_only=self._best_only)
            rbin = self._bins[key]

        updated = rbin.on_report(report)
        if not updated and not self._always_update:
            return

        with self._lock:
            if self._last is None or now - self._last > self._interval:
                self._redraw()
                self._last = datetime.now()

    def plot(self, df: pd.DataFrame) -> None:
        return  # pragma: no cover

    def finalize(self) -> None:
        self._redraw()

    def _redraw(self) -> None:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output

        df = pd.concat(
            [
                pd.DataFrame(
                    x.records,
                    columns=[
                        "partition",
                        "rung",
                        "time",
                        "id",
                        "metric",
                        "best_metric",
                    ],
                )
                for x in self._bins.values()
            ]
        )
        clear_output()
        self.plot(df)
        plt.show()
        for best in [x.best for x in self._bins.values() if x.best is not None]:
            if best is not None:
                print(best.trial.keys, best.metric, best)


class NotebookSimpleRungs(NotebookSimpleChart):
    def __init__(self, interval: Any = "1sec"):
        super().__init__(interval, best_only=False, always_update=True)

    def plot(self, df: pd.DataFrame) -> None:
        import seaborn as sns

        sns.lineplot(data=df, x="rung", y="metric", hue="id", marker="o", legend=False)


class NotebookSimpleTimeSeries(NotebookSimpleChart):
    def __init__(self, interval: Any = "1sec"):
        super().__init__(interval)

    def plot(self, df: pd.DataFrame) -> None:
        import seaborn as sns

        sns.lineplot(data=df, x="time", y="best_metric", hue="partition", marker="o")


class NotebookSimpleHist(NotebookSimpleChart):
    def __init__(self, interval: Any = "1sec"):
        super().__init__(interval, best_only=False, always_update=True)

    def plot(self, df: pd.DataFrame) -> None:
        import seaborn as sns

        sns.histplot(data=df, x="metric", hue="partition")


class _ReportBin(TrialReportLogger):
    def __init__(self, new_best_only: bool = False):
        super().__init__(new_best_only=new_best_only)
        self._values: List[List[Any]] = []

    def log(self, report: TrialReport) -> None:
        self._values.append(
            [
                str(report.trial.keys),
                report.rung,
                datetime.now(),
                report.trial_id,
                report.metric,
                self.best.metric,  # type: ignore
            ]
        )

    @property
    def records(self) -> List[List[Any]]:
        with self._lock:
            return list(self._values)
