from datetime import datetime
from threading import RLock
from typing import Any, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import clear_output
from tune import Monitor, TrialReport


class NotebookSimpleChart(Monitor):
    def __init__(self):
        super().__init__()
        self._values: List[Any] = []
        self._lock = RLock()
        self._last: Any = None
        self._min: Any = None

    def on_report(self, report: TrialReport) -> None:
        with self._lock:
            now = datetime.now()
            if self._min is None:
                self._min = report.sort_metric
            else:
                self._min = min(self._min, report.sort_metric)
            self._values.append(
                [
                    str(report.trial.keys),
                    report.rung,
                    now,
                    report.trial_id,
                    report.sort_metric,
                    self._min,
                ]
            )
            if self._last is None or (now - self._last).total_seconds() > 1.0:
                clear_output()
                df = pd.DataFrame(
                    self._values,
                    columns=["partition", "rung", "time", "id", "metric", "min_metric"],
                )
                self.plot(df)
                plt.show()
                self._last = datetime.now()

    def plot(self, df: pd.DataFrame) -> None:
        return  # pragma: no cover


class NotebookSimpleRungs(NotebookSimpleChart):
    def plot(self, df: pd.DataFrame) -> None:
        sns.lineplot(data=df, x="rung", y="metric", hue="id", marker="o", legend=False)


class NotebookSimpleTimeSeries(NotebookSimpleChart):
    def plot(self, df: pd.DataFrame) -> None:
        sns.lineplot(data=df, x="time", y="min_metric", hue="partition", marker="o")


class NotebookSimpleHist(NotebookSimpleChart):
    def plot(self, df: pd.DataFrame) -> None:
        sns.histplot(data=df, x="metric", hue="partition")
