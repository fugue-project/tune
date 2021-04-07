from datetime import datetime
from threading import RLock
from typing import Any, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import clear_output
from tune import Monitor, TrialReport


class NotebookSimpleChart(Monitor):
    def __init__(self, by_time: bool = False):
        super().__init__()
        self._values: List[Any] = []
        self._lock = RLock()
        self._last: Any = None
        self._by_time = by_time
        self._min: Any = None

    def on_report(self, report: TrialReport) -> None:
        with self._lock:
            now = datetime.now()
            if self._min is None:
                self._min = report.sort_metric
            else:
                self._min = min(self._min, report.sort_metric)
            self._values.append(
                [report.rung, now, report.trial_id, report.sort_metric, self._min]
            )
            if self._last is None or (now - self._last).total_seconds() > 1.0:
                clear_output()
                df = pd.DataFrame(
                    self._values, columns=["rung", "time", "id", "metric", "min_metric"]
                )
                if not self._by_time:
                    sns.lineplot(data=df, x="rung", y="metric", hue="id", legend=False)
                else:
                    sns.lineplot(data=df, x="time", y="min_metric", legend=False)
                plt.show()
                self._last = datetime.now()
