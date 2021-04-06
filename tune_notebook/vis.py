from datetime import datetime
from threading import RLock

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import clear_output
from tune import Monitor, TrialReport


class NotebookSimpleChart(Monitor):
    def __init__(self):
        super().__init__()
        self._values = []
        self._lock = RLock()
        self._last = None

    def on_report(self, report: TrialReport) -> None:
        now = datetime.now()
        with self._lock:
            self._values.append([report.rung, now, report.trial_id, report.sort_metric])
            if self._last is None or (now - self._last).total_seconds() > 1.0:
                clear_output()
                df = pd.DataFrame(
                    self._values, columns=["rung", "time", "id", "metric"]
                )
                sns.lineplot(data=df, x="rung", y="metric", hue="id", legend=False)
                plt.show()
                self._last = datetime.now()
