import os
import pickle
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from triad import FileSystem
from tune import NonIterativeObjectiveFunc, Trial, TrialReport
from tune.constants import SPACE_MODEL_NAME, TUNE_DATASET_DF_DEFAULT_NAME

from tune_sklearn.utils import to_sk_model, to_sk_model_expr


class SKCVObjective(NonIterativeObjectiveFunc):
    def __init__(
        self,
        scoring: Any,
        cv: int = 5,
        feature_prefix: str = "",
        label_col: str = "label",
        checkpoint_path: str = "",
    ) -> None:
        super().__init__()
        self._last_id = ""
        self._model_type: Any = None
        self._model_expr: str = ""

        self._scoring = scoring
        self._cv = cv
        self._feature_prefix = feature_prefix
        self._label_col = label_col
        self._checkpoint_path = checkpoint_path

    def generate_sort_metric(self, value: float) -> float:
        return -value

    def run(self, trial: Trial) -> TrialReport:
        params = dict(trial.params)
        if trial.trial_id != self._last_id:
            self._model_type = to_sk_model(params.pop(SPACE_MODEL_NAME))
            self._model_expr = to_sk_model_expr(self._model_type)
            self._reset_xy(trial.dfs[TUNE_DATASET_DF_DEFAULT_NAME])
            self._last_id = trial.trial_id
        else:
            params.pop(SPACE_MODEL_NAME)

        model = self._model_type(**params)
        s = cross_val_score(
            model, self._train_x, self._train_y, cv=self._cv, scoring=self._scoring
        )
        metadata = dict(model=self._model_expr, cv_scores=[float(x) for x in s])
        if self._checkpoint_path != "":
            model.fit(self._train_x, self._train_y)
            fp = os.path.join(self._checkpoint_path, str(uuid4()) + ".pkl")
            with FileSystem().openbin(fp, mode="wb") as f:
                pickle.dump(model, f)
            metadata["checkpoint_path"] = fp
        metric = float(np.mean(s))
        return TrialReport(
            trial,
            metric=metric,
            metadata=metadata,
            sort_metric=self.generate_sort_metric(metric),
        )

    def _reset_xy(self, df: pd.DataFrame) -> None:
        train_df = df.sample(frac=1, random_state=0).reset_index(drop=True)

        train_x = train_df.drop([self._label_col], axis=1)
        cols = [x for x in train_x.columns if x.startswith(self._feature_prefix)]
        self._train_x = train_x[cols]
        self._train_y = train_df[self._label_col]
