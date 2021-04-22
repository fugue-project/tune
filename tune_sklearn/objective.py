import os
import pickle
from tune.factory import TUNE_OBJECT_FACTORY
from typing import Any, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score
from triad import FileSystem
from tune import NonIterativeObjectiveFunc, Trial, TrialReport
from tune.constants import (
    SPACE_MODEL_NAME,
    TUNE_DATASET_DF_DEFAULT_NAME,
    TUNE_DATASET_VALIDATION_DF_DEFAULT_NAME,
)

from tune_sklearn.utils import to_sk_model, to_sk_model_expr


class SKObjective(NonIterativeObjectiveFunc):
    def __init__(
        self,
        scoring: Any,
        feature_prefix: str = "",
        label_col: str = "label",
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._last_id = ""
        self._model_type: Any = None
        self._model_expr: str = ""

        self._scoring = scoring
        self._feature_prefix = feature_prefix
        self._label_col = label_col
        if checkpoint_path is None:
            self._checkpoint_path = checkpoint_path
        else:
            self._checkpoint_path = TUNE_OBJECT_FACTORY.get_path_or_temp(
                checkpoint_path
            )

    def generate_sort_metric(self, value: float) -> float:
        return -value

    def run(self, trial: Trial) -> TrialReport:
        params = dict(trial.params)
        if trial.trial_id != self._last_id:
            self._model_type = to_sk_model(params.pop(SPACE_MODEL_NAME))
            self._model_expr = to_sk_model_expr(self._model_type)
            self._train_x, self._train_y = self._reset_xy(
                trial.dfs[TUNE_DATASET_DF_DEFAULT_NAME]
            )
            self._test_x, self._test_y = self._reset_xy(
                trial.dfs[TUNE_DATASET_VALIDATION_DF_DEFAULT_NAME]
            )
            self._last_id = trial.trial_id
        else:
            params.pop(SPACE_MODEL_NAME)

        model = self._model_type(**params).fit(self._train_x, self._train_y)
        metric = get_scorer(self._scoring)(model, self._test_x, self._test_y)
        metadata = dict(model=self._model_expr)
        if self._checkpoint_path is not None:
            fp = os.path.join(self._checkpoint_path, str(uuid4()) + ".pkl")
            with FileSystem().openbin(fp, mode="wb") as f:
                pickle.dump(model, f)
            metadata["checkpoint_path"] = fp
        return TrialReport(
            trial,
            metric=metric,
            metadata=metadata,
            sort_metric=self.generate_sort_metric(metric),
        )

    def _reset_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = df.sample(frac=1, random_state=0).reset_index(drop=True)

        train_x = train_df.drop([self._label_col], axis=1)
        cols = [x for x in train_x.columns if x.startswith(self._feature_prefix)]
        return train_x[cols], train_df[self._label_col]


class SKCVObjective(SKObjective):
    def __init__(
        self,
        scoring: Any,
        cv: int = 5,
        feature_prefix: str = "",
        label_col: str = "label",
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            scoring=scoring,
            feature_prefix=feature_prefix,
            label_col=label_col,
            checkpoint_path=checkpoint_path,
        )
        self._cv = cv

    def run(self, trial: Trial) -> TrialReport:
        params = dict(trial.params)
        if trial.trial_id != self._last_id:
            self._model_type = to_sk_model(params.pop(SPACE_MODEL_NAME))
            self._model_expr = to_sk_model_expr(self._model_type)
            self._train_x, self._train_y = self._reset_xy(
                trial.dfs[TUNE_DATASET_DF_DEFAULT_NAME]
            )
            self._last_id = trial.trial_id
        else:
            params.pop(SPACE_MODEL_NAME)

        model = self._model_type(**params)
        s = cross_val_score(
            model, self._train_x, self._train_y, cv=self._cv, scoring=self._scoring
        )
        metadata = dict(model=self._model_expr, cv_scores=[float(x) for x in s])
        if self._checkpoint_path is not None:
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
