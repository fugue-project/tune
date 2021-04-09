from typing import Dict, Optional, Type

from fs.base import FS as FSBase
from tensorflow import keras
from tune import IterativeObjectiveFunc, TrialReport

from tune_tensorflow.spec import KerasTrainingSpec
from tune_tensorflow.utils import extract_keras_spec


class KerasObjective(IterativeObjectiveFunc):
    def __init__(self, type_dict: Dict[str, Type[KerasTrainingSpec]]) -> None:
        super().__init__()
        self._epochs = 0
        self._spec: Optional[KerasTrainingSpec] = None
        self._model: Optional[keras.models.Model] = None
        self._type_dict = type_dict

    @property
    def model(self) -> keras.models.Model:
        assert self._model is not None
        return self._model

    @property
    def spec(self) -> KerasTrainingSpec:
        assert self._spec is not None
        return self._spec

    def copy(self) -> "KerasObjective":
        return KerasObjective(self._type_dict)

    def generate_sort_metric(self, value: float) -> float:
        return self.spec.generate_sort_metric(value)

    def save_checkpoint(self, fs: FSBase) -> None:
        self.spec.save_checkpoint(fs, self.model)
        fs.writetext("epoch", str(self._epochs))

    def load_checkpoint(self, fs: FSBase) -> None:
        self.spec.load_checkpoint(fs, self.model)
        self._epochs = int(fs.readtext("epoch"))

    def run_single_rung(self, budget: float) -> TrialReport:
        trial = self.current_trial
        fit_args, fit_kwargs = self.spec.get_fit_params()
        fit_kwargs = dict(fit_kwargs)
        fit_kwargs.update(
            dict(epochs=self._epochs + int(budget), initial_epoch=self._epochs)
        )
        h = self.model.fit(*fit_args, **fit_kwargs)
        metric = self.spec.get_fit_metric(h)
        self._epochs += int(budget)
        return TrialReport(trial=trial, metric=metric, cost=budget, rung=self.rung)

    def initialize(self) -> None:
        spec = extract_keras_spec(self.current_trial.params, self._type_dict)
        self._spec = spec(self.current_trial.params, self.current_trial.dfs)
        self._model = self.spec.compile_model()

    def finalize(self) -> None:
        self.spec.finalize()
