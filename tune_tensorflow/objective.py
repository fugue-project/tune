import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from tensorflow import keras
from fs.base import FS as FSBase
from triad import FileSystem
from tune import IterativeObjectiveFunc, TrialReport


class KerasTrainingSpec:
    def __init__(self, params: Dict[str, Any]):
        self._params = params

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    def finalize(self) -> None:
        pass

    def generate_sort_metric(self, metric: float) -> float:
        raise NotImplementedError  # pragma: no cover

    def get_fit_metric(self, history: keras.callbacks.History) -> float:
        raise NotImplementedError  # pragma: no cover

    def get_fit_params(self) -> Tuple[List[Any], Dict[str, Any]]:
        raise NotImplementedError  # pragma: no cover

    def get_compile_params(self) -> Dict[str, Any]:
        raise NotImplementedError  # pragma: no cover

    def get_model(self) -> keras.models.Model:
        raise NotImplementedError  # pragma: no cover

    def compile_model(self, **add_kwargs: Any) -> keras.models.Model:
        params = dict(self.get_compile_params())
        params.update(add_kwargs)
        model = self.get_model()
        model.compile(**params)
        return model

    def fit(self, **add_kwargs: Any) -> keras.callbacks.History:
        args, kwargs = self.get_fit_params()
        kwargs = dict(kwargs)
        kwargs.update(add_kwargs)
        model = self.compile_model()
        metric = model.fit(*args, **kwargs)
        self.finalize()
        return metric

    def compute_sort_metric(self, **add_kwargs: Any) -> float:
        metric = self.get_fit_metric(self.fit(**add_kwargs))
        return self.generate_sort_metric(metric)


class KerasObjective(IterativeObjectiveFunc):
    def __init__(
        self,
        init_spec: Union[
            Type[KerasTrainingSpec], Callable[[Dict[str, Any]], KerasTrainingSpec]
        ],
    ) -> None:
        super().__init__()
        self._epochs = 0
        self._spec: Optional[KerasTrainingSpec] = None
        self._model: Optional[keras.models.Model] = None
        self._init_spec = init_spec

    @property
    def model(self) -> keras.models.Model:
        assert self._model is not None
        return self._model

    @property
    def spec(self) -> KerasTrainingSpec:
        assert self._spec is not None
        return self._spec

    def copy(self) -> "KerasObjective":
        return KerasObjective(self._init_spec)

    def generate_sort_metric(self, value: float) -> float:
        return self.spec.generate_sort_metric(value)

    def save_checkpoint(self, fs: FSBase) -> None:
        with tempfile.NamedTemporaryFile(suffix=".h5") as tf:
            self.model.save_weights(tf.name)
            with open(tf.name, "rb") as fin:
                fs.writefile("model.h5", fin)
        fs.writetext("epoch", str(self._epochs))

    def load_checkpoint(self, fs: FSBase) -> None:
        with tempfile.NamedTemporaryFile(suffix=".h5") as tf:
            local_fs = FileSystem()
            with fs.open("model.h5", "rb") as fin:
                local_fs.writefile(tf.name, fin)
            self.model.load_weights(tf.name)
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
        self._spec = self._init_spec(self.current_trial.params)
        self._model = self.spec.compile_model()

    def finalize(self) -> None:
        self.spec.finalize()
