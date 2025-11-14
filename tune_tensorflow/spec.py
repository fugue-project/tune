import tempfile
from typing import Any, Dict, List, Tuple

from fsspec.implementations.dirfs import DirFileSystem
from tensorflow import keras

from tune.concepts.space import TuningParametersTemplate, to_template


class KerasTrainingSpec:
    def __init__(self, params: Any, dfs: Dict[str, Any]):
        self._params = to_template(params)
        self._dfs = dfs

    @property
    def params(self) -> TuningParametersTemplate:
        return self._params

    @property
    def dfs(self) -> Dict[str, Any]:
        return self._dfs

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

    def save_checkpoint(self, fs: DirFileSystem, model: keras.models.Model) -> None:
        with tempfile.NamedTemporaryFile(suffix=".weights.h5") as tf:
            model.save_weights(tf.name)
            with open(tf.name, "rb") as fin:
                with fs.open("model.h5", "wb") as fout:
                    fout.write(fin.read())

    def load_checkpoint(self, fs: DirFileSystem, model: keras.models.Model) -> None:
        with tempfile.NamedTemporaryFile(suffix=".weights.h5") as tf:
            with fs.open("model.h5", "rb") as fin:
                with open(tf.name, "wb") as fout:
                    fout.write(fin.read())
            model.load_weights(tf.name)

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
