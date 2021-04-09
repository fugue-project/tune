from tensorflow.keras import layers, models
from tensorflow.keras.datasets import boston_housing

from tune_tensorflow import KerasTrainingSpec


class MockSpec(KerasTrainingSpec):
    def __init__(self, params, dfs):
        assert 0 == len(dfs)
        super().__init__(params, dfs)
        (self.train_data, self.train_targets), (
            self.test_data,
            self.test_targets,
        ) = boston_housing.load_data()

    def get_model(self):
        l1, l2 = self.params["l1"], self.params["l2"]
        model = models.Sequential()
        model.add(
            layers.Dense(l1, activation="relu", input_shape=(self.train_data.shape[1],))
        )
        model.add(layers.Dense(l2, activation="relu"))
        model.add(layers.Dense(1))
        return model

    def get_compile_params(self):
        return dict(optimizer="rmsprop", loss="mse", metrics=["mae"])

    def get_fit_params(self):
        return [self.train_data, self.train_targets], dict(
            validation_data=(self.test_data, self.test_targets), shuffle=True
        )

    def get_fit_metric(self, history):
        return float(history.history["val_mae"][-1])

    def generate_sort_metric(self, metric: float) -> float:
        return metric
