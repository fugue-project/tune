from tune import Trial, validate_iterative_objective

from tune_tensorflow import KerasObjective, keras_space
from tune_tensorflow.suggest import _to_objective

from .mock import MockSpec


def test_spec():
    spec = MockSpec(dict(l1=16, l2=16), {})
    metric = spec.compute_sort_metric(epochs=10)
    assert metric < 10


def test_objective():
    def validate(reports):
        assert reports[-1].metric < 10

    space = keras_space(MockSpec, l1=16, l2=16)
    obj = KerasObjective(_to_objective(space))
    assert MockSpec in obj._spec_types

    for cont in [True, False]:
        validate_iterative_objective(
            obj,
            Trial("a", params=list(space)[0]),
            budgets=[3, 3, 4],
            continuous=cont,
            validator=validate,
        )
