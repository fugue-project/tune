from tune import Trial, validate_iterative_objective

from tune_tensorflow.objective import KerasObjective

from .mock import MockSpec


def test_spec():
    spec = MockSpec(dict(l1=16, l2=16))
    metric = spec.compute_sort_metric(epochs=10)
    assert metric < 10


def test_objective():
    def validate(reports):
        assert reports[-1].metric < 10

    obj = KerasObjective(MockSpec)

    for cont in [True, False]:
        validate_iterative_objective(
            obj,
            Trial("a", params=dict(l1=16, l2=16)),
            budgets=[3, 3, 4],
            continuous=cont,
            validator=validate,
        )
