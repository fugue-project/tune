from tests.tune_tensorflow.mock import MockSpec
from tune_tensorflow.utils import (
    extract_keras_spec,
    to_keras_spec,
    to_keras_spec_expr,
    keras_space,
)


def test_keras_space():
    space = keras_space(MockSpec, a=1, b=2)
    spec = extract_keras_spec(list(space.encode())[0], {})
    assert spec == MockSpec
    spec = extract_keras_spec(
        list(space.encode())[0], {to_keras_spec_expr(MockSpec): "dummy"}
    )
    assert "dummy" == spec


def test_to_keras_spec():
    expr = to_keras_spec_expr(MockSpec)
    assert to_keras_spec(expr) == MockSpec

    expr = to_keras_spec_expr(to_keras_spec_expr(MockSpec))
    assert to_keras_spec(expr) == MockSpec