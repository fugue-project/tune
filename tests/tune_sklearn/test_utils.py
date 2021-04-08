from pytest import raises
from tune import Grid
from tune.constants import SPACE_MODEL_NAME

from tune_sklearn.utils import sk_space, to_sk_model, to_sk_model_expr


def test_conversion():
    model = to_sk_model("sklearn.linear_model.LinearRegression")
    expr = to_sk_model_expr(model)
    assert "sklearn.linear_model._base.LinearRegression" == expr
    expr = to_sk_model_expr("sklearn.linear_model.LinearRegression")
    assert "sklearn.linear_model._base.LinearRegression" == expr

    raises(TypeError, lambda: to_sk_model("int"))


def test_sk_space():
    rows = list(
        sk_space(
            "sklearn.linear_model.LinearRegression", fit_intercept=Grid(True, False)
        )
    )
    assert 2 == len(rows)
    assert "sklearn.linear_model._base.LinearRegression" == rows[0][SPACE_MODEL_NAME]
