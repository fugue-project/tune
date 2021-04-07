from tune_sklearn.utils import to_sk_model, to_sk_model_expr
from pytest import raises

def test_conversion():
    model = to_sk_model("sklearn.linear_model.LinearRegression")
    expr = to_sk_model_expr(model)
    assert "sklearn.linear_model._base.LinearRegression" == expr
    expr = to_sk_model_expr("sklearn.linear_model.LinearRegression")
    assert "sklearn.linear_model._base.LinearRegression" == expr

    raises(TypeError, lambda: to_sk_model("int"))