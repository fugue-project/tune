from typing import Any, Dict, Type

from sklearn.base import is_classifier, is_regressor
from triad import assert_or_throw
from triad.utils.convert import get_full_type_path, to_type
from tune.constants import SPACE_MODEL_NAME
from tune.space.spaces import Space


def to_sk_model(obj: Any) -> Type:
    if isinstance(obj, str):
        obj = to_type(obj)
    assert_or_throw(
        is_classifier(obj) or is_regressor(obj),
        TypeError(f"{obj} is neither a sklearn classifier or regressor"),
    )
    return obj


def to_sk_model_expr(model: Any) -> Any:
    if isinstance(model, str):
        model = to_sk_model(model)
    return get_full_type_path(model)


def sk_space(model: str, **params: Dict[str, Any]) -> Space:
    data = {SPACE_MODEL_NAME: to_sk_model_expr(model)}
    data.update(params)
    return Space(**data)
