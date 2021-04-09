from typing import Any, Type, Dict

from triad.utils.convert import get_full_type_path, to_type

from tune_tensorflow.spec import KerasTrainingSpec
from tune import Space
from tune.constants import SPACE_MODEL_NAME

_TYPE_DICT: Dict[str, Type[KerasTrainingSpec]] = {}


def to_keras_spec(obj: Any) -> Type[KerasTrainingSpec]:
    if isinstance(obj, str) and obj in _TYPE_DICT:
        return _TYPE_DICT[obj]
    return to_type(obj, KerasTrainingSpec)


def to_keras_spec_expr(spec: Any) -> str:
    if isinstance(spec, str):
        spec = to_keras_spec(spec)
    return get_full_type_path(spec)


def keras_space(model: Any, **params: Any) -> Space:
    expr = to_keras_spec_expr(model)
    _TYPE_DICT[expr] = to_keras_spec(model)
    data = {SPACE_MODEL_NAME: expr}
    data.update(params)
    return Space(**data)


def extract_keras_spec(
    params: Dict[str, Any], type_dict: Dict[str, Any]
) -> Type[KerasTrainingSpec]:
    obj = params[SPACE_MODEL_NAME]
    if isinstance(obj, str) and obj in type_dict:
        return type_dict[obj]
    return to_keras_spec(obj)
