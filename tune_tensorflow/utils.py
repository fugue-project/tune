from typing import Any, Type, Dict

from triad.utils.convert import get_full_type_path, to_type

from tune_tensorflow.spec import KerasTrainingSpec
from tune import Space
from tune.constants import SPACE_MODEL_NAME


def to_keras_spec(obj: Any) -> Type[KerasTrainingSpec]:
    return to_type(obj, KerasTrainingSpec)


def to_keras_spec_expr(spec: Any) -> Any:
    if isinstance(spec, str):
        spec = to_keras_spec(spec)
    return get_full_type_path(spec)


def keras_space(model: str, **params: Any) -> Space:
    data = {SPACE_MODEL_NAME: to_keras_spec_expr(model)}
    data.update(params)
    return Space(**data)


def extract_keras_spec(params: Dict[str, Any]) -> Type[KerasTrainingSpec]:
    return to_keras_spec(params[SPACE_MODEL_NAME])
