import cloudpickle
import base64
from typing import Any


def to_base64(obj: Any) -> str:
    """Convert an object to base64 string

    :param obj: a picklable object
    :return: a base64 string
    """
    return base64.b64encode(cloudpickle.dumps(obj)).decode("ascii")


def from_base64(data: str) -> Any:
    """Convert back to an object from a serialized base64 string

    :param data: base64 string
    :return: the unpickled data object
    """
    return cloudpickle.loads(base64.b64decode(data.encode("ascii")))  # type: ignore
