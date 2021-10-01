import json
from typing import Any, List
from uuid import uuid4

from fs.base import FS as FSBase
from triad import assert_or_throw

_CHECKPOINT_STATE_FILE = "STATE"


class Checkpoint:
    """An abstraction for tuning checkpoint

    :param fs: the file system

    .. attention::

        Normally you don't need to create a checkpoint by yourself,
        please read :ref:`Checkpoint Tutorial </notebooks/checkpoint.ipynb>`
        if you want to understand how it works.
    """

    def __init__(self, fs: FSBase):
        self._fs = fs
        try:
            self._iterations: List[str] = json.loads(
                fs.readtext(_CHECKPOINT_STATE_FILE)
            )
        except Exception:
            self._iterations = []

    def __len__(self) -> int:
        """Count of the current iterations"""
        return len(self._iterations)

    @property
    def latest(self) -> FSBase:
        """latest checkpoint folder

        :raises AssertionError: if there was no checkpoint
        """
        assert_or_throw(len(self) > 0, "checkpoint history is empty")
        return self._fs.opendir(self._iterations[-1])

    def create(self) -> "NewCheckpoint":
        """Create a new checkpoint"""
        return NewCheckpoint(self)


class NewCheckpoint:
    """A helper class for adding new checkpoints

    :param checkpoint: the parent checkpoint

    .. attention::

        Do not construct this class directly, please read
        :ref:`Checkpoint Tutorial </notebooks/checkpoint.ipynb>`
        for details
    """

    def __init__(self, checkpoint: Checkpoint):
        self._parent = checkpoint
        self._name = str(uuid4())

    def __enter__(self) -> FSBase:
        return self._parent._fs.makedir(self._name)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        if exc_type is not None:
            try:
                self._parent._fs.removetree(self._name)
            except Exception:  # pragma: no cover
                pass
        else:
            new_iterations = self._parent._iterations + [self._name]
            self._parent._fs.writetext(
                _CHECKPOINT_STATE_FILE, json.dumps(new_iterations)
            )
            self._parent._iterations = new_iterations
