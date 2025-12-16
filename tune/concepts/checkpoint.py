import json
from typing import Any, List
from uuid import uuid4

from fsspec.implementations.dirfs import DirFileSystem
from triad import assert_or_throw

_CHECKPOINT_STATE_FILE = "STATE"


class Checkpoint:
    """An abstraction for tuning checkpoint

    :param fs: the file system with base directory for checkpoint

    .. attention::

        Normally you don't need to create a checkpoint by yourself,
        please read :ref:`Checkpoint Tutorial </notebooks/checkpoint.ipynb>`
        if you want to understand how it works.
    """

    def __init__(self, fs: DirFileSystem):
        self._fs = fs
        try:
            with self._fs.open(_CHECKPOINT_STATE_FILE, "r") as f:
                self._iterations: List[str] = json.load(f)
        except Exception:
            self._iterations = []

    def __len__(self) -> int:
        """Count of the current iterations"""
        return len(self._iterations)

    @property
    def latest(self) -> DirFileSystem:
        """latest checkpoint folder

        :raises AssertionError: if there was no checkpoint
        """
        assert_or_throw(len(self) > 0, "checkpoint history is empty")
        return DirFileSystem(path=self._iterations[-1], fs=self._fs)

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

    def __enter__(self) -> DirFileSystem:
        self._parent._fs.makedir(self._name)
        return DirFileSystem(path=self._name, fs=self._parent._fs)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        if exc_type is not None:
            try:
                self._parent._fs.rm(self._name, recursive=True)
            except Exception:  # pragma: no cover
                pass
        else:
            new_iterations = self._parent._iterations + [self._name]
            with self._parent._fs.open(_CHECKPOINT_STATE_FILE, "w") as f:
                json.dump(new_iterations, f)
            self._parent._iterations = new_iterations
