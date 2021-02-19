from typing import Any

from fugue.exceptions import FugueWorkflowCompileError, FugueWorkflowRuntimeError


class TuneCompileError(FugueWorkflowCompileError):
    def __init__(self, *args: Any):
        super().__init__(*args)


class TuneRuntimeError(FugueWorkflowRuntimeError):
    def __init__(self, *args: Any):
        super().__init__(*args)
