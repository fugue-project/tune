from fugue.exceptions import FugueWorkflowCompileError, FugueWorkflowRuntimeError


class TuneCompileError(FugueWorkflowCompileError):
    pass


class TuneRuntimeError(FugueWorkflowRuntimeError):
    pass
