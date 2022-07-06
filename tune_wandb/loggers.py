import os
from typing import Any, Dict, Optional
from uuid import uuid4

import wandb
from triad import assert_or_throw
from tune import parse_logger
from tune.concepts.logger import MetricLogger
from tune.exceptions import TuneRuntimeError
from wandb.env import get_project
from wandb.sdk.lib.apikey import api_key
from wandb.wandb_run import Run


@parse_logger.candidate(
    lambda obj: isinstance(obj, str) and (obj == "wandb" or obj.startswith("wandb:"))
)
def _express_logger(obj: str) -> "WandbGroupLogger":
    p = obj.split(":", 1)
    project_name = p[1] if len(p) > 1 else "TUNE-DEFAULT"
    return WandbGroupLogger(project_name=project_name)


class WandbLoggerBase(MetricLogger):
    def __init__(self, run: Optional[Run] = None):
        self._run = run

    def __getstate__(self) -> Dict[str, Any]:
        raise TuneRuntimeError(str(type(self)) + " is not serializable")

    @property
    def run(self) -> Run:
        assert_or_throw(
            self._run is not None, NotImplementedError("wandb run is not available")
        )
        return self._run

    @property
    def api_key(self) -> str:
        return api_key(settings=None if self._run is None else self.run._settings)

    @property
    def project_name(self) -> str:
        return self.run.project_name()

    @property
    def group(self) -> str:
        return self.run.group

    @property
    def run_id(self) -> str:
        return self.run.id

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        self.run.log(metrics, commit=True)

    def log_params(self, params: Dict[str, Any]) -> None:
        self.run.config.update(params)

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        self.run.log(metadata, commit=True)

    def create_child(self, name: str = None, is_step: bool = False) -> MetricLogger:
        raise TuneRuntimeError(str(type(self)) + " can't have a child logger")

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        exit_code = 0 if exc_type is None else 1
        if self._run is not None:
            self.run.finish(exit_code)


class WandbProjectLogger(WandbLoggerBase):
    def __init__(self, name: str):
        self._name = name or get_project()
        super().__init__(None)

    @property
    def project_name(self) -> str:
        return self._name

    def create_child(self, name: str = None, is_step: bool = False) -> MetricLogger:
        assert_or_throw(not is_step, ValueError("can't create step logger"))
        return WandbGroupLogger(self.project_name, name)


class WandbGroupLogger(WandbLoggerBase):
    def __init__(self, project_name: str, group: Optional[str] = None):
        self._project_name = project_name
        self._group = group or str(uuid4())[-5:]
        super().__init__(None)

    def __getstate__(self) -> Dict[str, Any]:
        return dict(
            project_name=self.project_name, group=self.group, api_key=self.api_key
        )

    def __setstate__(self, data: Dict[str, Any]) -> None:
        os.environ["WANDB_SILENT"] = "true"
        wandb.login(key=data["api_key"], relogin=True)
        self._project_name = data["project_name"]
        self._group = data["group"]
        self._run = None

    @property
    def project_name(self) -> str:
        return self._project_name

    @property
    def group(self) -> str:
        return self._group

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        pass

    def create_child(self, name: str = None, is_step: bool = False) -> MetricLogger:
        if is_step:
            return MetricLogger()
        else:
            run = wandb.init(
                project=self.project_name,
                settings={"silent": True},
                group=self.group,
                name=name,
                reinit=True,
                allow_val_change=True,
            )
            return WandbLogger(run)


class WandbLogger(WandbLoggerBase):
    def __init__(self, run: Run):
        super().__init__(run)
        self._step = 0

    def create_child(self, name: str = None, is_step: bool = False) -> MetricLogger:
        assert_or_throw(
            name is None and is_step, ValueError("can't create child logger")
        )
        self._step += 1
        return WandbStepLogger(self.run, self._step - 1)


class WandbStepLogger(WandbLoggerBase):
    def __init__(self, run: Run, step: int):
        super().__init__(run)
        self._step = step

    def create_child(self, name: str = None, is_step: bool = False) -> MetricLogger:
        raise ValueError("can't create child logger")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        self.run.log(metrics, commit=True, step=self._step)

    def log_params(self, params: Dict[str, Any]) -> None:
        raise NotImplementedError("can't log parameters from a step logger")

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        self.run.log(metadata, commit=True, step=self._step)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass
