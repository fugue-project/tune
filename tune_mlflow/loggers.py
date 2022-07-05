import os
from typing import Any, Dict, Optional, Union
from uuid import uuid4

import mlflow
from mlflow import ActiveRun
from mlflow.entities import Experiment, Run
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.tracking.context import registry as context_registry
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.mlflow_tags import (
    MLFLOW_PARENT_RUN_ID,
    MLFLOW_RUN_NAME,
    MLFLOW_RUN_NOTE,
)
from tune import parse_logger
from tune.concepts.logger import MetricLogger
from tune.exceptions import TuneRuntimeError


@parse_logger.candidate(lambda obj: isinstance(obj, (Run, ActiveRun)))
def _mlflow_run_to_logger(obj: Union[Run, ActiveRun]) -> "MLFlowRunLevelLogger":
    if MLFLOW_PARENT_RUN_ID in obj.data.tags:
        pr = mlflow.get_run(obj.data.tags[MLFLOW_PARENT_RUN_ID])
        parent: Any = _mlflow_run_to_logger(pr)
    else:
        client = MlflowClient()
        parent = MLFlowExperimentLevelLogger(
            client, mlflow.get_experiment(obj.info.experiment_id)
        )
    return MLFlowRunLevelLogger(parent, run_id=obj.info.run_id)


@parse_logger.candidate(lambda obj: isinstance(obj, str) and obj == "mlflow")
def _express_logger(obj: str) -> "MLFlowRunLevelLogger":
    return get_or_create_run()


def get_or_create_run(
    name: Optional[str] = None,
    description: Optional[str] = None,
    experiment_name: Optional[str] = None,
    run_id: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    registry_uri: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> "MLFlowRunLevelLogger":
    p_logger = get_or_create_experiment(
        experiment_name, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    return MLFlowRunLevelLogger(
        parent=p_logger, name=name, description=description, run_id=run_id, tags=tags
    )


def get_or_create_experiment(
    name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    registry_uri: Optional[str] = None,
) -> "MLFlowExperimentLevelLogger":
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    if name is None:
        eid = _get_experiment_id()
        exp = client.get_experiment(eid)
    else:
        try:
            eid = client.create_experiment(name)
            exp = client.get_experiment(eid)
        except MlflowException as e:
            if e.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
            exp = client.get_experiment_by_name(name)
    return MLFlowExperimentLevelLogger(client, exp)


class MLFlowLoggerBase(MetricLogger):
    def __init__(self, client: MlflowClient):
        self._client = client

    def __getstate__(self) -> Dict[str, Any]:
        raise TuneRuntimeError(str(type(self)) + " is not serializable")

    @property
    def client(self) -> MlflowClient:
        return self._client

    @property
    def tracking_uri(self) -> Optional[str]:
        return self.client._tracking_client.tracking_uri

    @property
    def registry_uri(self) -> Optional[str]:
        return self.client._registry_uri

    @property
    def run_id(self) -> str:
        raise TuneRuntimeError(str(type(self)) + " can't be directly used for logging")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self.client.log_metric(self.run_id, k, v)

    def log_params(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            self.client.log_param(self.run_id, k, v)

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        # self.client.log_dict(self.run_id, metadata)
        pass

    def create_child(self, name: str = None, is_step: bool = False) -> MetricLogger:
        raise TuneRuntimeError(str(type(self)) + " can't have a child logger")


class MLFlowExperimentLevelLogger(MLFlowLoggerBase):
    def __init__(self, client: MlflowClient, experiment: Experiment):
        super().__init__(client)
        self._experiment = experiment

    @property
    def experiment(self) -> Experiment:
        return self._experiment

    @property
    def experiment_id(self) -> str:
        return self.experiment.experiment_id

    def create_child(self, name: str = None, is_step: bool = False) -> MetricLogger:
        assert not is_step
        return MLFlowRunLevelLogger(self, name)


class MLFlowRunLevelLogger(MLFlowExperimentLevelLogger):
    def __init__(
        self,
        parent: MLFlowExperimentLevelLogger,
        name: Optional[str] = None,
        description: Optional[str] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(parent.client, parent.experiment)
        if run_id is None:
            t = {} if tags is None else dict(tags)
            if name is None:
                name = str(uuid4())[-5:]
            t[MLFLOW_RUN_NAME] = name
            if description is not None:
                t[MLFLOW_RUN_NOTE] = description
            if isinstance(parent, MLFlowRunLevelLogger):
                t[MLFLOW_PARENT_RUN_ID] = parent.run_id
                t["parent"] = (
                    parent.run_name
                    if parent.run_name is not None
                    else parent.run_id[-5:]
                )
                self._is_child = True
            else:
                self._is_child = False
            resolved_tags = context_registry.resolve_tags(t)
            self._run = self.client.create_run(self.experiment_id, tags=resolved_tags)
        else:
            self._run = self.client.get_run(run_id)
            self._is_child = False
        self._step = 0
        # Setup the env variable so mlflow.start_run() inside the objective will work
        os.environ["MLFLOW_RUN_ID"] = self.run_id

    def __getstate__(self) -> Dict[str, Any]:
        return dict(
            tracking_uri=self.tracking_uri,
            registry_uri=self.registry_uri,
            experiment_id=self.experiment_id,
            run_id=self.run_id,
            is_child=self._is_child,
            step=self._step,
        )

    def __setstate__(self, data: Dict[str, Any]) -> None:
        self._client = MlflowClient(
            tracking_uri=data["tracking_uri"], registry_uri=data["registry_uri"]
        )
        mlflow.tracking.set_tracking_uri(data["tracking_uri"])
        mlflow.tracking.set_registry_uri(data["registry_uri"])
        self._experiment = self._client.get_experiment(data["experiment_id"])
        self._run = self._client.get_run(data["run_id"])
        self._is_child = data["is_child"]
        self._step = data["step"]

    @property
    def run(self) -> Run:
        return self._run

    @property
    def run_id(self) -> str:
        return self.run.info.run_id

    @property
    def run_name(self) -> Optional[str]:
        return self.run.data.tags.get(MLFLOW_RUN_NAME, None)

    def create_child(self, name: str = None, is_step: bool = False) -> "MetricLogger":
        if is_step:
            self._step += 1
            return MLFlowStepLevelLogger(self, self._step - 1)
        else:
            assert not self._is_child
            return MLFlowRunLevelLogger(self, name=name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.set_terminated(
            self.run_id, "FINISHED" if exc_type is None else "FAILED"
        )


class MLFlowStepLevelLogger(MLFlowExperimentLevelLogger):
    def __init__(self, parent: MLFlowRunLevelLogger, step: int):
        super().__init__(parent.client, parent.experiment)
        self._run = parent.run
        self._step = step
        # Setup the env variable so mlflow.start_run() inside the objective will work
        os.environ["MLFLOW_RUN_ID"] = self.run_id

    @property
    def run(self) -> Run:
        return self._run

    @property
    def run_id(self) -> str:
        return self.run.info.run_id

    def create_child(self, name: str = None, is_step: bool = False) -> "MetricLogger":
        raise TuneRuntimeError(str(type(self)) + " can't have a child logger")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self.client.log_metric(self.run_id, k, v, step=self._step)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
