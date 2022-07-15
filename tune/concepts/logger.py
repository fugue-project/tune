from typing import Any, Dict, Optional
from uuid import uuid4

from tune.concepts.flow.report import TrialReport


def make_logger(obj: Any) -> "MetricLogger":
    """Convert an object to a MetricLogger. This function is usually called on
    the worker side. If ``obj`` is a function, then it can take the context of
    the worker environment to initialize. For example mlflow will be able to take
    the worker side environment variables to initialize.

    :param obj: the object, currently we support ``MetricLogger`` or
        a callalbe generating a ``MetricLogger`` or ``None`` for a dummy logger
    :return: the logger
    """
    if obj is None:
        return MetricLogger()
    if isinstance(obj, MetricLogger):
        return obj
    if callable(obj):
        return obj()
    raise ValueError(f"{obj} can't be converted to a MetricLogger")


class MetricLogger:
    """Hierarchical metric logger for objectectives"""

    def __init__(self):
        self._unique_id = str(uuid4())[-5:]

    @property
    def unique_id(self) -> str:
        return self._unique_id

    def create_child(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_step: bool = False,
    ) -> "MetricLogger":
        """Create a child logger

        :param name: the name of the child logger, defaults to None
        :param description: the long description of the child logger, defaults to None
        :param is_step: whether the child logger is a sub-step, for example for
            an epoch of a deep learning model, it should set to ``True``

        :return: the child logger
        """
        return self

    def log_report(
        self,
        report: TrialReport,
        log_params: bool = False,
        log_metadata: bool = False,
        extract_metrics: bool = False,
    ) -> None:
        """Log information from a TrialReport

        :param report: the report
        :param log_params: whether to log the hyperparameter from the report,
            defaults to True
        :param log_metadata: whether to log the metadata from the report,
            defaults to True
        :param extract_metrics: whether to extract more metrics from the report
            and log as metric_, defaults to True
        """
        all_metrics = {"OBJECTIVE_METRIC": report.metric}
        if extract_metrics:
            all_metrics.update(
                {
                    k: float(v)
                    for k, v in report.metadata.items()
                    if isinstance(v, (int, float))
                }
            )
        self.log_metrics(all_metrics)
        if log_params:
            self.log_params(report.trial.params.simple_value)
        if log_metadata:
            self.log_metadata(report.metadata)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """The output metrics of the objective

        :param metrics: a dictionary of float numbers
        """
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        """The hyperparameters of the objective

        :param params: a dictionary of hyperparameters
        """
        pass

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        """The metadata output of the objective

        :param metadata: a dictionary of metadata
        """
        pass

    def __enter__(self) -> "MetricLogger":
        """Context manager

        .. admonition:: Examples

            .. code-block:: python

                with MetricLogger() as logger:
                    logger.log_metrics({})
                    with logger.create_child("abc") as sublogger:
                        sublogger.log_metrics({})
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """On exit of the context"""
        pass
