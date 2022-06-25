from mlflow import ActiveRun
from tune.concepts.logger import register_logger_parser
from .loggers import _mlflow_run_to_logger, get_or_create_run


def register():
    register_logger_parser(lambda x: isinstance(x, ActiveRun), _mlflow_run_to_logger)
    register_logger_parser(
        lambda x: isinstance(x, str) and x == "mlflow", lambda y: get_or_create_run()
    )
