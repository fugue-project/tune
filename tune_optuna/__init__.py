# flake8: noqa

from tune_optuna.optimizer import OptunaLocalOptimizer
from tune import parse_noniterative_local_optimizer


@parse_noniterative_local_optimizer.candidate(
    lambda obj: isinstance(obj, str) and (obj == "optuna" or obj.startswith("optuna:"))
)
def _parse_optimizer(obj: str) -> OptunaLocalOptimizer:
    p = obj.split(":", 1)
    max_iter = 30 if len(p) == 1 else int(p[1])
    return OptunaLocalOptimizer(max_iter=max_iter)
