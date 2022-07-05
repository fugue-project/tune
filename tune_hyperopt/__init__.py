# flake8: noqa

from tune_hyperopt.optimizer import HyperoptLocalOptimizer
from tune import parse_noniterative_local_optimizer


@parse_noniterative_local_optimizer.candidate(
    lambda obj: isinstance(obj, str)
    and (obj == "hyperopt" or obj.startswith("hyperopt:"))
)
def _parse_optimizer(obj: str) -> HyperoptLocalOptimizer:
    p = obj.split(":", 1)
    max_iter = 30 if len(p) == 1 else int(p[1])
    return HyperoptLocalOptimizer(max_iter=max_iter, seed=0)
