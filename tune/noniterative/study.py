from typing import Any, Dict

from tune.noniterative.objective import NonIterativeObjectiveFunc
from tune.space import Space
from tune.trial import Trial


class NonIterativeStudy:
    def run(self, objective: NonIterativeObjectiveFunc, space: Space) -> None:
        # for params in space:
        #    self.make_trial(params)
        pass

    def make_trial(self, params: Dict[str, Any]) -> Trial:
        raise NotImplementedError

    def run_trial(self, trial: Trial) -> None:
        raise NotImplementedError
