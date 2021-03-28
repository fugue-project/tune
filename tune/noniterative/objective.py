from tune.trial import Trial, TrialReport


class NonIterativeObjectiveFunc:
    @property
    def min_better(self) -> bool:
        return True  # pragma: no cover

    def run(self, trial: Trial) -> TrialReport:
        raise NotImplementedError  # pragma: no cover


class NonIterativeObjectiveRunner:
    @property
    def distributable(self) -> bool:
        return True

    def run(self, func: NonIterativeObjectiveFunc, trial: Trial) -> TrialReport:
        # TODO: how to utilize execution_engine?
        return func.run(trial)
