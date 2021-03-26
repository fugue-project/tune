from tune.trial import Trial, TrialReport


class NonIterativeObjectiveFunc:
    def run(self, trial: Trial) -> TrialReport:
        raise NotImplementedError  # pragma: no cover


class NonIterativeObjectiveRunner:
    def run(self, func: NonIterativeObjectiveFunc, trial: Trial) -> TrialReport:
        return func.run(trial)
