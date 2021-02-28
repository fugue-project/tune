from tune.trial.trial import Trial


class Objective:
    def __call__(self, trial: Trial) -> None:
        return


class ObjectiveRunner:
    def run(self, objective: Objective, trial: Trial) -> None:
        objective(trial)
