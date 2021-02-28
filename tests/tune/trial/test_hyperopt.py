from typing import Any, Dict

from tune import Rand, RandInt
from tune.trial import Objective, ReportFeedback, ReportHandler, Trial
from tune.trial.hyperopt import HyperoptRunner


def test_hyperopt():
    params = dict(a=Rand(-10.0, 10.0), b=RandInt(-100, 100), c=2.0)
    handler = MockHandler()
    trial = Trial("a", params, metadata={}, report_handler=handler)
    he = HyperoptRunner(max_iter=100, seed=0)

    class Obj(Objective):
        def __call__(self, trial: Trial) -> None:
            trial.report(
                trial.params["a"] ** 2 + trial.params["b"] ** 2 + trial.params["c"],
                metadata=dict(a=1),
            )

    he.run(Obj(), trial)

    params = dict(a=0.0, b=0.0, c=2.0)
    handler = MockHandler()
    trial = Trial("a", params, metadata={}, report_handler=handler)
    he = HyperoptRunner(max_iter=100, seed=0)

    he.run(Obj(), trial)


class MockHandler(ReportHandler):
    def __init__(self):
        self.called = False

    def __call__(
        self,
        trial: "Trial",
        params: Dict[str, Any],
        metric: float,
        metadata: Dict[str, Any],
    ) -> ReportFeedback:
        assert not self.called
        self.called = True
        assert params["a"] < 1.0
        assert params["b"] < 2.0
        assert params["c"] == 2.0
        assert metric < 5.0
        assert dict(a=1) == metadata
        return ReportFeedback(should_stop=True, should_checkpoint=False)
