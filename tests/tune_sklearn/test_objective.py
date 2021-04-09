from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from tune import RandInt, Trial
from tune.constants import SPACE_MODEL_NAME, TUNE_DATASET_DF_DEFAULT_NAME
from tune.noniterative.objective import validate_noniterative_objective
from tune_hyperopt import HyperoptRunner

from tune_sklearn.objective import SKCVObjective


def test_objective(tmpdir):
    dfs = load_iris(as_frame=True)
    df = dfs["data"]
    df["label"] = dfs["target"]
    df = df[df.label <= 1]

    t = Trial(
        "x",
        params={
            "max_iter": RandInt(2, 4),
            SPACE_MODEL_NAME: "sklearn.linear_model.LogisticRegression",
        },
        dfs={TUNE_DATASET_DF_DEFAULT_NAME: df},
    )
    obj = SKCVObjective(scoring="accuracy")
    runner = HyperoptRunner(5, 0)

    def v(report):
        print(report.jsondict)
        assert report.sort_metric < 0
        assert "cv_scores" in report.metadata
        # assert report.trial.params["max_iter"] >= 2

    validate_noniterative_objective(obj, t, v, runner=runner)

    obj = SKCVObjective(scoring="accuracy", checkpoint_path=str(tmpdir))

    def v2(report):
        print(report.jsondict)
        assert report.sort_metric < 0
        assert "cv_scores" in report.metadata
        assert "checkpoint_path" in report.metadata
        # assert report.trial.params["max_iter"] >= 2

    validate_noniterative_objective(obj, t, v2, runner=runner)
