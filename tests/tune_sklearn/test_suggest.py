from sklearn.datasets import load_diabetes

from tune_sklearn import sk_space, suggest_sk_models_by_cv
from tune import Grid, TUNE_OBJECT_FACTORY, Rand
from sklearn.linear_model import Lasso, LinearRegression
from fugue_dask import DaskExecutionEngine

# from fugue_spark import SparkExecutionEngine


def test_suggest_cv(tmpdir):
    TUNE_OBJECT_FACTORY.set_temp_path(str(tmpdir))

    data = load_diabetes(as_frame=True)
    train = data["data"]
    train["label"] = data["target"]

    space1 = sk_space(LinearRegression, fit_intercept=Grid(True, False))
    space2 = sk_space(
        Lasso, fit_intercept=Grid(True, False), alpha=Rand(0.5, 1)
    ).sample(3, 0)
    result = suggest_sk_models_by_cv(
        space1 + space2, train, "neg_mean_absolute_error", top_n=0, distributed=False
    )
    # print([x.jsondict for x in result])
    assert 8 == len(result)
    assert 50 > result[0].sort_metric

    result = suggest_sk_models_by_cv(
        space1 + space2,
        train,
        "neg_mean_absolute_error",
        top_n=0,
        partition_keys=["sex"],
    )
    assert 16 == len(result)
    assert 50 > result[0].sort_metric

    result = suggest_sk_models_by_cv(
        space1 + space2,
        train,
        "neg_mean_absolute_error",
        top_n=1,
        partition_keys=["sex"],
        execution_engine=DaskExecutionEngine,
    )
    assert 2 == len(result)
    assert 50 > result[0].sort_metric

    # TODO: why the following unit test is so slow

    # result = suggest_sk_models_by_cv(
    #     space1 + space2,
    #     train,
    #     "neg_mean_absolute_error",
    #     top_n=1,
    #     partition_keys=["sex"],
    #     execution_engine=SparkExecutionEngine,
    #     execution_engine_conf={
    #         "spark.sql.shuffle.partitions": 4,
    #         "spark.default.parallelism": 4,
    #         "spark.executor.cores": 4,
    #         "spark.sql.adaptive.enabled": "false",
    #     },
    # )
    # assert 2 == len(result)
    # assert 50 > result[0].sort_metric
