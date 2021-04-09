import json

from fugue import (
    ArrayDataFrame,
    DataFrame,
    FugueWorkflow,
    NativeExecutionEngine,
    WorkflowDataFrames,
)
import pickle

import fugue

from tune.constants import (
    TUNE_DATASET_DF_PREFIX,
    TUNE_DATASET_PARAMS_PREFIX,
    TUNE_DATASET_TRIALS,
    TUNE_TEMP_PATH,
)
from tune.dataset import TuneDatasetBuilder, _to_trail_row, TuneDataset
from tune.space import Grid, Rand
from tune.space.spaces import Space
from tune.trial import Trial


def test_builder(tmpdir):
    space = Space(a=1, b=2, c=Grid(2, 3))
    builder = TuneDatasetBuilder(space, str(tmpdir))

    def assert_count(df: DataFrame, n: int, schema=None) -> None:
        assert len(df.as_array()) == n
        if schema is not None:
            assert df.schema == schema

    # test to_space
    with FugueWorkflow() as dag:
        df = builder.build(dag).data
        df.show()

    df1 = ArrayDataFrame([[0, 1], [1, 1], [0, 2]], "a:int,b:int")

    # test single df
    with FugueWorkflow() as dag:
        builder.add_dfs(WorkflowDataFrames(x=dag.df(df1)))
        dataset = builder.build(dag)
        assert ["x"] == dataset.dfs
        assert [] == dataset.keys
        df = dataset.data
        df.show()
        df.output(
            assert_count,
            params=dict(n=2, schema=f"__tune_df__x:str,{TUNE_DATASET_TRIALS}:str"),
        )

    space = Space(b=Rand(0, 1), a=1, c=Grid(2, 3), d=Grid("a", "b"))
    df2 = ArrayDataFrame([[0, 1], [1, 1], [3, 2]], "a:int,bb:int")
    df3 = ArrayDataFrame([[10, 1], [11, 1], [10, 2]], "a:int,c:int")
    builder = TuneDatasetBuilder(space)
    engine = NativeExecutionEngine(conf={TUNE_TEMP_PATH: str(tmpdir)})

    # test multiple dfs, batch_size and config
    with FugueWorkflow(engine) as dag:
        dfs = WorkflowDataFrames(
            a=dag.df(df1).partition_by("a"), b=dag.df(df2).partition_by("a")
        )
        dataset = (
            builder.add_dfs(dfs, "inner").add_df("c", dag.df(df3), "cross").build(dag)
        )
        assert ["a"] == dataset.keys
        assert ["a", "b", "c"] == dataset.dfs
        df = dataset.data
        df.show()
        df.output(
            assert_count,
            params=dict(
                n=8,
                schema="a:int,__tune_df__a:str,__tune_df__b:str,"
                f"__tune_df__c:str,{TUNE_DATASET_TRIALS}:str",
            ),
        )

        df = builder.build(dag, batch_size=3).data
        df.show()
        df.output(
            assert_count,
            params=dict(
                n=4,
                schema="a:int,__tune_df__a:str,__tune_df__b:str,"
                f"__tune_df__c:str,{TUNE_DATASET_TRIALS}:str",
            ),
        )


def test_dataset(tmpdir):
    space = Space(a=Grid(0, 1, 2, 3, 4), b=Grid(5, 6, 7, 8, 9))
    builder = TuneDatasetBuilder(space, str(tmpdir))

    dag = FugueWorkflow()
    dataset = builder.build(dag)
    ds = dataset.divide([4, 1], 0)
    assert 2 == len(ds)
    ds[0].data.yield_dataframe_as("a")
    ds[1].data.yield_dataframe_as("b")
    res = dag.run()
    assert 25 == len(res["a"].as_array()) + len(res["b"].as_array())
    assert len(res["b"].as_array()) < 10


def test_to_trial_row():
    data1 = {
        "b": 2,
        "a": 1,
        TUNE_DATASET_DF_PREFIX + "x": "x",
        TUNE_DATASET_PARAMS_PREFIX: pickle.dumps(
            [{"b": 10, "a": 11}, {"a": 11, "b": 10}, {"b": 100, "a": 110}],
        ),
    }
    res1 = _to_trail_row(data1, {"m": 1})
    trials1 = [Trial(**p) for p in json.loads(res1[TUNE_DATASET_TRIALS])]
    assert 2 == len(trials1)
    data2 = {
        "a": 1,
        "b": 2,
        TUNE_DATASET_DF_PREFIX + "y": "x",
        TUNE_DATASET_PARAMS_PREFIX: pickle.dumps(
            [{"b": 10, "a": 11}, {"b": 100, "a": 110}],
        ),
    }
    res2 = _to_trail_row(data2, {"m": 1})
    assert TUNE_DATASET_PARAMS_PREFIX not in res2
    trials2 = [Trial(**p) for p in json.loads(res2[TUNE_DATASET_TRIALS])]
    assert 2 == len(trials2)
    assert any(trials2[0].trial_id == x.trial_id for x in trials1)
    assert any(trials2[1].trial_id == x.trial_id for x in trials1)

    data3 = {
        "a": 10,
        "b": 2,
        TUNE_DATASET_DF_PREFIX + "y": "x",
        TUNE_DATASET_PARAMS_PREFIX: pickle.dumps(
            [{"b": 10, "a": 11}, {"b": 100, "a": 110}],
        ),
    }
    res3 = _to_trail_row(data3, {"m": 1})
    trials3 = [Trial(**p) for p in json.loads(res3[TUNE_DATASET_TRIALS])]
    assert 2 == len(trials2)
    assert not any(trials3[0].trial_id == x.trial_id for x in trials1)
    assert not any(trials3[1].trial_id == x.trial_id for x in trials1)
