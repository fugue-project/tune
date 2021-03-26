from tune.space.spaces import Space
from tune.dataset import TuneDatasetBuilder
from fugue import (
    FugueWorkflow,
    ArrayDataFrame,
    WorkflowDataFrames,
    DataFrame,
    NativeExecutionEngine,
)
from tune.space import Grid, Rand
from tune.constants import TUNE_TEMP_PATH, TUNE_DATASET_PARAMS_PREFIX


def test_builder(tmpdir):
    space = Space(a=1, b=2, c=Grid(2, 3))
    builder = TuneDatasetBuilder(space, str(tmpdir))

    def assert_count(df: DataFrame, n: int, schema=None) -> None:
        assert len(df.as_array()) == n
        if schema is not None:
            assert df.schema == schema

    # test to_space
    with FugueWorkflow() as dag:
        df = builder.build(dag)
        df.show()
        df.assert_eq(
            dag.df(
                [
                    ['[{"a": 1, "b": 2, "c": 2}]'],
                    ['[{"a": 1, "b": 2, "c": 3}]'],
                ],
                f"{TUNE_DATASET_PARAMS_PREFIX}:str",
            )
        )

    df1 = ArrayDataFrame([[0, 1], [1, 1], [0, 2]], "a:int,b:int")

    # test single df
    with FugueWorkflow() as dag:
        builder.add_dfs(WorkflowDataFrames(x=dag.df(df1)))
        df = builder.build(dag)
        df.show()
        df.output(
            assert_count,
            params=dict(n=2, schema="__tune_df__x:str,__tune_params__:str"),
        )

    space = Space(a=1, b=Rand(0, 1), c=Grid(2, 3), d=Grid("a", "b"))
    df2 = ArrayDataFrame([[0, 1], [1, 1], [3, 2]], "a:int,bb:int")
    df3 = ArrayDataFrame([[10, 1], [11, 1], [10, 2]], "a:int,c:int")
    builder = TuneDatasetBuilder(space)
    engine = NativeExecutionEngine(conf={TUNE_TEMP_PATH: str(tmpdir)})

    # test multiple dfs, batch_size and config
    with FugueWorkflow(engine) as dag:
        dfs = WorkflowDataFrames(
            a=dag.df(df1).partition_by("a"), b=dag.df(df2).partition_by("a")
        )
        builder.add_dfs(dfs, "inner")
        builder.add_df("c", dag.df(df3), "cross")
        df = builder.build(dag)
        df.show()
        df.output(
            assert_count,
            params=dict(
                n=8,
                schema="a:int,__tune_df__a:str,__tune_df__b:str,"
                "__tune_df__c:str,__tune_params__:str",
            ),
        )

        df = builder.build(dag, batch_size=3)
        df.show()
        df.output(
            assert_count,
            params=dict(
                n=4,
                schema="a:int,__tune_df__a:str,__tune_df__b:str,"
                "__tune_df__c:str,__tune_params__:str",
            ),
        )
