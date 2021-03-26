import json
from typing import Any, Dict, Iterable, List

import pandas as pd
from fugue import (
    ExecutionEngine,
    FugueWorkflow,
    IterableDataFrame,
    NativeExecutionEngine,
    WorkflowDataFrame,
    make_execution_engine,
)
from pytest import raises

from tune import (
    Grid,
    Space,
    select_best,
    serialize_df,
    space_to_df,
    tunable,
    tune,
    visualize_top_n,
)
from tune.constants import TUNE_TEMP_PATH
from tune.exceptions import TuneCompileError


def test_space_to_df():
    with FugueWorkflow() as dag:
        df = space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)))
        df.assert_eq(
            dag.df(
                [
                    ['[{"a": 0, "b": 2}]'],
                    ['[{"a": 0, "b": 3}]'],
                    ['[{"a": 1, "b": 2}]'],
                    ['[{"a": 1, "b": 3}]'],
                ],
                "__fmin_params__:str",
            )
        )

    with FugueWorkflow() as dag:
        df = space_to_df(
            dag, Space(a=Grid(0, 1), b=Grid(2, 3)), batch_size=3, shuffle=False
        )
        df.assert_eq(
            dag.df(
                [
                    ['[{"a": 0, "b": 2}, {"a": 0, "b": 3}, {"a": 1, "b": 2}]'],
                    ['[{"a": 1, "b": 3}]'],
                ],
                "__fmin_params__:str",
            )
        )


def test_tune_simple():
    def t1(a: int, b: int) -> float:
        return a + b

    for distributable in [True, False, None]:
        with FugueWorkflow() as dag:
            df = space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)))
            tune(df, t1, distributable=distributable).show()

    @tunable()
    def t2(e: ExecutionEngine, a: int, b: int) -> float:
        assert isinstance(e, ExecutionEngine)
        return a + b

    for distributable in [False, None]:
        with FugueWorkflow() as dag:
            df = space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)))
            tune(df, t2, distributable=distributable).show()

    # equivalent syntax sugar
    with FugueWorkflow() as dag:
        t2.space(a=Grid(0, 1), b=Grid(2, 3)).tune(dag).show()

    with raises(TuneCompileError):
        with FugueWorkflow() as dag:
            df = space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)))
            tune(df, t2, distributable=True).show()


def test_tune_df(tmpdir):
    @tunable()
    def t1(a: int, df: pd.DataFrame, b: int) -> float:
        return float(a + b + df["y"].sum())

    e = make_execution_engine(None, {TUNE_TEMP_PATH: str(tmpdir)})

    for distributable in [True, False, None]:
        with FugueWorkflow(e) as dag:
            s = space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)), batch_size=3)
            t = dag.df([[0, 1], [1, 2], [0, 2]], "x:int,y:int").partition(by=["x"])
            df = serialize_df(t, "df", str(tmpdir)).cross_join(s.broadcast())
            tune(df, t1, distributable=distributable).show()

    for distributable in [True, False, None]:
        with FugueWorkflow(e) as dag:
            df = dag.df([[0, 1], [1, 2], [0, 2]], "x:int,y:int")
            t1.space(a=Grid(0, 1), b=Grid(2, 3), df=df).tune().show()

    @tunable()
    def t2(df1: pd.DataFrame, df2: pd.DataFrame, a: int, b: int) -> Dict[str, Any]:
        return {
            "error": float(a + b + df1["y"].sum() + df2["y"].sum()),
            "metadata": {"a": a},
        }

    with FugueWorkflow(e) as dag:
        df1 = dag.df([[0, 1], [1, 2], [0, 2]], "x:int,y:int").partition(by=["x"])
        df2 = dag.df([[0, 10], [0, 20]], "x:int,y:int").partition(by=["x"])
        t2.space(df1=df1, df2=df2, a=Grid(0, 1), b=Grid(2, 3)).tune().show()


def test_select_best(tmpdir):
    def t1(a: int, b: int) -> float:
        return a + b

    with FugueWorkflow() as dag:
        df = space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)))
        select_best(tune(df, t1, distributable=False)).show()

    @tunable()
    def t2(df1: pd.DataFrame, df2: pd.DataFrame, a: int, b: int) -> Dict[str, Any]:
        return {
            "error": float(a + b + df1["y"].sum() + df2["y"].sum()),
            "metadata": {"a": a},
        }

    e = NativeExecutionEngine(conf={TUNE_TEMP_PATH: str(tmpdir)})
    with FugueWorkflow(e) as dag:
        df1 = dag.df([[0, 1], [1, 2], [0, 2]], "x:int,y:int").partition(by=["x"])
        df2 = dag.df([[0, 10], [1, 20]], "x:int,y:int").partition(by=["x"])
        res = t2.space(df1=df1, df2=df2, a=Grid(0, 1), b=Grid(2, 3)).tune()
        select_best(res, top=2).show()


def test_visualize_top_n(tmpdir):
    def t1(a: int, b: int) -> float:
        return a + b

    with FugueWorkflow() as dag:
        df = space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)))
        visualize_top_n(tune(df, t1, distributable=False), top=2)

    @tunable()
    def t2(df1: pd.DataFrame, df2: pd.DataFrame, a: int, b: int) -> Dict[str, Any]:
        return {
            "error": float(a + b + df1["y"].sum() + df2["y"].sum()),
            "metadata": {"a": a},
        }

    e = NativeExecutionEngine(conf={TUNE_TEMP_PATH: str(tmpdir)})
    with FugueWorkflow(e) as dag:
        df1 = dag.df([[0, 1], [1, 2], [0, 2]], "x:int,y:int").partition(by=["x"])
        df2 = dag.df([[0, 10], [1, 20]], "x:int,y:int").partition(by=["x"])
        res = t2.space(df1=df1, df2=df2, a=Grid(0, 1), b=Grid(2, 3)).tune()
        visualize_top_n(res, top=2)
