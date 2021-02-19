from typing import Any, Dict

import pandas as pd
from fugue import FugueWorkflow

from tune import Choice, Grid, Rand, RandInt, Space, tunable
from tune.hyperopt import HyperoptRunner
from tune.tune import space_to_df, tune


def test_run():
    @tunable()
    def func(df: pd.DataFrame, a: float, b: float, c: int) -> Dict[str, Any]:
        return {"error": a * a + b * b + df.shape[0] + c, "metadata": {"d": 1}}

    pdf = pd.DataFrame([[0]], columns=["a"])
    runner = HyperoptRunner(100, seed=3)

    res = runner.run(
        func, dict(df=pdf, b=Rand(-100, 100), a=10, c=Choice(1, -1)), {"a", "b", "c"}
    )
    assert res["error"] < 103.0
    assert res["hp"]["a"] == 10
    assert abs(res["hp"]["b"]) < 3.0
    assert res["hp"]["c"] == -1
    assert len(res) == 3
    assert res["metadata"] == {"d": 1}


def test_wf():
    @tunable()
    def func(a: float, b: float, c: int, d: int) -> float:
        return a * a + b * b + c + d

    with FugueWorkflow() as dag:
        space = space_to_df(
            dag,
            Space(a=Grid(1, 2), b=Rand(-100, 100), c=Choice(1, -1), d=RandInt(0, 3)),
        )
        tune(space, func, objective_runner=HyperoptRunner(100, seed=3)).show()

    with FugueWorkflow() as dag:
        space = space_to_df(dag, Space(a=Grid(1, 2), b=Grid(0, 1), c=1, d=2))
        tune(space, func, objective_runner=HyperoptRunner(100, seed=3)).show()
