from tune.concepts.dataset import TuneDataset
from fugue.workflow.workflow import FugueWorkflow
from pytest import raises
from tune.api.factory import TuneObjectFactory
from tune.concepts.flow.judge import Monitor
from tune.exceptions import TuneCompileError
from tune.iterative.objective import IterativeObjectiveFunc
from tune.noniterative.convert import to_noniterative_objective
from tune.noniterative.objective import (
    NonIterativeObjectiveFunc,
    NonIterativeObjectiveLocalOptimizer,
)
from tune.noniterative.stopper import NonIterativeStopper

from tune_optuna.optimizer import OptunaLocalOptimizer
import pandas as pd
from tune import Space


def _nobjective(a: int) -> float:
    return 0.0


def test_dataset(tmpdir):
    factory = TuneObjectFactory()
    factory.set_temp_path(str(tmpdir))
    df = pd.DataFrame([[0]], columns=["a"])
    dag = FugueWorkflow()
    data = factory.make_dataset(dag, Space(a=1, b=1), df=df)
    assert isinstance(data, TuneDataset)
    assert factory.make_dataset(dag, data) is data
    # TODO: the unit test is not complete, but it's covered by other functions
    with raises(TuneCompileError):
        factory.make_dataset(dag, 1)


def test_temp_path():
    factory = TuneObjectFactory()
    with raises(TuneCompileError):
        factory.get_path_or_temp("")  # must set temp path
    assert "/x" == factory.get_path_or_temp("/x")

    factory.set_temp_path("/tmp")
    assert "/x" == factory.get_path_or_temp("/x")
    assert "/tmp" == factory.get_path_or_temp("")


def test_noniterative_objective():
    factory = TuneObjectFactory()
    assert isinstance(
        factory.make_noniterative_objective(_nobjective),
        NonIterativeObjectiveFunc,
    )
    with raises(TuneCompileError):
        factory.make_noniterative_objective("x")

    factory.set_noniterative_objective_converter(
        lambda x: to_noniterative_objective(_nobjective)
    )
    assert isinstance(
        factory.make_noniterative_objective("x"),
        NonIterativeObjectiveFunc,
    )


def test_monitor():
    factory = TuneObjectFactory()

    class Mock(Monitor):
        pass

    assert isinstance(
        factory.make_monitor(Mock()),
        Monitor,
    )
    with raises(TuneCompileError):
        factory.make_monitor("x")

    factory.set_monitor_converter(lambda x: Mock())
    assert isinstance(
        factory.make_monitor("x"),
        Monitor,
    )


def test_noniterative_stopper():
    factory = TuneObjectFactory()

    class Mock(NonIterativeStopper):
        pass

    assert isinstance(
        factory.make_noniterative_stopper(Mock()),
        NonIterativeStopper,
    )
    with raises(TuneCompileError):
        factory.make_noniterative_stopper("x")

    factory.set_noniterative_stopper_converter(lambda x: Mock())
    assert isinstance(
        factory.make_noniterative_stopper("x"),
        NonIterativeStopper,
    )


def test_noniterative_local_optimizer_converter():
    factory = TuneObjectFactory()
    assert isinstance(
        factory.make_noniterative_local_optimizer(OptunaLocalOptimizer(0)),
        NonIterativeObjectiveLocalOptimizer,
    )
    with raises(TuneCompileError):
        factory.make_noniterative_local_optimizer("x")

    factory.set_noniterative_local_optimizer_converter(
        lambda x: OptunaLocalOptimizer(0)
    )
    assert isinstance(
        factory.make_noniterative_local_optimizer("x"),
        NonIterativeObjectiveLocalOptimizer,
    )


def test_iterative_objective():
    factory = TuneObjectFactory()

    class Mock(IterativeObjectiveFunc):
        pass

    assert isinstance(
        factory.make_iterative_objective(Mock()),
        IterativeObjectiveFunc,
    )
    with raises(TuneCompileError):
        factory.make_iterative_objective("x")

    factory.set_iterative_objective_converter(lambda x: Mock())
    assert isinstance(
        factory.make_iterative_objective("x"),
        IterativeObjectiveFunc,
    )
