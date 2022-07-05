import pandas as pd
from fugue.workflow.workflow import FugueWorkflow
from pytest import raises
from tune import Space, MetricLogger
from tune.api.factory import (
    TuneObjectFactory,
    parse_iterative_objective,
    parse_logger,
    parse_monitor,
    parse_noniterative_local_optimizer,
    parse_noniterative_objective,
    parse_noniterative_stopper,
)
from tune.concepts.dataset import TuneDataset
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


class _Dummy:
    pass


_DUMMY = _Dummy()


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
    assert isinstance(
        parse_noniterative_objective(_nobjective),
        NonIterativeObjectiveFunc,
    )
    with raises(TuneCompileError):
        parse_noniterative_objective("x")

    @parse_noniterative_objective.candidate(lambda obj: isinstance(obj, _Dummy))
    def _converter(obj):
        return to_noniterative_objective(_nobjective)

    assert isinstance(
        parse_noniterative_objective(_DUMMY),
        NonIterativeObjectiveFunc,
    )


def test_monitor():
    class Mock(Monitor):
        pass

    assert isinstance(
        parse_monitor(Mock()),
        Monitor,
    )
    with raises(TuneCompileError):
        parse_monitor("x")

    @parse_monitor.candidate(lambda obj: isinstance(obj, _Dummy))
    def _converter(obj):
        return Mock()

    assert isinstance(
        parse_monitor(_DUMMY),
        Monitor,
    )


def test_noniterative_stopper():
    class Mock(NonIterativeStopper):
        pass

    assert isinstance(
        parse_noniterative_stopper(Mock()),
        NonIterativeStopper,
    )
    with raises(TuneCompileError):
        parse_noniterative_stopper("x")

    @parse_noniterative_stopper.candidate(lambda obj: isinstance(obj, _Dummy))
    def _converter(obj):
        return Mock()

    assert isinstance(
        parse_noniterative_stopper(_DUMMY),
        NonIterativeStopper,
    )


def test_noniterative_local_optimizer_converter():
    assert isinstance(
        parse_noniterative_local_optimizer(OptunaLocalOptimizer(0)),
        NonIterativeObjectiveLocalOptimizer,
    )
    with raises(TuneCompileError):
        parse_noniterative_local_optimizer("x")

    @parse_noniterative_local_optimizer.candidate(lambda obj: isinstance(obj, _Dummy))
    def _converter(obj):
        return OptunaLocalOptimizer(0)

    assert isinstance(
        parse_noniterative_local_optimizer(_DUMMY),
        NonIterativeObjectiveLocalOptimizer,
    )


def test_iterative_objective():
    class Mock(IterativeObjectiveFunc):
        pass

    assert isinstance(
        parse_iterative_objective(Mock()),
        IterativeObjectiveFunc,
    )
    with raises(TuneCompileError):
        parse_iterative_objective("x")

    @parse_iterative_objective.candidate(lambda obj: isinstance(obj, _Dummy))
    def _converter(obj):
        return Mock()

    assert isinstance(
        parse_iterative_objective(_DUMMY),
        IterativeObjectiveFunc,
    )


def test_logger():
    def make_logger():
        return MetricLogger()

    assert isinstance(
        parse_logger(MetricLogger()),
        MetricLogger,
    )
    assert parse_logger(None) is None
    assert parse_logger(make_logger) is make_logger

    with raises(TuneCompileError):
        parse_logger("x")

    @parse_logger.candidate(lambda obj: isinstance(obj, _Dummy))
    def _converter(obj):
        return MetricLogger()

    assert isinstance(
        parse_logger(_DUMMY),
        MetricLogger,
    )
