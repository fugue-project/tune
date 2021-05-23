from time import sleep

from pytest import raises
from triad.utils.convert import to_timedelta
from tune._utils import run_monitored_process
from tune.exceptions import TuneInterrupted


def test_run_monitored_process():
    # happy case
    assert 10 == run_monitored_process(t1, [1], {}, lambda: True, "5sec")
    # stop before responding
    with raises(TuneInterrupted):
        run_monitored_process(t1, [1], dict(wait="20sec"), lambda: True, "0.2sec")
    # non stop before responding
    assert 10 == run_monitored_process(
        t1, [1], dict(wait="1sec"), lambda: False, "0.2sec"
    )
    with raises(NotImplementedError):
        run_monitored_process(t2, [], {}, lambda: True, "5sec")
    assert run_monitored_process(t3, [], {}, lambda: True, "5sec") is None


def t1(a, wait=None):
    if wait is not None:
        sleep(to_timedelta(wait).total_seconds())
    return a * 10


def t2():
    raise NotImplementedError


def t3():
    pass
