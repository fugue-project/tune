import numpy as np
from scipy import stats
from tune._utils import (
    assert_close,
    normal_to_continuous,
    normal_to_discrete,
    normal_to_integers,
    uniform_to_choice,
    uniform_to_continuous,
    uniform_to_discrete,
    uniform_to_integers,
)


def test_uniform_to_continuous():
    assert 10 == uniform_to_continuous(0.5, 10, 10)
    assert 10 == uniform_to_continuous(0.5, 10, 9)
    assert 12.5 == uniform_to_continuous(0.5, 10, 15)
    assert 10 == uniform_to_continuous(0.0, 10, 15)
    assert 15 == uniform_to_continuous(1.0, 10, 15)  # impossible case

    np.random.seed(0)
    values = np.random.uniform(0, 1.0, 100000)
    res = uniform_to_continuous(values, 10, 15)
    assert (res >= 10).all() and (res < 15).all()
    t = stats.kstest(res, "uniform", args=(10, 5))
    assert t.pvalue > 0.4

    start, end = 1000, 500000
    res = uniform_to_continuous(values, start, end, log=True)
    assert (res >= start).all() and (res < end).all()
    t = stats.kstest(
        np.log(res), "uniform", args=(np.log(start), np.log(end) - np.log(start))
    )
    assert t.pvalue > 0.4
    t = stats.kstest(res, "uniform", args=(start, end - start))
    assert t.pvalue < 1e-3  # not uniform distribution

    res = uniform_to_continuous(values, start, end, log=True, base=10)
    assert (res >= start).all() and (res < end).all()
    b = np.log(10)
    t = stats.kstest(
        np.log(res) / b,
        "uniform",
        args=(np.log(start) / b, np.log(end) / b - np.log(start) / b),
    )
    assert t.pvalue > 0.4
    t = stats.kstest(res, "uniform", args=(start, end - start))
    assert t.pvalue < 1e-3  # not uniform distribution


def test_uniform_to_discrete():
    assert 10 == uniform_to_discrete(0.5, 10, 10, q=1.1)

    np.random.seed(0)
    values = np.random.uniform(0, 1.0, 1000)
    res = uniform_to_discrete(values, 1, 3, q=0.5)
    assert_close(res, [1.0, 1.5, 2.0, 2.5, 3.0])
    assert sum(1 if v == 3.0 else 0 for v in res) >= 180  # close to 20%
    res = uniform_to_discrete(values, 1, 3, q=0.5, include_high=False)
    assert_close(res, [1.0, 1.5, 2.0, 2.5])
    assert sum(1 if v == 2.5 else 0 for v in res) >= 230  # close to 25%
    for ih in [True, False]:
        res = uniform_to_discrete(values, 1, 3, q=0.6, include_high=ih)
        assert_close(res, [1.0, 1.6, 2.2, 2.8])
        assert sum(1 if v == 2.8 else 0 for v in res) >= 230  # close to 25%


def test_uniform_to_integers():
    assert 10 == uniform_to_integers(0.5, 10, 10)

    np.random.seed(0)
    values = np.random.uniform(0, 1.0, 1000)
    res = uniform_to_integers(values, 1, 5)
    assert all(isinstance(x, int) for x in res)
    assert set(res) == set([1, 2, 3, 4, 5])
    assert sum(1 if v == 5 else 0 for v in res) >= 180  # close to 25%
    res = uniform_to_integers(values, 1, 5, include_high=False)
    assert all(isinstance(x, int) for x in res)
    assert set(res) == set([1, 2, 3, 4])
    assert sum(1 if v == 4 else 0 for v in res) >= 230  # close to 25%

    res = uniform_to_integers(values, 1, 5, q=2)
    assert set(res) == set([1, 3, 5])
    assert sum(1 if v == 5 else 0 for v in res) >= 300  # close to 33%
    res = uniform_to_integers(values, 1, 5, q=2, include_high=False)
    assert set(res) == set([1, 3])
    assert sum(1 if v == 3 else 0 for v in res) >= 480  # close to 50%

    for ih in [True, False]:
        res = uniform_to_integers(values, 1, 5, q=3, include_high=ih)
        assert set(res) == set([1, 4])
        assert sum(1 if v == 4 else 0 for v in res) >= 480  # close to 33%


def test_uniform_to_choice():
    assert "a" == uniform_to_choice(0.5, ["a"])

    np.random.seed(0)
    values = np.random.uniform(0, 1.0, 1000)
    res = uniform_to_choice(values, ["a", "b", "c", "d"])
    for c in ["a", "b", "c", "d"]:
        assert sum(1 if v == c else 0 for v in res) >= 230  # close to 25%


def test_normal_to_continuous():
    np.random.seed(0)
    values = np.random.normal(0, 1.0, 100000)
    res = normal_to_continuous(values, 10, 15)
    t = stats.kstest(res, "norm", args=(10, 15))
    assert t.pvalue > 0.4


def test_normal_to_discrete():
    np.random.seed(0)
    values = np.random.normal(0, 1.0, 100000)
    res = normal_to_discrete(values, 0.3, 3.0, 0.0001)
    t = stats.kstest(res, "norm", args=(0.3, 3.0))
    assert t.pvalue > 0.4

    res = normal_to_discrete(values, 0.1, 0.9, q=0.3)
    res = [x for x in res if x >= -0.9 and x <= 0.9]
    assert_close([-0.8, -0.5, -0.2, 0.1, 0.4, 0.7], res)


def test_normal_to_integers():
    assert 17 == normal_to_integers(0, 17, 19)

    np.random.seed(0)
    values = np.random.normal(0, 1.0, 1000)

    res = normal_to_integers(values, 17, 19)
    assert 17 in res

    res = normal_to_integers(values, 17, 19, q=3)
    res = [x for x in res if x >= 10 and x <= 26]
    assert_close([11, 14, 17, 20, 23, 26], res)
