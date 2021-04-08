from pytest import raises
from tune import (
    TUNE_OBJECT_FACTORY,
    RandInt,
    Space,
    suggest_by_continuous_asha,
    suggest_by_hyperband,
    suggest_by_sha,
)
from tune.exceptions import TuneCompileError

from tune_tensorflow import (
    keras_space,
    suggest_keras_models_by_continuous_asha,
    suggest_keras_models_by_hyperband,
    suggest_keras_models_by_sha,
)

from .mock import MockSpec


def test_sha(tmpdir):
    TUNE_OBJECT_FACTORY.set_temp_path(str(tmpdir))

    space = keras_space(MockSpec, l1=RandInt(8, 16), l2=RandInt(8, 24))
    with raises(TuneCompileError):
        suggest_keras_models_by_sha(space, plan=[(2.0, 4), (4.0, 2)])

    space = space.sample(6, 0)
    reports = suggest_keras_models_by_sha(space, plan=[(2.0, 4), (4.0, 2)], top_n=2)
    for r in reports:
        print(r.jsondict)
    assert 2 == len(reports)


def test_hyperband(tmpdir):
    TUNE_OBJECT_FACTORY.set_temp_path(str(tmpdir))

    space = keras_space(MockSpec, l1=RandInt(8, 16), l2=RandInt(8, 24))
    with raises(TuneCompileError):
        suggest_keras_models_by_hyperband(
            space,
            plans=[
                [(2.0, 4), (4.0, 2)],
                [(4.0, 2), (2.0, 4)],
            ],
        )

    space = space.sample(10, 0)
    reports = suggest_keras_models_by_hyperband(
        space,
        plans=[
            [(2.0, 4), (4.0, 2)],
            [(4.0, 2), (2.0, 4)],
        ],
        top_n=2,
    )
    for r in reports:
        print(r.jsondict)
    assert 2 == len(reports)


def test_asha(tmpdir):
    TUNE_OBJECT_FACTORY.set_temp_path(str(tmpdir))

    space = keras_space(MockSpec, l1=RandInt(8, 16), l2=RandInt(8, 24))
    with raises(TuneCompileError):
        suggest_keras_models_by_continuous_asha(space, plan=[(2.0, 4), (4.0, 2)])

    space = space.sample(6, 0)
    reports = suggest_keras_models_by_continuous_asha(
        space, plan=[(2.0, 4), (4.0, 2)], top_n=2
    )
    for r in reports:
        print(r.jsondict)
    assert 2 == len(reports)
