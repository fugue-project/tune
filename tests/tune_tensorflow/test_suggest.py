from pytest import raises
from tune import TUNE_OBJECT_FACTORY, RandInt, Space
from tune.exceptions import TuneCompileError

from tune_tensorflow.suggest import (
    suggest_keras_models_by_continuous_asha,
    suggest_keras_models_by_hyperband,
    suggest_keras_models_by_sha,
)

from .mock import MockSpec


def test_sha(tmpdir):
    TUNE_OBJECT_FACTORY.set_temp_path(str(tmpdir))

    space = Space(l1=RandInt(8, 16), l2=RandInt(8, 24))
    with raises(TuneCompileError):
        suggest_keras_models_by_sha(MockSpec, space, plan=[(2.0, 4), (4.0, 2)])

    space = space.sample(6, 0)
    reports = suggest_keras_models_by_sha(
        MockSpec, space, plan=[(2.0, 4), (4.0, 2)], top_n=2
    )
    for r in reports:
        print(r.jsondict)
    assert 2 == len(reports)


def test_hyperband(tmpdir):
    TUNE_OBJECT_FACTORY.set_temp_path(str(tmpdir))

    space = Space(l1=RandInt(8, 16), l2=RandInt(8, 24))
    with raises(TuneCompileError):
        suggest_keras_models_by_hyperband(
            MockSpec,
            space,
            plans=[
                [(2.0, 4), (4.0, 2)],
                [(4.0, 2), (2.0, 4)],
            ],
        )

    space = space.sample(10, 0)
    reports = suggest_keras_models_by_hyperband(
        MockSpec,
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

    space = Space(l1=RandInt(8, 16), l2=RandInt(8, 24))
    with raises(TuneCompileError):
        suggest_keras_models_by_continuous_asha(
            MockSpec, space, plan=[(2.0, 4), (4.0, 2)]
        )

    space = space.sample(6, 0)
    reports = suggest_keras_models_by_continuous_asha(
        MockSpec, space, plan=[(2.0, 4), (4.0, 2)], top_n=2
    )
    for r in reports:
        print(r.jsondict)
    assert 2 == len(reports)
