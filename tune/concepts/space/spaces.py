from typing import Any, Iterable

import numpy as np
from triad import assert_or_throw
from tune._utils import product
from tune.concepts.space.parameters import TuningParametersTemplate


class Space:
    """Search space object

    .. important::
        Please read |SpaceTutorial|.

    :param kwargs: parameters in the search space

    .. code-block:: python

        Space(a=1, b=1)  # static space
        Space(a=1, b=Grid(1,2), c=Grid("a", "b"))  # grid search
        Space(a=1, b=Grid(1,2), c=Rand(0, 1))  # grid search + level 2 search
        Space(a=1, b=Grid(1,2), c=Rand(0, 1)).sample(10, sedd=0)  # grid + random search

        # union
        Space(a=1, b=Grid(2,3)) + Space(b=Rand(1,5)).sample(10)

        # cross product
        Space(a=1, b=Grid(2,3)) * Space(c=Rand(1,5), d=Grid("a","b"))

        # combo (grid + random + level 2)
        space1 = Space(a=1, b=Grid(2,4))
        space2 = Space(b=RandInt(10, 20))
        space3 = Space(c=Rand(0,1)).sample(10)
        space = (space1 + space2) * space3

    .. code-block:: python

        assert Space(a=1, b=Rand(0,1)).has_stochastic
        assert not Space(a=1, b=Rand(0,1)).sample(10).has_stochastic
        assert not Space(a=1, b=Grid(0,1)).has_stochastic
        assert not Space(a=1, b=1).has_stochastic

        # get all configurations
        space = Space(a=Grid(2,4), b=Rand(0,1)).sample(100)
        for conf in space:
            print(conf)
        all_conf = list(space)

    """

    def __init__(self, *args, **kwargs: Any):
        if len(args) > 0:
            assert_or_throw(
                len(args) == 1 and len(kwargs) == 0,
                ValueError(
                    "when the first argument is a template or dict, "
                    "it must be the only argument of the constructor"
                ),
            )
            if isinstance(args[0], dict):
                self._templates = [TuningParametersTemplate(args[0])]
            elif isinstance(args[0], TuningParametersTemplate):
                self._templates = [args[0]]
            elif isinstance(args[0], Iterable):
                self._templates = list(args[0])
                assert_or_throw(
                    all(
                        isinstance(x, TuningParametersTemplate) for x in self._templates
                    ),
                    ValueError("not a list of templates"),
                )
            else:
                raise ValueError("invalid argument type " + str(type(args[0])))
        else:
            self._templates = [TuningParametersTemplate(kwargs)]

    def __iter__(self) -> Iterable[TuningParametersTemplate]:
        for template in self._templates:
            yield from template.product_grid()

    @property
    def has_stochastic(self):
        """Whether the space contains any
        :class:`~tune.concepts.space.parameters.StochasticExpression`
        """
        return any(x.has_stochastic for x in self._templates)

    def sample(self, n: int, seed: Any = None) -> "Space":
        """Draw random samples from the current space.
        Please read |SpaceTutorial|.

        :param n: number of samples to draw
        :param seed: random seed, defaults to None
        :return: a new Space containing all samples

        .. note::

            * it only applies to
              :class:`~tune.concepts.space.parameters.StochasticExpression`
            * if :meth:`~.has_stochastic` is False, then it will return
              the original space
            * After sampling, no
              :class:`~tune.concepts.space.parameters.StochasticExpression`
              will exist in the new space.

        """
        if n <= 0 or not self.has_stochastic:
            return self

        def get_templates() -> Iterable[TuningParametersTemplate]:
            if seed is not None:
                np.random.seed(seed)
            for template in self._templates:
                yield from template.sample(n)

        return Space(get_templates())

    def __mul__(self, other: Any) -> "Space":
        """Operator ``*``"""

        def get_templates() -> Iterable[TuningParametersTemplate]:
            o = other if isinstance(other, Space) else Space(other)

            for lr in product([self, o], safe=True, remove_empty=True):  # type: ignore
                yield lr[0].concat(lr[1])

        return Space(get_templates())

    def __add__(self, other: Any) -> "Space":
        """Operator ``+``"""

        def get_templates() -> Iterable[TuningParametersTemplate]:
            o = other if isinstance(other, Space) else Space(other)
            yield from self  # type: ignore
            yield from o  # type: ignore

        return Space(get_templates())

    def __radd__(self, other: Any) -> "Space":
        """Operator ``+``, this is to make compatible with ``sum``"""
        assert_or_throw(
            other is None or (isinstance(other, int) and other == 0), ValueError(other)
        )
        return self
