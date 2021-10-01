from typing import Any, Dict, List, Tuple

from tune.concepts.space.parameters import TuningParameterExpression
from copy import deepcopy


class TuningParametersTemplate:
    """Parameter template to extract tuning parameter expressions from
    nested data structure

    :param raw: the dictionary of input parameters.

    .. code-block:: python

        # common cases
        TuningParametersTemplate(dict(a=1, b=1))
        TuningParametersTemplate(dict(a=Rand(0, 1), b=1))

        # expressions may not be nested in dicts or arrays
        template = TuningParametersTemplate(
            dict(a=dict(x1=Rand(0, 1), x2=Rand(3,4)), b=[Grid("a", "b")]))

        assert [Rand(0, 1), Rand(3, 4), Grid("a", "b")] == template.params
        assert dict(
            p0=Rand(0, 1), p1=Rand(3, 4), p2=Grid("a", "b")
        ) == template.params_dict
        assert dict(a=1, x2=3), b=["a"]) == template.fill([1, 3, "a"])
        assert dict(a=1, x2=3), b=["a"]) == template.fill_dict(
            dict(p2="a", p1=3, p0=1)
        )

    .. note::

        Users commonly don't need to use this class directly

    """

    def __init__(self, raw: Dict[str, Any]):
        self._map: List[Tuple[Any, Any, TuningParameterExpression]] = []
        self._template = self._copy(raw)

    @property
    def params(self) -> List[TuningParameterExpression]:
        """Get all tuning parameter expressions in depth-first order"""
        return [x[2] for x in self._map]

    @property
    def params_dict(self) -> Dict[str, TuningParameterExpression]:
        """Get all tuning parameter expressions in depth-first order,
        with correspondent made-up new keys p0, p1, p2, ...
        """
        return {f"p{i}": x[2] for i, x in enumerate(self._map)}

    def fill(self, params: List[Any], copy: bool = False) -> Dict[str, Any]:
        """Fill the original data structure with values

        :param params: the list of values to be filled into the original
          data structure, in depth-first order
        :param copy: whether to return a deeply copied paramters,
          defaults to False
        :return: the original data structure filled with values
        """
        i = 0
        for p, k, _ in self._map:
            p[k] = params[i]
            i += 1
        return deepcopy(self._template) if copy else self._template

    def fill_dict(self, params: Dict[str, Any], copy: bool = False) -> Dict[str, Any]:
        """Fill the original data structure with dictionary of values

        :param params: the dictionary of values to be filled into the original
          data structure, keys must be p0, p1, p2, ...
        :param copy: whether to return a deeply copied paramters,
          defaults to False
        :return: the original data structure filled with values
        """
        temp = [params[f"p{i}"] for i in range(len(params))]
        return self.fill(temp, copy)

    def _copy(self, src: Any) -> Any:
        if isinstance(src, dict):
            ddest: Dict[str, Any] = {}
            for k, v in src.items():
                if isinstance(v, TuningParameterExpression):
                    ddest[k] = None
                    self._map.append((ddest, k, v))
                else:
                    ddest[k] = self._copy(v)
            return ddest
        elif isinstance(src, list):
            adest: List[Any] = []
            for i in range(len(src)):
                if isinstance(src[i], TuningParameterExpression):
                    adest.append(None)
                    self._map.append((adest, i, src[i]))
                else:
                    adest.append(self._copy(src[i]))
            return adest
        else:
            return src
