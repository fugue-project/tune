from typing import Any, Dict, List, Optional

from tune.concepts.space.parameters import _decode_params, _encode_params


class Trial:
    """The input data collection for running an objective.
    It is immutable.

    :param trial_id: the unique id for a trial
    :param params: parameters for tuning
    :param metadata: metadata for tuning, defaults to None. It is set
      during the construction of :class:`~.tune.concepts.dataset.TuneDataset`
    :param keys: partitions keys of the
      :class:`~.tune.concepts.dataset.TuneDataset`, defaults to None
    :param dfs: dataframes extracted from
      :class:`~.tune.concepts.dataset.TuneDataset`, defaults to None
    :param raw: if True, ``params`` were not encoded, otherwise,
      ``params`` were encoded to json serializable, defaults to False

    .. attention::

        This class is not for users to construct directly.
        Use :class:`~tune.concepts.space.spaces.Space` instead.

    """

    def __init__(
        self,
        trial_id: str,
        params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        keys: Optional[List[str]] = None,
        dfs: Optional[Dict[str, Any]] = None,
        raw: bool = False,
    ):
        self._trial_id = trial_id
        self._params = params if raw else _decode_params(params)
        self._metadata = metadata or {}
        self._keys = keys or []
        self._dfs = dfs or {}

    def copy(self) -> "Trial":
        """Copy the current object.

        :return: the copied object

        .. note::
            This is shallow copy, but it is also used by `__deepcopy__`
            of this object. This is because we disable deepcopy
            of Trial.
        """
        return Trial(
            trial_id=self._trial_id,
            params=self._params,
            metadata=self._metadata,
            keys=self._keys,
            dfs=self._dfs,
            raw=True,
        )

    def __copy__(self) -> "Trial":
        """Shallow copy"""
        return self.copy()

    def __deepcopy__(self, memo: Any) -> "Trial":
        """(Enforced) shallow copy"""
        return self.copy()

    @property
    def trial_id(self) -> str:
        """The unique id of this trial"""
        return self._trial_id

    @property
    def params(self) -> Dict[str, Any]:
        """Parameters for tuning"""
        return self._params

    @property
    def keys(self) -> List[str]:
        """Partitions keys of the
        :class:`~.tune.concepts.dataset.TuneDataset`
        """
        return self._keys

    @property
    def dfs(self) -> Dict[str, Any]:
        """Dataframes extracted from
        :class:`~.tune.concepts.dataset.TuneDataset`
        """
        return self._dfs

    def with_dfs(self, dfs: Dict[str, Any]) -> "Trial":
        """Set dataframes for the trial, a new Trial object will
        be constructed and with the new ``dfs``

        :param dfs: dataframes to attach to the trial

        """
        if len(dfs) == 0 and len(self.dfs) == 0:
            return self
        t = self.copy()
        t._dfs = dfs
        return t

    def with_params(self, params: Dict[str, Any], raw: bool = False) -> "Trial":
        """Set parameters for the trial, a new Trial object will
        be constructed and with the new ``params``

        :param params: parameters for tuning
        :param raw: if True, ``params`` were not encoded, otherwise,
          ``params`` were encoded to json serializable, defaults to False
        """
        t = self.copy()
        t._params = params if raw else _decode_params(params)
        return t

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata of the trial"""
        return self._metadata

    @property
    def jsondict(self) -> Dict[str, Any]:
        """Json serializable python dict of this object"""
        return {
            "trial_id": self.trial_id,
            "params": _encode_params(self.params),
            "metadata": self.metadata,
            "keys": self.keys,
        }

    @staticmethod
    def from_jsondict(data: Dict[str, Any]) -> "Trial":
        """Construct a Trail object from a json serializable python dict

        :param data: the python dict

        .. note::

            This is the counterpart of :meth:`~.jsondict`. They are designed
            for communication purposes.
        """
        return Trial(**data)
