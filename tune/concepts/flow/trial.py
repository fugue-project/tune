from typing import Any, Dict, List, Optional

from tune.concepts.space.parameters import decode_params, encode_params


class Trial:
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
        self._params = params if raw else decode_params(params)
        self._metadata = metadata or {}
        self._keys = keys or []
        self._dfs = dfs or {}

    def copy(self) -> "Trial":
        return Trial(
            trial_id=self._trial_id,
            params=self._params,
            metadata=self._metadata,
            keys=self._keys,
            dfs=self._dfs,
            raw=True,
        )

    def __copy__(self) -> "Trial":
        return self.copy()

    def __deepcopy__(self, memo: Any) -> "Trial":
        return self.copy()

    @property
    def trial_id(self) -> str:
        return self._trial_id

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    @property
    def keys(self) -> List[str]:
        return self._keys

    @property
    def dfs(self) -> Dict[str, Any]:
        return self._dfs

    def with_dfs(self, dfs: Dict[str, Any]) -> "Trial":
        if len(dfs) == 0 and len(self.dfs) == 0:
            return self
        t = self.copy()
        t._dfs = dfs
        return t

    def with_params(self, params: Dict[str, Any], raw: bool = False) -> "Trial":
        t = self.copy()
        t._params = params if raw else decode_params(params)
        return t

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def jsondict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "params": encode_params(self.params),
            "metadata": self.metadata,
            "keys": self.keys,
        }

    @staticmethod
    def from_jsondict(data: Dict[str, Any]) -> "Trial":
        return Trial(**data)
