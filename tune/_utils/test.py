from typing import Iterable, Any, List


def assert_close(seq1: Iterable[Any], seq2: Iterable[Any], error: float = 1e-5):
    def dedup(seq: List[Any]) -> Iterable[Any]:
        last: Any = None
        for x in sorted(seq):
            if last is None:
                last = x
                yield x
            elif abs(x - last) < error:
                continue
            else:
                last = x
                yield x

    _s1, _s2 = list(dedup(seq1)), list(dedup(seq2))  # type: ignore
    assert len(_s1) == len(_s2), f"{_s1},{_s2}"
    for x, y in zip(_s1, _s2):
        assert abs(x - y) < error, f"{_s1},{_s2}"
