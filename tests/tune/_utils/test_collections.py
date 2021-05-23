from tune._utils import _EMPTY_ITER, dict_product, product, safe_iter


def test_safe_iter():
    assert [1] == list(safe_iter([1]))
    assert [1] == list(safe_iter(safe_iter([1])))
    assert [None] == list(safe_iter(safe_iter([None])))
    assert [1] == list(safe_iter([1], safe=False))

    assert [_EMPTY_ITER] == list(safe_iter([]))
    assert [_EMPTY_ITER] == list(safe_iter(safe_iter([])))
    assert [] == list(safe_iter([], safe=False))


def test_product():
    res = list(product([], safe=False))
    assert 0 == len(res)
    res = list(product([], safe=True))
    assert 1 == len(res)

    res = list(product([[]], safe=False))
    assert 0 == len(res)
    res = list(product([[]], safe=True))
    assert [[]] == res

    res = list(product([[], []]))
    assert 0 == len(res)
    res = list(product([[], []], safe=True))
    assert [[]] == res
    res = list(product([[], []], safe=True, remove_empty=False))
    assert [[None, None]] == res

    for safe in [True, False]:
        for remove_empty in [True, False]:
            res = list(product([[1], [2]], safe=safe, remove_empty=remove_empty))
            assert 1 == len(res)
            assert [1, 2] == res[0]

            res = list(product([[1], [2, 3]], safe=safe, remove_empty=remove_empty))
            assert [[1, 2], [1, 3]] == res

    res = list(product([[1], []]))
    assert [] == res
    res = list(product([[None], [], [2, 3]], safe=True, remove_empty=False))
    assert [[None, None, 2], [None, None, 3]] == res
    res = list(product([[None], [], [2, 3]], safe=True, remove_empty=True))
    assert [[None, 2], [None, 3]] == res

    res = list(product([[1], safe_iter([])], safe=False, remove_empty=True))
    assert [[1]] == res
    res = list(product([[1], safe_iter([])], safe=False, remove_empty=False))
    assert [[1, None]] == res


def test_dict_product():
    res = list(dict_product({}, safe=True))
    assert [{}] == res
    res = list(dict_product({"a": []}, safe=True))
    assert [{}] == res

    res = list(dict_product({}, safe=False))
    assert [] == res
    res = list(dict_product({"a": []}, safe=False))
    assert [] == res
