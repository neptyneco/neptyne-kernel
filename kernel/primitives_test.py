import pytest

from .primitives import Empty, NeptyneInt, NeptyneStr


def test_empty_math():
    e = Empty(None)

    assert e + 1 == 1
    assert 1 + e == 1

    assert e - 1 == -1
    assert 1 - e == 1

    assert e * 1 == 0
    assert 1 * e == 0

    assert e / 1 == 0
    with pytest.raises(ZeroDivisionError):
        1 / e


def test_string_concat():
    e = Empty(None)
    assert e & "a" == "a"
    assert "a" & e == "a"

    i = NeptyneInt(1, None)
    assert i & "a" == "1a"

    s = NeptyneStr("a", None)
    assert "b" & s == "ba"
