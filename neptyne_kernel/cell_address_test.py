import pytest

from .cell_address import Address, Range


@pytest.mark.parametrize(
    ("a1", "expected"),
    [
        ("A1:B2", (0, 1, 0, 1)),
        ("B2:A1", (0, 1, 0, 1)),
        ("A2:B1", (0, 1, 0, 1)),
        ("B1:A2", (0, 1, 0, 1)),
    ],
)
def test_parse_range(a1, expected):
    assert Range.from_a1(a1) == Range(*expected, sheet=0)


@pytest.mark.parametrize(
    ("outer", "inner", "result"),
    [
        (Range.from_a1("A1:B2"), Range.from_a1("A1:B2"), True),
        (Range(0, 1, 0, 1, 1), Range.from_a1("A1:B2"), False),
        (Range.from_a1("A1:B2"), Address.from_a1("B2"), True),
        (Range.from_a1("A1:B2"), Address.from_a1("B3"), False),
        (Range(0, 1, 0, -1, 0), Address.from_a1("B2"), True),
        (Range(0, 1, 0, -1, 0), Range.from_a1("A1:B2"), True),
        (Range(0, 1, 1, -1, 0), Range.from_a1("A1:B2"), False),
        (Range(0, 1, 0, -1, 0), Range(0, 1, 1, -1, 0), True),
        (Range(0, -1, 0, -1, 0), Range(0, -1, 1, -1, 0), True),
        (Range(0, 1, 0, 50, 0), Range(0, 1, 1, -1, 0), False),
    ],
)
def test_contains(outer, inner, result):
    assert (inner in outer) == result


@pytest.mark.parametrize(
    ("r1", "r2", "result"),
    [
        (Range.from_a1("A1:B2"), Range.from_a1("A2:B3"), True),
        (Range.from_a1("A3:B5"), Range.from_a1("A2:B7"), True),
        (Range.from_a1("A3:B5"), Range.from_a1("A5:B7"), True),
        (Range.from_a1("A3:B5"), Range.from_a1("A6:B7"), False),
        (Range(0, -1, 0, -1, 1), Range(0, -1, 0, -1, 1), True),
        (Range(0, -1, 0, -1, 1), Range(0, -1, 0, -1, 0), False),
        (Range(5, 10, 0, -1, 1), Range(10, 20, 0, -1, 1), True),
        (Range(5, 10, 0, -1, 1), Range(11, 20, 0, -1, 1), False),
    ],
)
def test_intersects(r1, r2, result):
    assert r1.intersects(r2) == result
    assert r2.intersects(r1) == result
