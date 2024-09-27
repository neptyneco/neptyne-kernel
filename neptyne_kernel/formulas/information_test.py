import pytest

from ..cell_range import CellRange
from ..spreadsheet_datetime import SpreadsheetDate
from ..spreadsheet_error import NA_ERROR, NULL_ERROR, REF_ERROR, VALUE_ERROR
from .boolean import FALSE, TRUE
from .information import (
    CELL,
    ERROR,
    ISBLANK,
    ISERR,
    ISERROR,
    ISEVEN,
    ISLOGICAL,
    ISNA,
    ISNONTEXT,
    ISNUMBER,
    ISODD,
    ISTEXT,
    NA,
    TYPE,
    N,
)


@pytest.mark.parametrize(
    "value, result",
    [
        ("", FALSE),
        (None, TRUE),
        (1, FALSE),
        ("hello", FALSE),
        (VALUE_ERROR, FALSE),
    ],
)
def test_ISBLANK(value, result):
    assert ISBLANK(value) == result


@pytest.mark.parametrize(
    "value, result",
    [
        ("", FALSE),
        (NA_ERROR, FALSE),
        (VALUE_ERROR, TRUE),
    ],
)
def test_ISERR(value, result):
    assert ISERR(value) == result


@pytest.mark.parametrize(
    "value, result",
    [
        ("", FALSE),
        (NA_ERROR, TRUE),
        (VALUE_ERROR, TRUE),
    ],
)
def test_ISERROR(value, result):
    assert ISERROR(value) == result


@pytest.mark.parametrize(
    "value, result",
    [
        ("", VALUE_ERROR),
        (2, TRUE),
        (3, FALSE),
        (2.5, TRUE),
        (-1, FALSE),
        (0, TRUE),
        (VALUE_ERROR, VALUE_ERROR),
    ],
)
def test_ISEVEN(value, result):
    assert ISEVEN(value) == result


@pytest.mark.parametrize(
    "error_val, result",
    [(NULL_ERROR, 1), (555, NA_ERROR)],
)
def test_ERROR_TYPE(error_val, result):
    assert ERROR.TYPE(error_val) == result


@pytest.mark.parametrize(
    "value, result",
    [
        (FALSE, TRUE),
        (100 > 50, TRUE),
        (2 + 2, FALSE),
        (1, FALSE),
        (0, FALSE),
    ],
)
def test_ISLOGICAL(value, result):
    assert ISLOGICAL(value) == result


@pytest.mark.parametrize(
    "value, result",
    [
        (NA_ERROR, TRUE),
        (100 > 50, FALSE),
        (1, FALSE),
        (VALUE_ERROR, FALSE),
    ],
)
def test_ISNA(value, result):
    assert ISNA(value) == result


def test_NA():
    assert NA() == NA_ERROR


@pytest.mark.parametrize(
    "value, result",
    [
        (10, 1),
        (SpreadsheetDate(12345), 1),
        ("apple", 2),
        (None, 1),
        ("", 1),
        (FALSE, 4),
        (TRUE, 4),
        (REF_ERROR, 16),
        (VALUE_ERROR, 16),
        (CellRange([1, 2, 3]), 64),
    ],
)
def test_TYPE(value, result):
    assert TYPE(value) == result


@pytest.mark.parametrize(
    "value, result",
    [
        (7, 7),
        ("Even", 0),
        (TRUE, 1),
        (FALSE, 0),
        (SpreadsheetDate(12345), 12345),
        ("7", 0),
    ],
)
def test_N(value, result):
    assert N(value) == result


@pytest.mark.parametrize(
    "value, result",
    [
        ("", VALUE_ERROR),
        (3, TRUE),
        (2, FALSE),
        (2.5, FALSE),
        (-1, TRUE),
        (0, FALSE),
        (VALUE_ERROR, VALUE_ERROR),
    ],
)
def test_ISODD(value, result):
    assert ISODD(value) == result


@pytest.mark.parametrize(
    "value, result",
    [
        ("", TRUE),
        ("apple", TRUE),
        (2, FALSE),
        (VALUE_ERROR, FALSE),
    ],
)
def test_ISTEXT(value, result):
    assert ISTEXT(value) == result


@pytest.mark.parametrize(
    "value, result",
    [
        ("", FALSE),
        ("apple", FALSE),
        (2, TRUE),
        (VALUE_ERROR, TRUE),
        (None, TRUE),
    ],
)
def test_ISNONTEXT(value, result):
    assert ISNONTEXT(value) == result


@pytest.mark.parametrize(
    "value, result",
    [
        ("", FALSE),
        ("apple", FALSE),
        (2, TRUE),
        (-1.5, TRUE),
        (VALUE_ERROR, FALSE),
        (None, FALSE),
    ],
)
def test_ISNUMBER(value, result):
    assert ISNUMBER(value) == result


CELL_TEST = CellRange([["hello", "world"], [5, 10]])


@pytest.mark.parametrize(
    "info_type, reference, result",
    [
        ("contents", CELL_TEST, "hello"),
        ("contents", CELL_TEST[0][0], "hello"),
        ("contents", CELL_TEST[0], "hello"),
        ("contents", None, 0),
    ],
)
def test_CELL(info_type, reference, result):
    assert CELL(info_type, reference) == result
