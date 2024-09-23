import sys

import pytest

from .neptyne_protocol import MIMETypes
from .primitives import NeptyneStr
from .spreadsheet_error import (
    NAME_ERROR,
    PYTHON_ERROR,
    VALUE_ERROR,
    ZERO_DIV_ERROR,
    SpreadsheetError,
)


def test_mime_bundle():
    err = SpreadsheetError("name", "msg")
    bundle = err._repr_mimebundle_()

    assert (
        bundle[MIMETypes.APPLICATION_VND_NEPTYNE_ERROR_V1_JSON.value]["ename"] == "name"
    )


def test_equality():
    assert PYTHON_ERROR == PYTHON_ERROR.with_message("That didn't work")


@pytest.mark.parametrize(
    "exp, error",
    [
        (lambda: NeptyneStr("aaa", None) ** 3, VALUE_ERROR),
        (lambda: NeptyneStr("aaa", None) / 3, VALUE_ERROR),
        (lambda: 1 / 0, ZERO_DIV_ERROR),
        (lambda: hello_world, NAME_ERROR),  # noqa: F821
    ],
)
def test_parse_unsupported_type_exception(exp, error):
    try:
        exp()
    except Exception:
        etype, evalue, tb = sys.exc_info()
        se = SpreadsheetError.from_python_exception(etype, evalue, [])
        assert se == error


def test_line_number():
    err = SpreadsheetError.from_python_exception(
        ZeroDivisionError("Division by zerro"),
        "division by zero",
        [
            "\x1b[0;31m---------------------------------------------------------------------------\x1b[0m",
            "\x1b[0;31mZeroDivisionError\x1b[0m                         Traceback (most recent call last)",
            "\x1b[0;31mZeroDivisionError\x1b[0m: division by zero",
        ],
    )
    assert err.line_number is None
