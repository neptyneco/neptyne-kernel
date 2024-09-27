# ruff: noqa: F405
import pytest

from ..spreadsheet_error import NA_ERROR, NULL_ERROR
from .boolean import *  # noqa: F403


def test_IF():
    assert IF(3 > 2, "ja", "nee") == "ja"


@pytest.mark.parametrize("cond", [3 > 2, 1 == 1, 2 < 3])
def test_NOT(cond):
    assert NOT(cond) != cond


def test_AND_true():
    assert AND(1 == 1, 2 > 0)


@pytest.mark.parametrize(
    "cond1, cond2",
    [
        (3 < 2, 1 == 0),  # False AND False
        (3 == 3, 2 != 2),  # True AND False
        (2 > 3, 1 < 2),  # False AND True
    ],
)
def test_AND_false(cond1, cond2):
    assert not AND(cond1, cond2)


@pytest.mark.parametrize(
    "cond1, cond2",
    [
        (4 > 2, 1 < 0),  # True OR False
        (10 == 1, 5 == 5),  # False OR True
        (5 < 6, 1 > 2),  # True OR True
    ],
)
def test_OR_true(cond1, cond2):
    assert OR(cond1, cond2)


def test_OR_false():
    assert not OR(3 < 2, 5 != 5)


@pytest.mark.parametrize(
    "cond1, cond2",
    [
        (5 < 6, 1 > 2),  # True XOR False
        (4 == 2, 3 > 2),  # False XOR True
    ],
)
def test_XOR_true(cond1, cond2):
    assert XOR(cond1, cond2)


@pytest.mark.parametrize(
    "cond1, cond2",
    [
        (4 > 2, 1 > 0),  # both are True
        (10 == 1, 5 < 5),  # both are False
    ],
)
def test_XOR_false(cond1, cond2):
    assert not XOR(cond1, cond2)


def test_TRUE():
    assert TRUE()
    assert TRUE


def test_FALSE():
    assert not FALSE()
    assert not FALSE


def test_IFS():
    assert IFS(4 < 3, "no", 1 == 3, "no", 2 + 2 == 4, "yes") == "yes"
    assert IFS(1 > 10, "no", True, "yes") == "yes"
    assert IFS(1 > 10, "no", 2 > 3, "no") == NA_ERROR
    assert IFS(1 > 10, "no", True) == NA_ERROR


def test_IFNA():
    assert IFNA(IFS(1 > 10, "no", 2 > 3, "no"), "yes") == "yes"
    assert IFNA(2 + 3, "no") == 5


def test_IFERROR():
    assert IFERROR(NULL_ERROR, "yes") == "yes"
    assert IFERROR("yes", "no") == "yes"
