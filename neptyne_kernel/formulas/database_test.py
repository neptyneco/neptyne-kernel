import pytest

from ..cell_range import CellRange
from ..spreadsheet_error import NUM_ERROR, VALUE_ERROR
from .database import (
    DAVERAGE,
    DCOUNT,
    DCOUNTA,
    DGET,
    DMAX,
    DMIN,
    DPRODUCT,
    DSTDEV,
    DSTDEVP,
    DSUM,
    DVAR,
    DVARP,
)
from .test_helpers import approx_or_error

DAVERAGE_RANGE = CellRange(
    [
        ["Tree", "Height", "Age", "Yield", "Profit"],
        ["Apple", 18, 20, 14, 105],
        ["Pear", 12, 12, 10, 96],
        ["Cherry", 13, 14, 9, 105],
        ["Apple", 14, 15, 10, 75],
        ["Pear", 9, 8, 8, 76.8],
        ["Apple", 8, 9, 6, 45],
    ]
)

DA_CRITERIA_RANGE = CellRange(
    [
        ["Tree", "Height", "Age", "Yield", "Profit", "Height"],
        ["=Apple", ">10", None, None, None, "<16"],
        ["=Pear", None, None, None, None, None],
    ]
)

DB_OO_RANGE = CellRange(
    [
        ["Name", "Grade", "Age", "Distance to School", "Weight"],
        ["Andy", 3, 9, 150, 40],
        ["Betty", 4, 10, 1000, 42],
        ["Charles", 3, 10, 300, 51],
        ["Daniel", 5, 11, 1200, 48],
        ["Eva", 2, 8, 650, 33],
        ["Frank", 2, 7, 300, 42],
        ["Greta", 1, 7, 200, 36],
        ["Harry", 3, 9, 1200, 44],
        ["Irene", 2, 8, 1000, 42],
    ]
)


@pytest.mark.parametrize(
    "database, field, criteria, result",
    [
        (DAVERAGE_RANGE, "Yield", DA_CRITERIA_RANGE[:2, :2], 12),
        (DAVERAGE_RANGE, 3, DAVERAGE_RANGE, 13),
        (DAVERAGE_RANGE, 15, DAVERAGE_RANGE, VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange([None]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column"]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column", None]), VALUE_ERROR),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 7]), 39),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 8]), 37.5),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 9]), 42),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 10]), 46.5),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 11]), 48),
        (DB_OO_RANGE, "Weight", CellRange(["Age", None]), 42),
    ],
)
def test_DAVERAGE(database, field, criteria, result):
    assert DAVERAGE(database, field, criteria) == result


@pytest.mark.parametrize(
    "database, field, criteria, result",
    [
        (DAVERAGE_RANGE, "Profit", DA_CRITERIA_RANGE, 96),
        (DAVERAGE_RANGE, 15, DAVERAGE_RANGE, VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange([None]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column"]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column", None]), VALUE_ERROR),
        (
            DB_OO_RANGE,
            "Distance to School",
            CellRange(
                [
                    ["Name", "Grade", "Age", "Distance to School", "Weight"],
                    [None, None, None, None, None],
                ]
            ),
            1200,
        ),
        (
            DB_OO_RANGE,
            "Distance to School",
            CellRange(
                [
                    ["Name", "Grade", "Age", "Distance to School", "Weight"],
                    ["", "", "", "", ""],
                ]
            ),
            1200,
        ),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 7]), 42),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 8]), 42),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 9]), 44),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 10]), 51),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 11]), 48),
    ],
)
def test_DMAX(database, field, criteria, result):
    assert DMAX(database, field, criteria) == result


@pytest.mark.parametrize(
    "database, field, criteria, result",
    [
        (DAVERAGE_RANGE, "Profit", DA_CRITERIA_RANGE, 75),
        (DAVERAGE_RANGE, 15, DAVERAGE_RANGE, VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange([None]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column"]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column", None]), VALUE_ERROR),
        (
            DB_OO_RANGE,
            "Distance to School",
            CellRange(
                [
                    ["Name", "Grade", "Age", "Distance to School", "Weight"],
                    [None, None, None, ">0", None],
                ]
            ),
            150,
        ),
        (
            DB_OO_RANGE,
            "Distance to School",
            CellRange(
                [
                    ["Name", "Grade", "Age", "Distance to School", "Weight"],
                    ["", "", "", ">1000", ""],
                ]
            ),
            1200,
        ),
        (DB_OO_RANGE, "Distance to School", CellRange(["Age", 7]), 200),
        (DB_OO_RANGE, "Distance to School", CellRange(["Age", 8]), 650),
        (DB_OO_RANGE, "Distance to School", CellRange(["Age", 9]), 150),
        (DB_OO_RANGE, "Distance to School", CellRange(["Age", 10]), 300),
        (DB_OO_RANGE, "Distance to School", CellRange(["Age", 11]), 1200),
    ],
)
def test_DMIN(database, field, criteria, result):
    assert DMIN(database, field, criteria) == result


@pytest.mark.parametrize(
    "database, field, criteria, result",
    [
        (DAVERAGE_RANGE, "Yield", DA_CRITERIA_RANGE, 800),
        (DAVERAGE_RANGE, 15, DAVERAGE_RANGE, VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange([None]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column"]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column", None]), VALUE_ERROR),
        (DB_OO_RANGE, "Age", CellRange(["Grade", "<>0"]), 279417600),
    ],
)
def test_DPRODUCT(database, field, criteria, result):
    assert DPRODUCT(database, field, criteria) == result


@pytest.mark.parametrize(
    "database, field, criteria, result",
    [
        (DAVERAGE_RANGE, "Profit", DA_CRITERIA_RANGE[:2, :1], 225),
        (DAVERAGE_RANGE, "Profit", DA_CRITERIA_RANGE, 247.8),
        (DAVERAGE_RANGE, 3, CellRange([None]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column"]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column", None]), VALUE_ERROR),
        (
            DB_OO_RANGE,
            "Distance to School",
            CellRange(
                [
                    ["Name", "Grade", "Age", "Distance to School", "Weight"],
                    [None, None, None, ">0", None],
                ]
            ),
            6000,
        ),
        (DB_OO_RANGE, "Distance to School", CellRange(["Grade", 2]), 1950),
        (DB_OO_RANGE, "Weight", CellRange(["Grade", 2]), 117),
    ],
)
def test_DSUM(database, field, criteria, result):
    assert DSUM(database, field, criteria) == result


DCOUNT_RANGE = CellRange(
    [
        ["Tree", "Height", "Age", "Yield", "Profit"],
        ["Apple", 18, 20, 14, 105],
        ["Pear", 12, 12, 10, 96],
        ["Cherry", 13, 14, 9, 105],
        ["Apple", 14, None, 10, 75],
        ["Pear", 9, 8, 8, 76.8],
        ["Apple", 12, 11, 6, 45],
    ]
)


@pytest.mark.parametrize(
    "database, field, criteria, result",
    [
        (DCOUNT_RANGE, "Age", DA_CRITERIA_RANGE[:2, :], 1),
        (DCOUNT_RANGE, "Age", DA_CRITERIA_RANGE, 3),
        (DCOUNT_RANGE, 3, CellRange([None]), VALUE_ERROR),
        (DCOUNT_RANGE, 3, CellRange(["Column"]), VALUE_ERROR),
        (DCOUNT_RANGE, 3, CellRange(["Column", None]), VALUE_ERROR),
        (
            DB_OO_RANGE,
            "Distance to School",
            CellRange(
                [
                    ["Name", "Grade", "Age", "Distance to School", "Weight"],
                    [None, None, None, ">600", None],
                ]
            ),
            5,
        ),
        (
            DB_OO_RANGE,
            "Name",
            CellRange(
                [
                    ["Name", "Grade", "Age", "Distance to School", "Weight"],
                    [None, None, None, ">600", None],
                ]
            ),
            0,
        ),
        (
            DB_OO_RANGE,
            "Distance to School",
            CellRange([["Age", "Grade"], [">7", 2]]),
            2,
        ),
        (DB_OO_RANGE, "Name", CellRange(["Age", "<>0"]), 0),
        (DB_OO_RANGE, "Age", CellRange(["Age", "<>0"]), 9),
        (DB_OO_RANGE[:, 2], "Age", CellRange(["Age", 10]), 2),
    ],
)
def test_DCOUNT(database, field, criteria, result):
    assert DCOUNT(database, field, criteria) == result


@pytest.mark.parametrize(
    "database, field, criteria, result",
    [
        (DB_OO_RANGE, "Name", CellRange(["Name", ">=E"]), 5),
        (DB_OO_RANGE, "Name", CellRange(["Age", "<>0"]), 9),
    ],
)
def test_DCOUNTA(database, field, criteria, result):
    assert DCOUNTA(database, field, criteria) == result


GET_CRITERIA_RANGE = CellRange(
    [
        ["Tree", "Height", "Age", "Yield", "Profit", "Height"],
        ["=Apple", ">10", None, None, None, "<16"],
        ["=Pear", ">12", None, None, None, None],
    ]
)


@pytest.mark.parametrize(
    "database, field, criteria, result",
    [
        (DAVERAGE_RANGE, "Yield", GET_CRITERIA_RANGE[:3, :1], NUM_ERROR),
        (DAVERAGE_RANGE, "Yield", GET_CRITERIA_RANGE, 10),
        (DAVERAGE_RANGE, 3, CellRange([None]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column"]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column", None]), VALUE_ERROR),
        (
            DB_OO_RANGE,
            "Name",
            CellRange(
                [
                    ["Name", "Grade", "Age", "Distance to School", "Weight"],
                    [None, None, 11, None, None],
                ]
            ),
            "Daniel",
        ),
        (
            DB_OO_RANGE,
            "Name",
            CellRange(
                [
                    ["Name", "Grade", "Age", "Distance to School", "Weight"],
                    [None, None, 10, None, None],
                ]
            ),
            NUM_ERROR,
        ),
        (DB_OO_RANGE, "Grade", CellRange(["Name", "Frank"]), 2),
        (DB_OO_RANGE, "Age", CellRange(["Name", "Frank"]), 7),
    ],
)
def test_DGET(database, field, criteria, result):
    assert DGET(database, field, criteria) == result


@pytest.mark.parametrize(
    "database, field, criteria, result",
    [
        (DAVERAGE_RANGE, "Yield", GET_CRITERIA_RANGE[:3, :1], 2.96648),
        (DAVERAGE_RANGE, 3, CellRange([None]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column"]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column", None]), VALUE_ERROR),
        (
            DB_OO_RANGE,
            "Weight",
            CellRange(
                [
                    ["Name", "Grade", "Age", "Distance to School", "Weight"],
                    [None, None, None, None, ">0"],
                ]
            ),
            5.5,
        ),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 7]), 4.24264068711929),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 8]), 6.36396103067893),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 9]), 2.82842712474619),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 10]), 6.36396103067893),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 11]), NUM_ERROR),
    ],
)
def test_DSTDEV(database, field, criteria, result):
    assert DSTDEV(database, field, criteria) == approx_or_error(result)


@pytest.mark.parametrize(
    "database, field, criteria, result",
    [
        (DAVERAGE_RANGE, "Yield", GET_CRITERIA_RANGE[:3, :1], 2.6532998),
        (DAVERAGE_RANGE, 3, CellRange([None]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column"]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column", None]), VALUE_ERROR),
        (
            DB_OO_RANGE,
            "Weight",
            CellRange(
                [
                    ["Name", "Grade", "Age", "Distance to School", "Weight"],
                    [None, None, None, None, ">0"],
                ]
            ),
            5.18544972870135,
        ),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 7]), 3),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 8]), 4.5),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 9]), 2),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 10]), 4.5),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 11]), 0),
    ],
)
def test_DSTDEVP(database, field, criteria, result):
    assert DSTDEVP(database, field, criteria) == approx_or_error(result)


@pytest.mark.parametrize(
    "database, field, criteria, result",
    [
        (DAVERAGE_RANGE, "Yield", GET_CRITERIA_RANGE[:3, :1], 8.8),
        (DAVERAGE_RANGE, 3, CellRange([None]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column"]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column", None]), VALUE_ERROR),
        (
            DB_OO_RANGE,
            "Distance to School",
            CellRange(
                [
                    ["Name", "Grade", "Age", "Distance to School", "Weight"],
                    [None, None, None, ">0", None],
                ]
            ),
            193125,
        ),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 7]), 18),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 8]), 40.5),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 9]), 8),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 10]), 40.5),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 11]), NUM_ERROR),
    ],
)
def test_DVAR(database, field, criteria, result):
    assert DVAR(database, field, criteria) == approx_or_error(result)


@pytest.mark.parametrize(
    "database, field, criteria, result",
    [
        (DAVERAGE_RANGE, "Yield", GET_CRITERIA_RANGE[:3, :1], 7.04),
        (DAVERAGE_RANGE, 3, CellRange([None]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column"]), VALUE_ERROR),
        (DAVERAGE_RANGE, 3, CellRange(["Column", None]), VALUE_ERROR),
        (
            DB_OO_RANGE,
            "Distance to School",
            CellRange(
                [
                    ["Name", "Grade", "Age", "Distance to School", "Weight"],
                    [None, None, None, ">0", None],
                ]
            ),
            171666.6667,
        ),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 7]), 9),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 8]), 20.25),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 9]), 4),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 10]), 20.25),
        (DB_OO_RANGE, "Weight", CellRange(["Age", 11]), 0),
    ],
)
def test_DVARP(database, field, criteria, result):
    assert DVARP(database, field, criteria) == approx_or_error(result)
