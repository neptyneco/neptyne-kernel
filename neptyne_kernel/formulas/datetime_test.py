from datetime import date, time

import pytest

from ..cell_range import CellRange
from ..spreadsheet_datetime import (
    SpreadsheetDate,
    SpreadsheetDateTime,
    SpreadsheetTime,
    datetime,
    excel2date,
    excel2datetime,
    timedelta,
)
from ..spreadsheet_error import NUM_ERROR, VALUE_ERROR
from .boolean import FALSE, TRUE
from .date_time import (
    DATE,
    DATEDIF,
    DATEVALUE,
    DAY,
    DAYS,
    DAYS360,
    EDATE,
    EOMONTH,
    HOUR,
    ISOWEEKNUM,
    MINUTE,
    MONTH,
    NETWORKDAYS,
    NOW,
    SECOND,
    TIME,
    TIMEVALUE,
    TODAY,
    WEEKDAY,
    WEEKNUM,
    WORKDAY,
    YEAR,
    YEARFRAC,
)

WORKDAY_INTL_RANGE = CellRange(
    [
        date(2014, 11, 1),
        5,
        DATE(2014, 11, 11),
        datetime(2014, 11, 27),
        datetime(2014, 11, 28),
        DATE(2001, 12, 24),
        DATE(2001, 12, 25),
        DATE(2001, 12, 26),
        DATE(2001, 12, 31),
        DATE(2002, 1, 1),
        17,
        DATE(2001, 12, 1),
        -5,
    ]
)


@pytest.mark.parametrize(
    "value, result",
    [
        (datetime(2020, 1, 1, 14, 35, 0), 14),
        (time(16, 45, 33), 16),
        (0.75, 18),
        (1.2, 4),
        (1.23, 5),
        (-1.5, 12),
        (0, 0),
        ("7:45PM", 19),
        ("02-03-2022", 0),
        (SpreadsheetDateTime("6:10"), 6),
    ],
)
def test_HOUR(value, result):
    assert HOUR(value) == result


@pytest.mark.parametrize(
    "value, result",
    [
        (datetime(2020, 1, 1, 14, 35, 0), 35),
        (time(16, 45, 33), 45),
        (0.2, 48),
        (-1.2, 12),
        (0, 0),
        (-100, 0),
        (0.51, 14),
        ("7:45PM", 45),
        ("02-03-2022", 0),
        (SpreadsheetDateTime("6:10"), 10),
    ],
)
def test_MINUTE(value, result):
    assert MINUTE(value) == result


@pytest.mark.parametrize(
    "value, result",
    [
        (datetime(2020, 1, 1, 14, 35, 0), 0),
        (time(16, 45, 33), 33),
        (0.21, 24),
        (-1.21, 36),
        (0, 0),
        (-100.01, 36),
        (0.51, 24),
        ("4:48:18 PM", 18),
        ("02-03-2022", 0),
        (SpreadsheetDateTime("4:48 PM"), 0),
    ],
)
def test_SECOND(value, result):
    assert SECOND(value) == result


def test_TIME():
    assert TIME(12, 35, 20) == SpreadsheetTime("12:35:20")


def test_NOW():
    assert excel2datetime(NOW()) - datetime.now(
        tz=SpreadsheetDateTime.TZ_INFO
    ) < timedelta(seconds=1)


def test_TODAY():
    assert excel2date(TODAY()) - datetime.now(
        tz=SpreadsheetDate.TZ_INFO
    ).date() < timedelta(seconds=1)


@pytest.mark.parametrize(
    "date_text, value",
    [("8/22/2011", 40777), ("22-MAY-2011", 40685), ("2011/02/23", 40597)],
)
def test_DATEVALUE(date_text, value):
    assert DATEVALUE(date_text) == value


@pytest.mark.parametrize(
    "time_text, value", [("2:24 AM", 0.1), ("22-Aug-2011 6:35 AM", 0.2743)]
)
def test_TIMEVALUE(time_text, value):
    assert abs(TIMEVALUE(time_text) - value) < 0.01


@pytest.mark.parametrize(
    "start_date, end_date, unit, result",
    [
        (date(1969, 7, 16), date(1969, 7, 24), "D", 8),
        (datetime(1969, 7, 16, 23, 59, 59), date(1969, 7, 24), "D", 8),
        ("1969/07/16", "1969/07/24", "D", 8),
        ("1969/05/16", "1970/07/24", "YD", 69),
        ("1969/05/16", "1970/07/24", "MD", 8),
        ("2019/11/30", "2019/12/29", "M", 0),
        ("2019/11/30", "2019/12/30", "M", 1),
        ("2020/12/29", "2019/11/30", "M", NUM_ERROR),
        ("2001/1/1", "2003/1/1", "Y", 2),
        ("2001/6/1", "2002/8/15", "D", 440),
        ("2001/6/1", "2002/8/15", "YD", 75),
        ("2001/6/1", "2020/8/15", "YD", 75),
        ("2001/6/1", "2001/8/15", "YD", 75),
        ("2001/6/1", "2001/8/15", "YM", 2),
        ("2001/6/1", "2020/8/15", "YM", 2),
        (44566, "2022/1/7", "D", 2),
    ],
)
def test_DATEDIF(start_date, end_date, unit, result):
    assert DATEDIF(start_date, end_date, unit) == result


@pytest.mark.parametrize(
    "year, month, day, result",
    [
        (2012, 3, 14, 40982),
        (2008, -3, 2, 39327),
        (2008, 1, -15, 39432),
        (2008, 1, 35, 39482),
        (2008, 14, 2, 39846),
        (-2008, 14, 2, NUM_ERROR),
    ],
)
def test_DATE(year, month, day, result):
    assert DATE(year, month, day) == result


@pytest.mark.parametrize(
    "value, result",
    [
        (44238, 11),
        ("2021/02/11", 11),
        (DATE(1969, 7, 20), 20),
        (datetime(1569, 7, 13), 13),
        (date(2024, 1, 21), 21),
        (40909.34, 1),
    ],
)
def test_DAY(value, result):
    assert DAY(value) == result


@pytest.mark.parametrize(
    "end_date, start_date, result",
    [
        (37000.33, 35950.5, 1050),
        (datetime(2021, 3, 15), date(2021, 2, 1), 42),
        (DATE(2021, 3, 15), DATE(2021, 2, 1), 42),
    ],
)
def test_DAYS(end_date, start_date, result):
    assert DAYS(end_date, start_date) == result


@pytest.mark.parametrize(
    "start_date, end_date, method, result",
    [
        (35950.5, 37000.33, FALSE, 1035),
        (datetime(2021, 2, 1), date(2021, 3, 15), FALSE, 44),
        (DATE(2021, 2, 1), DATE(2021, 3, 15), FALSE, 44),
        (DATE(2020, 1, 1), DATE(2021, 1, 31), FALSE, 390),
        (DATE(2020, 1, 1), DATE(2021, 1, 31), TRUE, 389),
    ],
)
def test_DAYS360(start_date, end_date, method, result):
    assert DAYS360(start_date, end_date, method) == result


@pytest.mark.parametrize(
    "start_date, months, result",
    [
        (datetime(2011, 1, 11), 1, DATE(2011, 2, 11)),
        (DATE(2011, 1, 11), 1, DATE(2011, 2, 11)),
        (DATE(2011, 1, 15), -1, DATE(2010, 12, 15)),
        (DATE(2011, 1, 11), 2, DATE(2011, 3, 11)),
        (DATE(2020, 11, 30), 6, 44346),
        (44242, -3, 44150),
        (DATE(2020, 5, 31), -3, 43890),
        ("date", -3, VALUE_ERROR),
    ],
)
def test_EDATE(start_date, months, result):
    assert EDATE(start_date, months) == result


@pytest.mark.parametrize(
    "start_date, months, result",
    [
        (datetime(2011, 1, 11), 1, 40602),
        (DATE(2011, 1, 11), 1, 40602),
        (DATE(2011, 1, 15), -1, 40543),
        (DATE(2011, 1, 11), 2, 40633),
        (DATE(2020, 11, 30), 6, 44347),
        (44242, -3, 44165),
        (DATE(2020, 5, 31), -3, 43890),
        ("date", -3, VALUE_ERROR),
    ],
)
def test_EOMONTH(start_date, months, result):
    assert EOMONTH(start_date, months) == result


@pytest.mark.parametrize(
    "source_date, result",
    [
        (date(2012, 3, 9), 10),
        (datetime(2012, 3, 9), 10),
        (DATE(2012, 3, 9), 10),
        (DATE(2021, 2, 17), 7),
        (DATE(1995, 1, 1), 52),
        (DATE(1999, 1, 1), 53),
        (DATE(2012, 5, 31), 22),
    ],
)
def test_ISOWEEKNUM(source_date, result):
    assert ISOWEEKNUM(source_date) == result


@pytest.mark.parametrize(
    "source_date, result",
    [
        (date(2012, 3, 9), 3),
        (datetime(2012, 3, 9), 3),
        (DATE(2012, 3, 9), 3),
        (DATE(2021, 2, 17), 2),
        (DATE(1995, 1, 1), 1),
        (DATE(1999, 1, 1), 1),
        (DATE(2012, 5, 31), 5),
        (44238, 2),
    ],
)
def test_MONTH(source_date, result):
    assert MONTH(source_date) == result


@pytest.mark.parametrize(
    "start_date, end_date, holidays, result",
    [
        (datetime(2012, 1, 10), date(2013, 3, 1), None, 299),
        (DATE(2012, 1, 10), DATE(2013, 3, 1), None, 299),
        (
            DATE(2020, 1, 1),
            DATE(2020, 12, 31),
            CellRange(
                [
                    DATE(2020, 1, 1),
                    DATE(2020, 4, 10),
                    DATE(2020, 4, 13),
                    DATE(2020, 5, 8),
                    DATE(2020, 5, 25),
                    DATE(2020, 8, 31),
                    DATE(2020, 12, 25),
                    DATE(2020, 12, 28),
                ]
            ),
            254,
        ),
        (43831, 43861, CellRange([43845, 43852]), 12),
        (
            DATE(2012, 1, 10),
            DATE(2013, 3, 1),
            CellRange([DATE(2012, 11, 22), DATE(2012, 12, 4), DATE(2013, 1, 21)]),
            107,
        ),
    ],
)
def test_NETWORKDAYS(start_date, end_date, holidays, result):
    NETWORKDAYS(start_date, end_date, holidays) == result


@pytest.mark.parametrize(
    "start_date, end_date, weekend, holidays, result",
    [
        (
            date(2006, 1, 1),
            datetime(2006, 2, 1),
            7,
            CellRange([datetime(2006, 1, 2), DATE(2006, 1, 16)]),
            22,
        ),
        (
            DATE(2006, 1, 1),
            DATE(2006, 2, 1),
            7,
            CellRange([DATE(2006, 1, 2), DATE(2006, 1, 16)]),
            22,
        ),
        (
            DATE(2006, 1, 1),
            DATE(2006, 2, 1),
            "0010001",
            CellRange([DATE(2006, 1, 2), DATE(2006, 1, 16)]),
            20,
        ),
        (DATE(2020, 1, 1), DATE(2020, 12, 31), "0000000111", None, VALUE_ERROR),
        # Copied from OpenOffice tests
        # https://github.com/LibreOffice/core/blob/0ad94d52c022a0acb20be21b5a1dfcf445e12f0c/sc/qa/unit/data/functions/date_time/fods/networkdays.intl.fods
        (
            DATE(2014, 11, 1),
            DATE(2014, 11, 30),
            1,
            CellRange([DATE(2014, 11, 11), DATE(2014, 11, 27), DATE(2014, 11, 28)]),
            17,
        ),
        (DATE(2014, 11, 1), DATE(2014, 11, 1), 1, DATE(2014, 11, 1), 0),
        (DATE(2014, 11, 1), DATE(2014, 11, 30), 1, None, 20),
        (DATE(2014, 11, 30), DATE(2014, 11, 1), 1, None, -20),
        (
            DATE(2014, 11, 1),
            DATE(2014, 11, 30),
            1,
            CellRange([DATE(2014, 11, 11), DATE(2014, 11, 28), DATE(2014, 11, 27)]),
            17,
        ),
        (
            DATE(2014, 11, 1),
            DATE(2014, 11, 30),
            2,
            CellRange([DATE(2014, 11, 11), DATE(2014, 11, 28), DATE(2014, 11, 27)]),
            18,
        ),
        (
            DATE(2014, 11, 1),
            DATE(2014, 11, 30),
            3,
            CellRange([DATE(2014, 11, 11), DATE(2014, 11, 28), DATE(2014, 11, 27)]),
            20,
        ),
        (
            DATE(2014, 11, 1),
            DATE(2014, 11, 30),
            4,
            CellRange([DATE(2014, 11, 11), DATE(2014, 11, 28), DATE(2014, 11, 27)]),
            20,
        ),
        (
            DATE(2014, 11, 1),
            DATE(2014, 11, 30),
            5,
            CellRange([DATE(2014, 11, 11), DATE(2014, 11, 28), DATE(2014, 11, 27)]),
            20,
        ),
        (
            DATE(2014, 11, 1),
            DATE(2014, 11, 30),
            6,
            CellRange([DATE(2014, 11, 11), DATE(2014, 11, 28), DATE(2014, 11, 27)]),
            21,
        ),
        (
            DATE(2014, 11, 1),
            DATE(2014, 11, 30),
            7,
            CellRange([DATE(2014, 11, 11), DATE(2014, 11, 28), DATE(2014, 11, 27)]),
            19,
        ),
        (
            DATE(2014, 11, 1),
            DATE(2014, 11, 30),
            11,
            CellRange([DATE(2014, 11, 11), DATE(2014, 11, 28), DATE(2014, 11, 27)]),
            22,
        ),
        (
            DATE(2014, 11, 1),
            DATE(2014, 11, 30),
            12,
            CellRange([DATE(2014, 11, 11), DATE(2014, 11, 28), DATE(2014, 11, 27)]),
            23,
        ),
        (
            DATE(2014, 11, 1),
            DATE(2014, 11, 30),
            13,
            CellRange([DATE(2014, 11, 11), DATE(2014, 11, 28), DATE(2014, 11, 27)]),
            24,
        ),
        (
            DATE(2014, 11, 1),
            DATE(2014, 11, 30),
            14,
            CellRange([DATE(2014, 11, 11), DATE(2014, 11, 28), DATE(2014, 11, 27)]),
            23,
        ),
        (
            DATE(2014, 11, 1),
            DATE(2014, 11, 30),
            15,
            CellRange([DATE(2014, 11, 11), DATE(2014, 11, 28), DATE(2014, 11, 27)]),
            24,
        ),
        (
            DATE(2014, 11, 1),
            DATE(2014, 11, 30),
            16,
            CellRange([DATE(2014, 11, 11), DATE(2014, 11, 28), DATE(2014, 11, 27)]),
            24,
        ),
        (
            DATE(2014, 11, 1),
            DATE(2014, 11, 30),
            17,
            CellRange([DATE(2014, 11, 11), DATE(2014, 11, 28), DATE(2014, 11, 27)]),
            22,
        ),
        (
            DATE(2006, 1, 1),
            DATE(2006, 2, 1),
            "0000110",
            CellRange([DATE(2006, 1, 2), DATE(2006, 1, 16)]),
            22,
        ),
        (
            DATE(2006, 1, 1),
            DATE(2006, 2, 1),
            "1111111",
            CellRange([DATE(2006, 1, 2), DATE(2006, 1, 16)]),
            0,
        ),
    ],
)
def test_NETWORKDAYS_INTL(start_date, end_date, weekend, holidays, result):
    assert NETWORKDAYS.INTL(start_date, end_date, weekend, holidays) == result


@pytest.mark.parametrize(
    "source_date, return_type, result",
    [
        (date(2008, 2, 14), 1, 5),
        (datetime(2008, 2, 14), 1, 5),
        (DATE(2008, 2, 14), 1, 5),
        (DATE(2008, 2, 14), 2, 4),
        (DATE(2008, 2, 14), 3, 3),
        (DATE(2021, 2, 24), 1, 4),
        (44251, 3, 2),
        (DATE(2021, 2, 24), 14, 7),
        (DATE(2021, 2, 24), 22, NUM_ERROR),
    ],
)
def test_WEEKDAY(source_date, return_type, result):
    assert WEEKDAY(source_date, return_type) == result


@pytest.mark.parametrize(
    "source_date, return_type, result",
    [
        (datetime(2012, 3, 9), 1, 10),
        (date(2012, 3, 9), 1, 10),
        (DATE(2012, 3, 9), 1, 10),
        (DATE(2012, 3, 9), 2, 11),
        (DATE(2021, 2, 24), 22, NUM_ERROR),
        (DATE(2021, 1, 1), 1, 1),
        (DATE(2021, 1, 3), 1, 2),
        (DATE(2021, 1, 1), 21, 53),
        (DATE(2021, 1, 4), 21, 1),
        (44251, 13, 9),
    ],
)
def test_WEEKNUM(source_date, return_type, result):
    assert WEEKNUM(source_date, return_type) == result


@pytest.mark.parametrize(
    "start_date, days, holidays, result",
    [
        (date(2021, 2, 10), 10, None, 44251),
        (DATE(2021, 2, 10), 10, None, 44251),
        (44256, -5, None, 44249),
        (
            DATE(2020, 1, 1),
            254,
            CellRange(
                [
                    DATE(2020, 1, 1),
                    DATE(2020, 4, 10),
                    DATE(2020, 4, 13),
                    DATE(2020, 5, 8),
                    DATE(2020, 5, 25),
                    DATE(2020, 8, 31),
                    DATE(2020, 12, 25),
                    DATE(2020, 12, 28),
                ]
            ),
            44196,
        ),
        (DATE(2020, 1, 1), -25, None, 43796),
    ],
)
def test_WORKDAY(start_date, days, holidays, result):
    assert WORKDAY(start_date, days, holidays) == result


@pytest.mark.parametrize(
    "start_date, days, weekend, holidays, result",
    [
        (datetime(2012, 1, 1), 30, 0, None, NUM_ERROR),
        (DATE(2012, 1, 1), 30, 0, None, NUM_ERROR),
        (DATE(2012, 1, 1), 90, 11, None, 41013),
        (
            DATE(2020, 1, 1),
            254,
            1,
            CellRange(
                [
                    DATE(2020, 1, 1),
                    DATE(2020, 4, 10),
                    DATE(2020, 4, 13),
                    DATE(2020, 5, 8),
                    DATE(2020, 5, 25),
                    DATE(2020, 8, 31),
                    DATE(2020, 12, 25),
                    DATE(2020, 12, 28),
                ]
            ),
            DATE(2020, 12, 31),
        ),
        (
            datetime(2020, 1, 1),
            254,
            1,
            CellRange(
                [
                    date(2020, 1, 1),
                    datetime(2020, 4, 10),
                    DATE(2020, 4, 13),
                    date(2020, 5, 8),
                    datetime(2020, 5, 25),
                    DATE(2020, 8, 31),
                    date(2020, 12, 25),
                    datetime(2020, 12, 28),
                ]
            ),
            DATE(2020, 12, 31),
        ),
        (DATE(2021, 2, 10), 10, 11, None, DATE(2021, 2, 22)),
        # Copied from OpenOffice tests
        # https://github.com/LibreOffice/core/blob/0ad94d52c022a0acb20be21b5a1dfcf445e12f0c/sc/qa/unit/data/functions/date_time/fods/workday.intl.fods
        (
            WORKDAY_INTL_RANGE[0],
            WORKDAY_INTL_RANGE[1],
            1,
            WORKDAY_INTL_RANGE[2:4],
            41950,
        ),
        (WORKDAY_INTL_RANGE[0], WORKDAY_INTL_RANGE[1], 1, WORKDAY_INTL_RANGE[0], 41950),
        (
            DATE(2014, 11, 1),
            2,
            5,
            [DATE(2014, 11, 2), DATE(2014, 11, 3), DATE(2014, 11, 4)],
            41951,
        ),
        (WORKDAY_INTL_RANGE[0], WORKDAY_INTL_RANGE[1], 2, WORKDAY_INTL_RANGE[0], 41951),
        (WORKDAY_INTL_RANGE[0], WORKDAY_INTL_RANGE[1], 3, WORKDAY_INTL_RANGE[0], 41951),
        (WORKDAY_INTL_RANGE[0], WORKDAY_INTL_RANGE[1], 4, WORKDAY_INTL_RANGE[0], 41951),
        (WORKDAY_INTL_RANGE[0], WORKDAY_INTL_RANGE[1], 5, WORKDAY_INTL_RANGE[0], 41951),
        (WORKDAY_INTL_RANGE[0], WORKDAY_INTL_RANGE[1], 6, WORKDAY_INTL_RANGE[0], 41951),
        (WORKDAY_INTL_RANGE[0], WORKDAY_INTL_RANGE[1], 7, WORKDAY_INTL_RANGE[0], 41949),
        (
            WORKDAY_INTL_RANGE[0],
            WORKDAY_INTL_RANGE[1],
            11,
            WORKDAY_INTL_RANGE[0],
            41950,
        ),
        (
            WORKDAY_INTL_RANGE[0],
            WORKDAY_INTL_RANGE[1],
            12,
            WORKDAY_INTL_RANGE[0],
            41950,
        ),
        (
            WORKDAY_INTL_RANGE[0],
            WORKDAY_INTL_RANGE[1],
            13,
            WORKDAY_INTL_RANGE[0],
            41950,
        ),
        (
            WORKDAY_INTL_RANGE[0],
            WORKDAY_INTL_RANGE[1],
            14,
            WORKDAY_INTL_RANGE[0],
            41950,
        ),
        (
            WORKDAY_INTL_RANGE[0],
            WORKDAY_INTL_RANGE[1],
            15,
            WORKDAY_INTL_RANGE[0],
            41950,
        ),
        (
            WORKDAY_INTL_RANGE[0],
            WORKDAY_INTL_RANGE[1],
            16,
            WORKDAY_INTL_RANGE[0],
            41949,
        ),
        (
            WORKDAY_INTL_RANGE[0],
            WORKDAY_INTL_RANGE[1],
            17,
            WORKDAY_INTL_RANGE[0],
            41949,
        ),
        (
            DATE(2014, 11, 1),
            -2,
            "1100000",
            [DATE(2014, 11, 2), DATE(2014, 11, 3), DATE(2014, 11, 4)],
            41942,
        ),
        (
            DATE(2014, 11, 1),
            -2,
            "1111111",
            [DATE(2014, 11, 2), DATE(2014, 11, 3), DATE(2014, 11, 4)],
            VALUE_ERROR,
        ),
        (
            DATE(2014, 11, 1),
            -8,
            "0011001",
            [DATE(2014, 10, 20), DATE(2014, 10, 23), DATE(2014, 10, 24)],
            41926,
        ),
    ],
)
def test_WORKDAY_INTL(start_date, days, weekend, holidays, result):
    assert WORKDAY.INTL(start_date, days, weekend, holidays) == result


@pytest.mark.parametrize(
    "source_date, result",
    [
        (datetime(2023, 3, 1), 2023),
        (date(2020, 1, 11), 2020),
        (DATE(2021, 3, 1), 2021),
        (33333.33, 1991),
    ],
)
def test_YEAR(source_date, result):
    assert YEAR(source_date) == result


@pytest.mark.parametrize(
    "start_date, end_date, basis, result",
    [
        (datetime(2012, 1, 1), date(2012, 7, 30), 0, 0.58055556),
        (DATE(2012, 1, 1), DATE(2012, 7, 30), 0, 0.58055556),
        (DATE(2012, 1, 1), DATE(2012, 7, 30), 1, 0.57650273),
        (DATE(2012, 1, 1), DATE(2012, 7, 30), 3, 0.57808219),
    ],
)
def test_YEARFRAC(start_date, end_date, basis, result):
    assert YEARFRAC(start_date, end_date, basis) == pytest.approx(result, rel=1e-3)
