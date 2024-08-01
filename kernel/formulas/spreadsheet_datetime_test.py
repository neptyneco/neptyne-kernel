from datetime import date, datetime, time, timedelta

import pytest

from .spreadsheet_datetime import (
    EPOCH_FIRST_YEAR,
    SpreadsheetDate,
    SpreadsheetDateTime,
    SpreadsheetTime,
    excel2date,
    excel2datetime,
    excel2time,
)
from .spreadsheet_error import VALUE_ERROR


class TestSpreadsheetDateTime:
    def test_init(self):
        assert excel2datetime(SpreadsheetDateTime()) - datetime.now(
            tz=SpreadsheetDateTime.TZ_INFO
        ) < timedelta(seconds=1)
        assert excel2datetime(SpreadsheetDateTime("01/05/2022 20:22:55")) == datetime(
            2022, 1, 5, 20, 22, 55, tzinfo=SpreadsheetDateTime.TZ_INFO
        )

        with pytest.raises(ValueError):
            SpreadsheetDateTime("date")

    @pytest.mark.parametrize(
        "d1, d2, result",
        [
            (
                SpreadsheetDateTime("05/01/2022 20:22:10"),
                1,
                SpreadsheetDateTime("04/30/2022 20:22:10"),
            ),
            (
                SpreadsheetDateTime("05/01/2022 20:22:10"),
                0.5,
                SpreadsheetDateTime("05/01/2022 08:22:10"),
            ),
            (
                SpreadsheetDateTime("05/02/2022 20:22:10"),
                SpreadsheetDateTime("05/01/2022 20:22:10"),
                1,
            ),
        ],
    )
    def test_sub(self, d1, d2, result):
        assert d1 - d2 == result

    @pytest.mark.parametrize(
        "d1, d2, result",
        [
            (
                SpreadsheetDateTime("05/01/2022 20:22:10"),
                1,
                SpreadsheetDateTime("05/02/2022 20:22:10"),
            ),
            (
                SpreadsheetDateTime("05/01/2022 20:22:10"),
                0.5,
                SpreadsheetDateTime("05/02/2022 08:22:10"),
            ),
        ],
    )
    def test_add(self, d1, d2, result):
        assert d1 + d2 == result


class TestTime:
    def test_init(self):
        assert excel2time(SpreadsheetTime("20:22:55")) == time(20, 22, 55)
        assert excel2time(
            SpreadsheetTime(datetime(EPOCH_FIRST_YEAR, 1, 1, 20, 22, 55))
        ) == time(20, 22, 55)

        assert excel2time(SpreadsheetTime("20:22:55")) == time(20, 22, 55)
        with pytest.raises(ValueError):
            SpreadsheetTime("time")

    @pytest.mark.parametrize(
        "left, right, result",
        [
            (SpreadsheetTime("20:22:10"), 0.1, SpreadsheetTime("17:58:10")),
            (SpreadsheetTime("20:22:10"), 0.5, SpreadsheetTime("8:22:10")),
        ],
    )
    def test_sub(self, left, right, result):
        assert left - right == pytest.approx(result, 1e-3)

    def test_sub_error(self):
        with pytest.raises(ValueError):
            SpreadsheetTime("time")

    @pytest.mark.parametrize(
        "left, right, result",
        [
            (SpreadsheetTime("20:22:10"), 0.1, SpreadsheetTime("22:46:10")),
            (SpreadsheetTime("20:22:15"), 0.15, SpreadsheetTime("23:58:10")),
        ],
    )
    def test_add(self, left, right, result):
        assert left + right == pytest.approx(result, 1e-3)

    def test_add_error(self):
        assert SpreadsheetTime("20:22:15") + "5 sec" == VALUE_ERROR


class TestDate:
    def test_init(self):
        assert excel2date(SpreadsheetDate("01/04/2022")) == date(2022, 1, 4)
        assert excel2date(SpreadsheetDate(datetime(2022, 5, 1))) == date(2022, 5, 1)
        assert excel2date(SpreadsheetDate(44566)) == date(2022, 1, 5)
        with pytest.raises(ValueError):
            SpreadsheetDate("time")

    @pytest.mark.parametrize(
        "left, right, result",
        [
            (SpreadsheetDate("07/10/2021"), 1, SpreadsheetDate("07/09/2021")),
            (SpreadsheetDate("05/01/2022"), 0.5, SpreadsheetDate("04/30/2022")),
        ],
    )
    def test_sub(self, left, right, result):
        assert left - right == pytest.approx(result, 1e-3)

    def test_sub_error(self):
        with pytest.raises(ValueError):
            SpreadsheetDate("date")

    @pytest.mark.parametrize(
        "left, right, result",
        [
            (SpreadsheetDate("07/10/2021"), 1, SpreadsheetDate("07/11/2021")),
            (SpreadsheetDate("07/10/2021"), 0.5, SpreadsheetDate("07/10/2021")),
        ],
    )
    def test_add(self, left, right, result):
        assert left + right == pytest.approx(result, 1e-3)

    def test_add_error(self):
        assert SpreadsheetDate("07/10/2021") + "2 days" == VALUE_ERROR
