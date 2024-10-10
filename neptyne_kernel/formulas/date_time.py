from datetime import date, datetime, time
from typing import Iterable

from calweek import weeknum
from dateutil.relativedelta import relativedelta
from numpy import busday_count, busday_offset

from ..cell_range import CellRange
from ..spreadsheet_datetime import (
    EPOCH_FIRST_YEAR,
    SEC_IN_HOUR,
    SpreadsheetDate,
    SpreadsheetDateTime,
    SpreadsheetTime,
    timedelta,
)
from ..spreadsheet_error import NUM_ERROR, VALUE_ERROR, SpreadsheetError
from .boolean import FALSE, BooleanValue
from .date_time_helpers import (
    _DAY_COUNT,
    DateTimeValue,
    DateValue,
    Method360Us,
    calc_year_frac,
    change_month,
    convert_args_to_pydatetime,
    date_diff360eu,
    date_diff360us,
)

__all__ = [
    "DATE",
    "DATEDIF",
    "DATEVALUE",
    "DAY",
    "DAYS",
    "DAYS360",
    "EDATE",
    "EOMONTH",
    "HOUR",
    "ISOWEEKNUM",
    "MINUTE",
    "MONTH",
    "NETWORKDAYS",
    "NOW",
    "SECOND",
    "TIME",
    "TIMEVALUE",
    "TODAY",
    "WEEKDAY",
    "WEEKNUM",
    "WORKDAY",
    "YEAR",
    "YEARFRAC",
]


_WEEKMASKS = {
    1: "1111100",  # Saturday, Sunday
    2: "0111110",  # Sunday, Monday
    3: "0011111",  # Monday, Tuesday
    4: "1001111",  # Tuesday, Wednesday
    5: "1100111",  # Wednesday, Thursday
    6: "1110011",  # Thursday, Friday
    7: "1111001",  # Friday, Saturday
    11: "1111110",  # Sunday only
    12: "0111111",  # Monday only
    13: "1011111",  # Tuesday only
    14: "1101111",  # Wednesday only
    15: "1110111",  # Thursday only
    16: "1111011",  # Friday only
    17: "1111101",  # Saturday only
}


def _prepare_weekmask(weekend: int | str) -> str | SpreadsheetError:
    if isinstance(weekend, str):
        if len(weekend) != 7 or not all(char in "01" for char in weekend):
            return VALUE_ERROR
        return weekend.translate(str.maketrans("01", "10"))
    elif isinstance(weekend, int):
        weekmask = _WEEKMASKS.get(weekend)
        if not weekmask:
            return NUM_ERROR
        return weekmask
    return VALUE_ERROR


def _is_datetime(value: str) -> bool:
    """Check if value can be parsed into date/time."""
    try:
        SpreadsheetDateTime(value)
        return True
    except ValueError:
        return False


def NOW() -> SpreadsheetDateTime:
    """Returns the current date and time."""
    return SpreadsheetDateTime(datetime.now(tz=SpreadsheetDateTime.TZ_INFO))


def TIME(hour: int, minute: int, second: int) -> SpreadsheetTime:
    """Returns the current time."""
    return SpreadsheetTime(
        datetime(
            EPOCH_FIRST_YEAR,
            1,
            1,
            hour,
            minute,
            second,
            tzinfo=SpreadsheetTime.TZ_INFO,
        )
    )


def HOUR(value: DateTimeValue) -> int:
    """Converts a serial number to an hour."""
    if isinstance(value, datetime | time):
        return value.hour
    if isinstance(value, str):
        value = SpreadsheetDateTime(value)
    return timedelta(days=value).seconds // SEC_IN_HOUR


def MINUTE(value: DateTimeValue) -> int:
    """Converts a serial number to a minute."""
    if isinstance(value, datetime | time):
        return value.minute
    if isinstance(value, str):
        value = SpreadsheetDateTime(value)
    return timedelta(days=value).seconds // 60 % 60


def SECOND(value: DateTimeValue) -> int:
    """Converts a serial number to a second."""
    if isinstance(value, datetime | time):
        return value.second
    if isinstance(value, str):
        value = SpreadsheetDateTime(value)
    return timedelta(days=value).seconds % 60


def TODAY() -> SpreadsheetDate:
    """Returns the today's date"""
    return SpreadsheetDate(datetime.now(tz=SpreadsheetDate.TZ_INFO))


def DATEVALUE(date_text: str) -> int:
    """Converts a date in the form of text to a serial number"""
    return SpreadsheetDate(date_text)


def TIMEVALUE(time_text: str) -> float:
    """Converts a time in the form of text to a serial number"""
    return SpreadsheetTime(time_text)


@convert_args_to_pydatetime([0, 1])
def DATEDIF(start_date: DateValue, end_date: DateValue, unit: str) -> int:
    """Calculates the number of days, months, or years between two dates."""
    if start_date > end_date:
        return NUM_ERROR

    if unit == "Y":
        return relativedelta(end_date, start_date).years
    elif unit == "M":
        rel_delta = relativedelta(end_date, start_date)
        return rel_delta.months + rel_delta.years * 12
    elif unit == "D":
        return (end_date - start_date).days
    elif unit == "MD":
        return relativedelta(end_date, start_date).days
    elif unit == "YM":
        return relativedelta(end_date, start_date).months
    elif unit == "YD":
        rel_delta = relativedelta(end_date, start_date)
        if rel_delta.years:
            this_year = end_date - relativedelta(years=rel_delta.years)
            return (this_year - start_date).days
        return (end_date - start_date).days


def DATE(year: int, month: int, day: int) -> int:
    """Returns the serial number of a particular date"""
    if year < 0 or year >= 10000:
        return NUM_ERROR
    if 0 <= year <= 1899:
        year += EPOCH_FIRST_YEAR
    if month > 12 or month < 1:
        year += month // 12
        month %= 12
    dt = date(year, month, 1) + timedelta(days=day - 1)
    return SpreadsheetDate(dt)


@convert_args_to_pydatetime()
def DAY(value: DateValue) -> int:
    """Converts a serial number to a day of the month"""
    return value.day


@convert_args_to_pydatetime()
def DAYS(end_date: DateValue, start_date: DateValue) -> int:
    """Returns the number of days between two dates"""
    return (end_date - start_date).days


@convert_args_to_pydatetime([0, 1])
def DAYS360(
    start_date: DateValue, end_date: DateValue, method: BooleanValue = FALSE
) -> int:
    """Calculates the number of days between two dates based on a 360-day year"""
    return (
        date_diff360eu(start_date, end_date)
        if method
        else date_diff360us(start_date, end_date, Method360Us.ModifyStartDate)
    )


@convert_args_to_pydatetime([0])
def EDATE(start_date: DateValue, months: int) -> SpreadsheetDate:
    """Returns the serial number of the date that is the indicated number of months before or after the start date"""
    return SpreadsheetDate(change_month(start_date, months, False))


@convert_args_to_pydatetime([0])
def EOMONTH(start_date: DateValue, months: int) -> SpreadsheetDate:
    """Returns the serial number of the last day of the month before or after a specified number of months"""
    return SpreadsheetDate(change_month(start_date, months, True))


@convert_args_to_pydatetime()
def ISOWEEKNUM(source_date: DateValue) -> int:
    """Returns the number of the ISO week number of the year for a given date"""
    return source_date.isocalendar().week


@convert_args_to_pydatetime()
def MONTH(source_date: DateValue) -> int:
    """Converts a serial number to a month"""
    return source_date.month


@convert_args_to_pydatetime()
def NETWORKDAYS(
    start_date: DateValue, end_date: DateValue, holidays: CellRange | None = None
):
    """Returns the number of whole workdays between two dates"""
    return busday_count(start_date, end_date + timedelta(days=1), holidays=holidays)


@convert_args_to_pydatetime([0, 1, 3])
def NETWORKDAYS_INTL(
    start_date: DateValue,
    end_date: DateValue,
    weekend: str | int = 1,
    holidays: DateValue | CellRange | None = None,
):
    """Returns the number of whole workdays between two dates using parameters to indicate which and how many days are weekend days"""
    weekmask = _prepare_weekmask(weekend)
    if isinstance(weekmask, SpreadsheetError):
        return weekmask

    if weekmask == "0000000":
        return 0

    if (
        holidays is not None
        and not isinstance(holidays, Iterable)
        and not isinstance(holidays, str)
    ):
        holidays = [holidays]

    return busday_count(
        start_date, end_date + timedelta(days=1), weekmask=weekmask, holidays=holidays
    )


NETWORKDAYS.INTL = NETWORKDAYS_INTL


@convert_args_to_pydatetime([0])
def WEEKDAY(source_date: DateValue, return_type: int = 1) -> int:
    """Converts a serial number to a day of the week"""
    res = source_date.weekday()
    if return_type == 1:
        res = 1 if res == 6 else res + 2
    elif return_type == 2:
        res += 1
    elif 11 <= return_type <= 17:
        if res < return_type - 11:
            res += 19 - return_type
        else:
            res -= return_type - 12
    elif return_type != 3:
        return NUM_ERROR
    return res


@convert_args_to_pydatetime([0])
def WEEKNUM(source_date: DateValue, return_type: int = 1) -> int:
    """Converts a serial number to a number representing where the week falls numerically with a year"""
    if return_type not in {1, 2, 11, 12, 13, 14, 15, 16, 17, 21}:
        return NUM_ERROR
    return weeknum(source_date, return_type)


@convert_args_to_pydatetime([0, 2])
def WORKDAY(
    start_date: DateValue, days: int, holidays: CellRange | None = None
) -> SpreadsheetDate:
    """Returns the serial number of the date before or after a specified number of workdays"""
    return SpreadsheetDate(
        busday_offset(start_date, days, holidays=holidays, roll="backward").astype(date)
    )


@convert_args_to_pydatetime([0, 3])
def WORKDAY_INTL(
    start_date: DateValue,
    days: int,
    weekend: int = 1,
    holidays: DateValue | CellRange | None = None,
) -> SpreadsheetDate:
    """Returns the serial number of the date before or after a specified number of workdays using parameters to indicate which and how many days are weekend days"""

    weekmask = _prepare_weekmask(weekend)
    if isinstance(weekmask, SpreadsheetError):
        return weekmask

    if (
        holidays is not None
        and not isinstance(holidays, Iterable)
        and not isinstance(holidays, str)
    ):
        holidays = [holidays]

    try:
        offset = busday_offset(
            start_date, days, weekmask=weekmask, holidays=holidays, roll="backward"
        ).astype(date)
        return SpreadsheetDate(offset)
    except ValueError:
        return VALUE_ERROR


WORKDAY.INTL = WORKDAY_INTL


@convert_args_to_pydatetime()
def YEAR(source_date: DateValue) -> int:
    """Converts a serial number to a year"""
    return source_date.year


@convert_args_to_pydatetime([0, 1])
def YEARFRAC(start_date, end_date, basis: int = 0):
    """Returns the year fraction representing the number of whole days between start_date and end_date"""
    if basis not in _DAY_COUNT:
        return NUM_ERROR
    return calc_year_frac(start_date, end_date, basis)
