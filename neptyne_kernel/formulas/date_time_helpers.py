from calendar import isleap, monthrange
from datetime import date, datetime, time
from enum import IntEnum
from functools import wraps
from typing import Callable, Iterable

from dateutil.parser import ParserError
from dateutil.relativedelta import relativedelta

from ..spreadsheet_datetime import (
    SpreadsheetDate,
    SpreadsheetDateTime,
    excel2date,
    excel2datetime,
)
from ..spreadsheet_error import VALUE_ERROR, SpreadsheetError
from .helpers import Numeric, _flatten_range

BaseType = int | float | str
DateTimeValue = BaseType | SpreadsheetDateTime
DateValue = BaseType | SpreadsheetDate


class DayCountBasis(IntEnum):
    UsPsa30_360 = 0
    ActualActual = 1
    Actual360 = 2
    Actual365 = 3
    Europ30_360 = 4


class Method360Us(IntEnum):
    ModifyStartDate = 0
    ModifyBothDates = 1


class AccrIntCalcMethod(IntEnum):
    FromFirstToSettlement = 0
    FromIssueToSettlement = 1


class NumDenumPosition(IntEnum):
    Denumerator = 0
    Numerator = 1


def last_day_of_month(y: int, m: int, d: int) -> bool:
    return monthrange(y, m)[1] == d


def last_day_of_february(check_date: date) -> bool:
    return check_date.month == 2 and last_day_of_month(
        check_date.year, check_date.month, check_date.day
    )


def freq2months(freq: int) -> int:
    return 12 // freq


def last_day_of_month_basis(y: int, m: int, d: int, basis: int) -> bool:
    return last_day_of_month(y, m, d) or (
        d == 30 and basis == DayCountBasis.UsPsa30_360
    )


def change_month(org_date: date, num_months: int, return_last_day: bool) -> date:
    def get_last_day(y: int, m: int) -> int:
        return monthrange(y, m)[1]

    new_date = org_date + relativedelta(months=num_months)
    if return_last_day:
        return new_date.replace(day=get_last_day(new_date.year, new_date.month))
    return new_date


def days_of_month(y: int, m: int) -> int:
    return monthrange(y, m)[1]


def dates_aggregate1(
    start_date: date,
    end_date: date,
    num_months: int,
    f: Callable,
    acc: Numeric,
    return_last_month: bool,
) -> tuple[date, date, Numeric]:
    def rec(front_date: date, trailing_date: date, acc: Numeric):
        if front_date >= end_date if num_months > 0 else front_date <= end_date:
            return front_date, trailing_date, acc
        else:
            trailing_date = front_date
            front_date = change_month(front_date, num_months, return_last_month)
            acc += f(front_date, trailing_date)
            return rec(front_date, trailing_date, acc)

    return rec(start_date, end_date, acc)


def find_pcd_ncd(
    start_date: date, end_date: date, num_months: int, return_last_month: bool
) -> tuple[date, date]:
    pcd, ncd, _ = dates_aggregate1(
        start_date, end_date, num_months, lambda x1, x2: 0, 0, return_last_month
    )
    return pcd, ncd


def find_coupon_dates(settl: date, mat: date, freq: int) -> tuple[date, date]:
    end_month = last_day_of_month(mat.year, mat.month, mat.day)
    num_months = -freq2months(freq)
    return find_pcd_ncd(mat, settl, num_months, end_month)


def find_previous_coupon_date(settl: date, mat: date, freq: int) -> date:
    return find_coupon_dates(settl, mat, freq)[0]


def find_next_coupon_date(settl: date, mat: date, freq: int) -> date:
    return find_coupon_dates(settl, mat, freq)[1]


def number_of_coupons(settl: date, mat: date, freq: int) -> float:
    pcdate = find_previous_coupon_date(settl, mat, freq)
    months = (mat.year - pcdate.year) * 12 + mat.month - pcdate.month
    return months * freq / 12


def less_or_equal_to_year_apart(date1: date, date2: date) -> bool:
    return date1.year == date2.year or (
        date2.year == date1.year + 1
        and (
            date1.month > date2.month
            or (date1.month == date2.month and date1.day >= date2.day)
        )
    )


def is_feb29_between_consecutive_years(date1: date, date2: date) -> bool:
    y1, m1 = date1.year, date1.month
    y2, m2 = date2.year, date2.month
    if y1 == y2 and isleap(y1):
        return m1 <= 2 and m2 > 2
    elif y1 == y2:
        return False
    elif y2 == y1 + 1:
        if isleap(y1):
            return m1 <= 2
        elif isleap(y2):
            return m2 > 2
        else:
            return False
    return False


def consider_as_bisestile(date1: date, date2: date) -> bool:
    return (
        (date1.year == date2.year and isleap(date1.year))
        or (date2.month == 2 and date2.day == 29)
        or is_feb29_between_consecutive_years(date1, date2)
    )


def date_diff360(sd: int, sm: int, sy: int, ed: int, em: int, ey: int) -> int:
    return (ey - sy) * 360 + (em - sm) * 30 + (ed - sd)


def date_diff365(start_date: date, end_date: date) -> int:
    sd1, sm1, sy1 = start_date.day, start_date.month, start_date.year
    ed1, em1, ey1 = end_date.day, end_date.month, end_date.year
    if sd1 > 28 and sm1 == 2:
        sd1 = 28
    if ed1 > 28 and em1 == 2:
        ed1 = 28
    startd, endd = datetime.date(sy1, sm1, sd1), datetime.date(ey1, em1, ed1)
    return (ey1 - sy1) * 365 + (endd - startd).days


def date_diff360us(start_date: date, end_date: date, method360: int) -> int:
    sd1, sm1, sy1 = start_date.day, start_date.month, start_date.year
    ed1, em1, ey1 = end_date.day, end_date.month, end_date.year
    if (
        last_day_of_february(end_date)
        and (
            last_day_of_february(start_date) or method360 == Method360Us.ModifyBothDates
        )
    ) or (ed1 == 31 and (sd1 >= 30 or method360 == Method360Us.ModifyBothDates)):
        ed1 = 30
    if sd1 == 31 or last_day_of_february(start_date):
        sd1 = 30
    return date_diff360(sd1, sm1, sy1, ed1, em1, ey1)


def date_diff360eu(start_date: date, end_date: date) -> int:
    sd1, sm1, sy1 = start_date.day, start_date.month, start_date.year
    ed1, em1, ey1 = end_date.day, end_date.month, end_date.year
    if sd1 == 31:
        sd1 = 30
    if ed1 == 31:
        ed1 = 30
    return date_diff360(sd1, sm1, sy1, ed1, em1, ey1)


class UsPsa30_360:
    @staticmethod
    def coup_days(settl: date, mat: date, freq: int) -> float:
        return 360 / freq

    @staticmethod
    def coup_pcd(settl: date, mat: date, freq: int) -> date:
        return find_previous_coupon_date(settl, mat, freq)

    @staticmethod
    def coup_ncd(settl: date, mat: date, freq: int) -> date:
        return find_next_coupon_date(settl, mat, freq)

    @staticmethod
    def coup_num(settl: date, mat: date, freq: int) -> float:
        return number_of_coupons(settl, mat, freq)

    @staticmethod
    def coup_days_bs(settl: date, mat: date, freq: int) -> int:
        return date_diff360us(
            UsPsa30_360.coup_pcd(settl, mat, freq),
            settl,
            Method360Us.ModifyStartDate,
        )

    @staticmethod
    def coup_days_nc(settl: date, mat: date, freq: int) -> int:
        pdc = find_previous_coupon_date(settl, mat, freq)
        ndc = find_next_coupon_date(settl, mat, freq)
        tot_days_in_coup = date_diff360us(pdc, ndc, Method360Us.ModifyBothDates)
        days_to_settl = date_diff360us(pdc, settl, Method360Us.ModifyStartDate)
        return tot_days_in_coup - days_to_settl

    @staticmethod
    def days_between(issue: date, settl: date, position: int) -> int:
        return date_diff360us(issue, settl, Method360Us.ModifyStartDate)

    @staticmethod
    def days_in_year(issue: date, settl: date) -> int:
        return 360

    @staticmethod
    def change_month(cdate: date, months: int, return_last_day: bool) -> date:
        return change_month(cdate, months, return_last_day)


class Europ30_360:
    @staticmethod
    def coup_days(settl: date, mat: date, freq: int) -> float:
        return 360 / freq

    @staticmethod
    def coup_pcd(settl: date, mat: date, freq: int) -> date:
        return find_previous_coupon_date(settl, mat, freq)

    @staticmethod
    def coup_ncd(settl: date, mat: date, freq: int) -> date:
        return find_next_coupon_date(settl, mat, freq)

    @staticmethod
    def coup_num(settl: date, mat: date, freq: int) -> float:
        return number_of_coupons(settl, mat, freq)

    @staticmethod
    def coup_days_bs(settl: date, mat: date, freq: int) -> int:
        return date_diff360eu(Europ30_360.coup_pcd(settl, mat, freq), settl)

    @staticmethod
    def coup_days_nc(settl: date, mat: date, freq: int) -> int:
        return date_diff360eu(settl, Europ30_360.coup_ncd(settl, mat, freq))

    @staticmethod
    def days_between(issue: date, settl: date, position) -> int:
        return date_diff360eu(issue, settl)

    @staticmethod
    def days_in_year(issue: date, settl: date) -> int:
        return 360

    @staticmethod
    def change_month(cdate: date, months: int, return_last_day: bool) -> date:
        return change_month(cdate, months, return_last_day)


class Actual360:
    @staticmethod
    def coup_days(settl: date, mat: date, freq: int) -> float:
        return 360 / freq

    @staticmethod
    def coup_pcd(settl: date, mat: date, freq: int) -> date:
        return find_previous_coupon_date(settl, mat, freq)

    @staticmethod
    def coup_ncd(settl: date, mat: date, freq: int) -> date:
        return find_next_coupon_date(settl, mat, freq)

    @staticmethod
    def coup_num(settl: date, mat: date, freq: int) -> float:
        return number_of_coupons(settl, mat, freq)

    @staticmethod
    def coup_days_bs(settl: date, mat: date, freq: int) -> int:
        return (settl - Actual360.coup_pcd(settl, mat, freq)).days

    @staticmethod
    def coup_days_nc(settl: date, mat: date, freq: int) -> int:
        return (Actual360.coup_ncd(settl, mat, freq) - settl).days

    @staticmethod
    def days_between(issue: date, settl: date, position: int) -> int:
        if position == NumDenumPosition.Numerator:
            return (settl - issue).days
        else:
            return date_diff360us(issue, settl, Method360Us.ModifyStartDate)

    @staticmethod
    def days_in_year(issue: date, settl: date) -> int:
        return 360

    @staticmethod
    def change_month(cdate: date, months: int, return_last_day: bool) -> date:
        return change_month(cdate, months, return_last_day)


class Actual365:
    @staticmethod
    def coup_days(settl: date, mat: date, freq: int) -> float:
        return 365 / freq

    @staticmethod
    def coup_pcd(settl: date, mat: date, freq: int):
        return find_previous_coupon_date(settl, mat, freq)

    @staticmethod
    def coup_ncd(settl: date, mat: date, freq: int):
        return find_next_coupon_date(settl, mat, freq)

    @staticmethod
    def coup_num(settl: date, mat: date, freq: int):
        return number_of_coupons(settl, mat, freq)

    @staticmethod
    def coup_days_bs(settl: date, mat: date, freq: int):
        return (settl - Actual365.coup_pcd(settl, mat, freq)).days

    @staticmethod
    def coup_days_nc(settl: date, mat: date, freq: int):
        return (Actual365.coup_ncd(settl, mat, freq) - settl).days

    @staticmethod
    def days_between(issue: date, settl: date, position: int) -> int:
        if position == NumDenumPosition.Numerator:
            return (settl - issue).days
        else:
            return date_diff365(issue, settl)

    @staticmethod
    def days_in_year(issue: date, settl: date) -> int:
        return 365

    @staticmethod
    def change_month(cdate: date, months: int, return_last_day: bool) -> date:
        return change_month(cdate, months, return_last_day)


def actual_coup_days(settl: date, mat: date, freq: int) -> int:
    pcd = find_previous_coupon_date(settl, mat, freq)
    ncd = find_next_coupon_date(settl, mat, freq)
    return (ncd - pcd).days


class ActualActual:
    @staticmethod
    def coup_days(settl: date, mat: date, freq: int) -> int:
        return actual_coup_days(settl, mat, freq)

    @staticmethod
    def coup_pcd(settl: date, mat: date, freq: int) -> date:
        return find_previous_coupon_date(settl, mat, freq)

    @staticmethod
    def coup_ncd(settl: date, mat: date, freq: int) -> date:
        return find_next_coupon_date(settl, mat, freq)

    @staticmethod
    def coup_num(settl: date, mat: date, freq: int) -> float:
        return number_of_coupons(settl, mat, freq)

    @staticmethod
    def coup_days_bs(settl: date, mat: date, freq: int) -> int:
        return (settl - ActualActual.coup_pcd(settl, mat, freq)).days

    @staticmethod
    def coup_days_nc(settl: date, mat: date, freq: int) -> int:
        return (ActualActual.coup_ncd(settl, mat, freq) - settl).days

    @staticmethod
    def days_between(start_date: date, end_date: date, position: int) -> int:
        return (end_date - start_date).days

    @staticmethod
    def days_in_year(issue: date, settl: date) -> Numeric:
        if not less_or_equal_to_year_apart(issue, settl):
            tot_years = settl.year - issue.year + 1
            tot_days = (date(settl.year + 1, 1, 1) - date(issue.year, 1, 1)).days
            return tot_days / tot_years
        else:
            return 366 if consider_as_bisestile(issue, settl) else 365

    @staticmethod
    def change_month(cdate: date, months: int, return_last_day: bool) -> date:
        return change_month(cdate, months, return_last_day)


_DAY_COUNT = {
    DayCountBasis.UsPsa30_360: UsPsa30_360,
    DayCountBasis.ActualActual: ActualActual,
    DayCountBasis.Actual360: Actual360,
    DayCountBasis.Actual365: Actual365,
    DayCountBasis.Europ30_360: Europ30_360,
}


def calc_coup_days(settlement: date, maturity: date, frequency: int, basis: int) -> int:
    if maturity <= settlement:
        return VALUE_ERROR
    dc = _DAY_COUNT[basis]
    return dc.coup_days(settlement, maturity, frequency)


def calc_coup_pcd(settlement: date, maturity: date, frequency: int, basis: int) -> date:
    if maturity <= settlement:
        return VALUE_ERROR
    dc = _DAY_COUNT[basis]
    return dc.coup_pcd(settlement, maturity, frequency)


def calc_coup_ncd(settlement: date, maturity: date, frequency: int, basis: int) -> date:
    if maturity <= settlement:
        return VALUE_ERROR
    dc = _DAY_COUNT[basis]
    return dc.coup_ncd(settlement, maturity, frequency)


def calc_coup_num(
    settlement: date, maturity: date, frequency: int, basis: int
) -> float:
    if maturity <= settlement:
        return VALUE_ERROR
    dc = _DAY_COUNT[basis]
    return dc.coup_num(settlement, maturity, frequency)


def calc_coup_days_bs(
    settlement: date, maturity: date, frequency: int, basis: int
) -> int:
    if maturity <= settlement:
        return VALUE_ERROR
    dc = _DAY_COUNT[basis]
    return dc.coup_days_bs(settlement, maturity, frequency)


def calc_coup_days_nc(
    settlement: date, maturity: date, frequency: int, basis: int
) -> int:
    if maturity <= settlement:
        return VALUE_ERROR
    dc = _DAY_COUNT[basis]
    return dc.coup_days_nc(settlement, maturity, frequency)


def calc_year_frac(start_date: date, end_date: date, basis: int) -> Numeric:
    if end_date <= start_date:
        return VALUE_ERROR
    dc = _DAY_COUNT[basis]
    return dc.days_between(
        start_date, end_date, NumDenumPosition.Numerator
    ) / dc.days_in_year(start_date, end_date)


def convert_args_to_pydatetime(
    datetime_args=True, return_pydatetime=True, cast_to_date=True
):
    def decorator(f):
        @wraps(f)
        def wrapper(*args):
            def cast(arg):
                try:
                    arg = (
                        SpreadsheetDate(arg)
                        if cast_to_date
                        else SpreadsheetDateTime(arg)
                    )
                    if return_pydatetime:
                        return excel2date(arg) if cast_to_date else excel2datetime(arg)
                    return arg
                except (ValueError, ParserError):
                    return VALUE_ERROR

            new_args = []
            for i, arg in enumerate(args):
                if datetime_args is True or i in datetime_args:
                    if isinstance(arg, date | time):
                        if cast_to_date and isinstance(arg, datetime):
                            arg = arg.date()
                    elif isinstance(arg, int | float | str):
                        arg = cast(arg)
                    elif isinstance(arg, Iterable):
                        range_args = []
                        for a in _flatten_range(arg):
                            a = cast(a)
                            if isinstance(a, SpreadsheetError):
                                return a
                            range_args.append(a)
                        arg = range_args
                    elif arg is None:
                        arg = []
                    elif not isinstance(
                        arg, SpreadsheetDate if cast_to_date else SpreadsheetDateTime
                    ):
                        return VALUE_ERROR
                    if isinstance(arg, SpreadsheetError):
                        return arg
                new_args.append(arg)

            return f(*new_args)

        return wrapper

    return decorator
