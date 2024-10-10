import math
from calendar import isleap
from datetime import date
from enum import IntEnum
from functools import reduce
from typing import Callable

import pyxirr
from dateutil.relativedelta import relativedelta
from scipy.optimize import root

from ..cell_range import CellRange
from ..spreadsheet_datetime import SpreadsheetDate
from ..spreadsheet_error import NUM_ERROR, ZERO_DIV_ERROR
from .boolean import FALSE, TRUE, BooleanValue
from .date_time_helpers import (
    _DAY_COUNT,
    AccrIntCalcMethod,
    DateValue,
    DayCountBasis,
    Method360Us,
    NumDenumPosition,
    change_month,
    convert_args_to_pydatetime,
    date_diff360us,
    dates_aggregate1,
    days_of_month,
    find_pcd_ncd,
    freq2months,
    last_day_of_month,
)
from .helpers import Numeric, _flatten_range, round_to_decimals, sign

__all__ = [
    "ACCRINT",
    "ACCRINTM",
    "AMORDEGRC",
    "AMORLINC",
    "COUPDAYBS",
    "COUPDAYS",
    "COUPDAYSNC",
    "COUPNCD",
    "COUPNUM",
    "COUPPCD",
    "CUMIPMT",
    "CUMPRINC",
    "DB",
    "DDB",
    "DISC",
    "DOLLARDE",
    "DOLLARFR",
    "DURATION",
    "EFFECT",
    "FV",
    "FVSCHEDULE",
    "INTRATE",
    "IPMT",
    "IRR",
    "ISPMT",
    "MDURATION",
    "MIRR",
    "NOMINAL",
    "NPER",
    "NPV",
    "ODDFPRICE",
    "ODDFYIELD",
    "ODDLPRICE",
    "ODDLYIELD",
    "PDURATION",
    "PMT",
    "PPMT",
    "PRICE",
    "PRICEDISC",
    "PRICEMAT",
    "PV",
    "RATE",
    "RECEIVED",
    "RRI",
    "SLN",
    "SYD",
    "TBILLEQ",
    "TBILLPRICE",
    "TBILLYIELD",
    "VDB",
    "XIRR",
    "XNPV",
    "YIELD",
    "YIELDDISC",
    "YIELDMAT",
]


class PaymentDue(IntEnum):
    EndOfPeriod = 0
    BeginningOfPeriod = 1


@convert_args_to_pydatetime([0, 1, 2])
def ACCRINT(
    issue: DateValue,
    first_interest: DateValue,
    settlement: DateValue,
    rate: Numeric,
    par: Numeric,
    frequency: int,
    basis: int = 0,
    calc_method: BooleanValue = TRUE,
) -> Numeric:
    """Returns the accrued interest for a security that pays periodic interest"""

    if (
        rate <= 0
        or par <= 0
        or frequency not in [1, 2, 4]
        or basis not in _DAY_COUNT
        or issue >= settlement
    ):
        return NUM_ERROR

    dc = _DAY_COUNT[basis]
    num_months = freq2months(frequency)
    num_months_neg = -num_months

    end_month_bond = last_day_of_month(
        first_interest.year, first_interest.month, first_interest.day
    )
    pcd = (
        find_pcd_ncd(first_interest, settlement, num_months, end_month_bond)[0]
        if settlement > first_interest
        and calc_method == AccrIntCalcMethod.FromIssueToSettlement
        else dc.change_month(first_interest, num_months_neg, end_month_bond)
    )
    first_date = issue if issue > pcd else pcd
    days = dc.days_between(first_date, settlement, NumDenumPosition.Numerator)
    coup_days = dc.coup_days(pcd, first_interest, frequency)

    def agg_function(pcd, ncd):
        first_date = issue if issue > pcd else pcd
        if basis == DayCountBasis.UsPsa30_360:
            psa_method = (
                Method360Us.ModifyStartDate
                if issue > pcd
                else Method360Us.ModifyBothDates
            )
            days = date_diff360us(first_date, ncd, psa_method)
            coup_days = date_diff360us(pcd, ncd, Method360Us.ModifyBothDates)
        else:
            if basis == DayCountBasis.Actual365:
                coup_days = 365 / frequency
            else:
                coup_days = dc.days_between(pcd, ncd, NumDenumPosition.Denumerator)
            days = dc.days_between(first_date, ncd, NumDenumPosition.Numerator)
        if issue <= pcd:
            return calc_method
        else:
            return days / coup_days

    a = dates_aggregate1(
        pcd, issue, num_months_neg, agg_function, days / coup_days, end_month_bond
    )[2]
    return par * rate / frequency * a


@convert_args_to_pydatetime([0, 1])
def ACCRINTM(
    issue: DateValue, settlement: DateValue, rate: Numeric, par: Numeric, basis: int = 0
) -> Numeric:
    """Returns the accrued interest for a security that pays interest at maturity"""

    if rate <= 0 or par <= 0 or basis not in _DAY_COUNT or issue >= settlement:
        return NUM_ERROR

    dc = _DAY_COUNT[basis]
    days = dc.days_between(issue, settlement, NumDenumPosition.Numerator)
    days_in_year = dc.days_in_year(issue, settlement)
    return par * rate * (days / days_in_year)


def depr_coeff(asset_life: Numeric) -> float:
    def between(x1, x2):
        return asset_life >= x1 and asset_life <= x2

    if between(3, 4):
        return 1.5
    elif between(5, 6):
        return 2.0
    elif asset_life > 6:
        return 2.5
    else:
        return 1.0


def days_in_year(cdate: date, basis: int) -> int:
    if basis == DayCountBasis.ActualActual:
        return 366 if isleap(cdate.year) else 365
    else:
        return _DAY_COUNT[basis].days_in_year(cdate, cdate)


def first_depr_linc(
    cost: Numeric,
    date_purch: date,
    first_p: date,
    salvage: Numeric,
    rate: Numeric,
    asset_life: Numeric,
    basis: int,
):
    def fix29_february(d1):
        if (
            (basis == DayCountBasis.ActualActual or basis == DayCountBasis.Actual365)
            and isleap(d1.year)
            and d1.month == 2
            and d1.day >= 28
        ):
            return d1.replace(days=28)
        return d1

    dc = _DAY_COUNT[basis]
    days_in_yr = days_in_year(date_purch, basis)
    date_purchased, first_period = fix29_february(date_purch), fix29_february(first_p)
    first_len = dc.days_between(
        date_purchased, first_period, NumDenumPosition.Numerator
    )
    first_depr_temp = first_len / days_in_yr * rate * cost
    first_depr = cost * rate if not first_depr_temp else first_depr_temp
    a_life = asset_life if not first_depr_temp else asset_life + 1.0
    avail_depr = cost - salvage
    return (avail_depr, a_life) if first_depr > avail_depr else (first_depr, a_life)


@convert_args_to_pydatetime([1, 2])
def AMORDEGRC(
    cost: Numeric,
    date_purchased: DateValue,
    first_period: DateValue,
    salvage: Numeric,
    period: Numeric,
    rate: Numeric,
    basis: int = 0,
) -> Numeric:
    """Returns the depreciation for each accounting period by using a depreciation coefficient"""
    if (
        cost <= 0
        or rate <= 0
        or salvage < 0
        or period < 0
        or date_purchased > first_period
        or basis not in _DAY_COUNT
    ):
        return NUM_ERROR
    alife = math.ceil(1 / rate)
    if cost == salvage or period > alife:
        return 0
    else:
        deprc = depr_coeff(alife)
        depr_r = rate * deprc
        first_dep_linc, asset_life = first_depr_linc(
            cost, date_purchased, first_period, salvage, depr_r, alife, basis
        )
        first_depr = round(first_dep_linc)

        def find_depr(counted_period, depr, depr_rate, remain_cost):
            if counted_period > period:
                return round(depr)
            else:
                counted_period += 1
                calc_t = asset_life - counted_period
                depr_temp = (
                    remain_cost * 0.5
                    if abs(calc_t - 2) < 0.0001
                    else depr_rate * remain_cost
                )
                depr_rate = 1 if abs(calc_t - 2) < 0.0001 else depr_rate
                if remain_cost < salvage:
                    if remain_cost - salvage < 0:
                        depr = 0
                    else:
                        depr = remain_cost - salvage
                else:
                    depr = depr_temp
                remain_cost -= depr
                return find_depr(counted_period, depr, depr_rate, remain_cost)

        return first_depr if not period else find_depr(1, 0, depr_r, cost - first_depr)


@convert_args_to_pydatetime([1, 2])
def AMORLINC(
    cost: Numeric,
    date_purchased: Numeric,
    first_period: Numeric,
    salvage: Numeric,
    period: Numeric,
    rate: Numeric,
    basis: int = 0,
) -> Numeric:
    """Returns the depreciation for each accounting period"""
    if (
        cost <= 0
        or rate <= 0
        or salvage < 0
        or period < 0
        or date_purchased > first_period
        or basis not in _DAY_COUNT
    ):
        return NUM_ERROR

    asset_life_temp = math.ceil(1 / rate)

    def find_depr(counted_period, depr, avail_depr):
        if counted_period > period:
            return depr
        else:
            depr = avail_depr if depr > avail_depr else depr
            avail_depr_temp = avail_depr - depr
            avail_depr = 0 if avail_depr_temp < 0 else avail_depr_temp
            return find_depr(counted_period + 1, depr, avail_depr)

    if cost == salvage or period > asset_life_temp:
        return 0
    else:
        first_depr = first_depr_linc(
            cost, date_purchased, first_period, salvage, rate, asset_life_temp, basis
        )[0]
        return (
            first_depr
            if not period
            else find_depr(1.0, rate * cost, cost - salvage - first_depr)
        )


@convert_args_to_pydatetime([0, 1])
def COUPDAYBS(
    settlement: DateValue, maturity: DateValue, frequency: int, basis: int = 0
) -> Numeric:
    """Returns the number of days from the beginning of the coupon period to the settlement date"""
    if frequency not in [1, 2, 4] or basis not in _DAY_COUNT or maturity <= settlement:
        return NUM_ERROR

    dc = _DAY_COUNT[basis]
    return dc.coup_days_bs(settlement, maturity, frequency)


@convert_args_to_pydatetime([0, 1])
def COUPDAYS(
    settlement: DateValue, maturity: DateValue, frequency: int, basis: int = 0
) -> Numeric:
    """Returns the number of days in the coupon period that contains the settlement date"""
    if frequency not in [1, 2, 4] or basis not in _DAY_COUNT or maturity <= settlement:
        return NUM_ERROR

    dc = _DAY_COUNT[basis]
    return dc.coup_days(settlement, maturity, frequency)


@convert_args_to_pydatetime([0, 1])
def COUPDAYSNC(
    settlement: DateValue, maturity: DateValue, frequency: int, basis: int = 0
) -> Numeric:
    """Returns the number of days from the settlement date to the next coupon date"""
    if frequency not in [1, 2, 4] or basis not in _DAY_COUNT or maturity <= settlement:
        return NUM_ERROR

    dc = _DAY_COUNT[basis]
    return dc.coup_days_nc(settlement, maturity, frequency)


@convert_args_to_pydatetime([0, 1])
def COUPNCD(
    settlement: DateValue, maturity: DateValue, frequency: int, basis: int = 0
) -> SpreadsheetDate:
    """Returns the next coupon date after the settlement date"""
    if frequency not in [1, 2, 4] or basis not in _DAY_COUNT or maturity <= settlement:
        return NUM_ERROR

    dc = _DAY_COUNT[basis]
    return SpreadsheetDate(dc.coup_ncd(settlement, maturity, frequency))


@convert_args_to_pydatetime([0, 1])
def COUPNUM(
    settlement: DateValue, maturity: DateValue, frequency: int, basis: int = 0
) -> Numeric:
    """Returns the number of coupons payable between the settlement date and maturity date"""
    if frequency not in [1, 2, 4] or basis not in _DAY_COUNT or maturity <= settlement:
        return NUM_ERROR

    dc = _DAY_COUNT[basis]
    return dc.coup_num(settlement, maturity, frequency)


@convert_args_to_pydatetime([0, 1])
def COUPPCD(
    settlement: DateValue, maturity: DateValue, frequency: int, basis: int = 0
) -> SpreadsheetDate:
    """Returns the previous coupon date before the settlement date"""
    if frequency not in [1, 2, 4] or basis not in _DAY_COUNT or maturity <= settlement:
        return NUM_ERROR

    dc = _DAY_COUNT[basis]
    return SpreadsheetDate(dc.coup_pcd(settlement, maturity, frequency))


def fv_factor(r: Numeric, nper: int) -> Numeric:
    return (1 + r) ** nper


def pv_factor(r: Numeric, nper: int) -> Numeric:
    return 1 / fv_factor(r, nper)


def annuity_certain_pv_factor(r: Numeric, nper: int, pd: PaymentDue) -> Numeric:
    return nper if r == 0 else (1 + r * pd) * (1 - pv_factor(r, nper)) / r


def annuity_certain_fv_factor(r: Numeric, nper: int, pd: PaymentDue) -> Numeric:
    return annuity_certain_pv_factor(r, nper, pd) * fv_factor(r, nper)


def ispmt(r: Numeric, per: Numeric, nper: int, pv: Numeric) -> Numeric:
    coupon = -pv * r
    return coupon - (coupon / nper * per)


def IPMT(
    rate: Numeric, per: int, nper: int, pv: Numeric, fv: Numeric = 0, _type: int = 0
) -> Numeric:
    """Returns the interest payment for an investment for a given period"""
    if (
        not 1 <= per <= nper
        or pv == 0
        or rate <= -1
        or nper <= 0
        or _type not in [0, 1]
    ):
        return NUM_ERROR

    return pyxirr.ipmt(
        rate, per, nper, pv, fv, pmt_at_beginning=_type == PaymentDue.BeginningOfPeriod
    )


def aggr_between(
    start_period: int, end_period: int, f: Callable, initial_value: Numeric
) -> Numeric:
    s = (
        range(start_period, end_period + 1)
        if start_period <= end_period
        else range(end_period, start_period - 1, -1)
    )
    for i in s:
        initial_value = f(initial_value, i)
    return initial_value


def CUMIPMT(
    rate: Numeric,
    nper: int,
    pv: Numeric,
    start_period: int,
    end_period: int,
    _type: int,
) -> Numeric:
    """Returns the cumulative interest paid between two periods"""
    if (
        rate <= 0
        or nper <= 0
        or pv <= 0
        or start_period < 1
        or end_period < 1
        or start_period > end_period
        or _type not in [0, 1]
    ):
        return NUM_ERROR

    return aggr_between(
        math.ceil(start_period),
        end_period,
        lambda acc, per: acc + IPMT(rate, per, nper, pv, 0, _type),
        0,
    )


def PPMT(
    rate: Numeric, per: int, nper: int, pv: Numeric, fv: Numeric = 0, _type: int = 0
) -> Numeric:
    """Returns the payment on the principal for an investment for a given period"""
    if (
        not 1 <= per <= nper
        or pv == 0
        or rate <= -1
        or nper <= 0
        or _type not in [0, 1]
    ):
        return NUM_ERROR

    if abs(per - 1) < 0.0001 and _type == PaymentDue.BeginningOfPeriod:
        return pyxirr.pmt(rate, nper, pv, fv, True)
    elif rate == -1:
        return 0
    else:
        return pyxirr.ppmt(
            rate,
            per,
            nper,
            pv,
            fv,
            pmt_at_beginning=_type == PaymentDue.BeginningOfPeriod,
        )


def CUMPRINC(
    rate: Numeric,
    nper: int,
    pv: Numeric,
    start_period: int,
    end_period: int,
    _type: int,
) -> Numeric:
    """Returns the cumulative principal paid on a loan between two periods"""
    if (
        rate <= 0
        or nper <= 0
        or pv <= 0
        or start_period < 1
        or end_period < 1
        or start_period > end_period
        or _type not in [0, 1]
    ):
        return NUM_ERROR
    return aggr_between(
        math.ceil(start_period),
        end_period,
        lambda acc, per: acc + PPMT(rate, float(per), nper, pv, 0, _type),
        0,
    )


def depr_rate(cost: Numeric, salvage: Numeric, life: Numeric) -> Numeric:
    return round_to_decimals(1 - ((salvage / cost) ** (1 / life)), 3)[0]


def depr_for_first_period(cost: Numeric, rate: Numeric, month: Numeric) -> Numeric:
    return cost * rate * month / 12


def depr_for_period(cost: Numeric, tot_depr: Numeric, rate: Numeric) -> Numeric:
    return (cost - tot_depr) * rate


def depr_for_last_period(
    cost: Numeric, tot_depr: Numeric, rate: Numeric, month: Numeric
) -> Numeric:
    return ((cost - tot_depr) * rate * (12 - month)) / 12


def DB(
    cost: Numeric, salvage: Numeric, life: Numeric, period: Numeric, month: int = 12
) -> Numeric:
    """Returns the depreciation of an asset for a specified period by using the fixed-declining balance method"""
    if (
        not 1 <= month <= 12
        or cost < 0
        or salvage < 0
        or life <= 0
        or period > (life + 1)
        or period <= 0
    ):
        return NUM_ERROR
    rate = depr_rate(cost, salvage, life)

    def _db(tot_depr, per):
        per = int(per)
        if not per:
            depr = depr_for_first_period(cost, rate, month)
            return depr if period <= 1 else _db(depr, per + 1)
        elif per == int(life):
            return depr_for_last_period(cost, tot_depr, rate, month)
        elif per == int(period - 1):
            return depr_for_period(cost, tot_depr, rate)
        else:
            depr = depr_for_period(cost, tot_depr, rate)
            return _db(tot_depr + depr, per + 1)

    return _db(0, 0)


def sln(cost: Numeric, salvage: Numeric, life: int) -> Numeric:
    if not life:
        return 0
    return (cost - salvage) / life


def syd(cost: Numeric, salvage: Numeric, life: int, per: Numeric) -> Numeric:
    return ((cost - salvage) * (life - per + 1) * 2) / (life * (life + 1))


def total_depr(
    cost: Numeric,
    salvage: Numeric,
    life: int,
    period: Numeric,
    factor: Numeric,
    straight_line: bool,
) -> Numeric:
    def _ddb(tot_depr, per):
        frac = period % 1

        def ddb_depr_formula(tot_depr):
            return min((cost - tot_depr) * (factor / life), (cost - salvage - tot_depr))

        def sln_depr_formula(tot_depr, a_period):
            return sln(cost - tot_depr, salvage, life - a_period)

        ddb_depr, sln_depr = ddb_depr_formula(tot_depr), sln_depr_formula(tot_depr, per)
        is_sln = straight_line and ddb_depr < sln_depr
        depr = sln_depr if is_sln else ddb_depr
        new_total_depr = tot_depr + depr
        if not int(period):
            return new_total_depr * frac
        elif int(per) == int(period) - 1:
            ddb_depr_next_period = ddb_depr_formula(new_total_depr)
            sln_depr_next_period = sln_depr_formula(new_total_depr, per + 1)
            is_sln_next_period = (
                straight_line and ddb_depr_next_period < sln_depr_next_period
            )
            if is_sln_next_period:
                if int(period) == int(life):
                    deprNextPeriod = 0
                else:
                    deprNextPeriod = sln_depr_next_period
            else:
                deprNextPeriod = ddb_depr_next_period
            return new_total_depr + deprNextPeriod * frac
        else:
            return _ddb(new_total_depr, per + 1)

    return _ddb(0, 0)


def depr_between_periods(
    cost: Numeric,
    salvage: Numeric,
    life: int,
    start_period,
    end_period,
    factor: Numeric,
    straight_line: bool,
) -> Numeric:
    return total_depr(
        cost, salvage, life, end_period, factor, straight_line
    ) - total_depr(cost, salvage, life, start_period, factor, straight_line)


def ddb(
    cost: Numeric, salvage: Numeric, life: int, period: int, factor: Numeric
) -> Numeric:
    if period >= 2:
        return depr_between_periods(
            cost, salvage, life, period - 1, period, factor, False
        )
    return total_depr(cost, salvage, life, period, factor, False)


def DDB(
    cost: Numeric, salvage: Numeric, life: int, period: int, factor: Numeric = 2
) -> Numeric:
    """Returns the depreciation of an asset for a specified period by using the double-declining balance method or some other method that you specify"""
    if (
        cost < 0
        or salvage < 0
        or factor <= 0
        or salvage > cost
        or period < 1
        or period > life
    ):
        return NUM_ERROR
    if not int(period):
        return min(cost * (factor / life), cost - salvage)
    return ddb(cost, salvage, life, period, factor)


def get_common_factors(settlement: date, maturity: date, basis: int) -> tuple[int, int]:
    dc = _DAY_COUNT[basis]
    dim = dc.days_between(settlement, maturity, NumDenumPosition.Numerator)
    b = dc.days_in_year(settlement, maturity)
    return dim, b


def disc(
    settlement: date, maturity: date, pr: Numeric, redemption: Numeric, basis: int
) -> Numeric:
    dim, b = get_common_factors(settlement, maturity, basis)
    return (-pr / redemption + 1) * b / dim


@convert_args_to_pydatetime([0, 1])
def DISC(
    settlement: DateValue,
    maturity: DateValue,
    pr: Numeric,
    redemption: Numeric,
    basis: int = 0,
) -> Numeric:
    """Returns the discount rate for a security"""
    if maturity <= settlement or pr <= 0 or redemption <= 0 or basis not in _DAY_COUNT:
        return NUM_ERROR
    return disc(settlement, maturity, pr, redemption, basis)


def dollar(fractional_dollar: Numeric, fraction: Numeric, f: Callable) -> Numeric:
    a_base = math.floor(fraction)
    dol = (
        math.floor(fractional_dollar)
        if fractional_dollar > 0
        else math.ceiling(fractional_dollar)
    )
    remainder = fractional_dollar - dol
    digits = 10 ** (math.ceil(math.log10(a_base)))
    return f(a_base, dol, remainder, digits)


def dollar_de(
    a_base: int, dollar: Numeric, remainder: Numeric, digits: Numeric
) -> Numeric:
    return remainder * digits / a_base + dollar


def DOLLARDE(fractional_dollar: Numeric, fraction: Numeric) -> Numeric:
    """Converts a dollar price, expressed as a fraction, into a dollar price, expressed as a decimal number"""
    if fraction < 0:
        return NUM_ERROR
    if 0 <= fraction < 1:
        return ZERO_DIV_ERROR
    return dollar(fractional_dollar, fraction, dollar_de)


def dollar_fr(
    a_base: int, dollar: Numeric, remainder: Numeric, digits: Numeric
) -> Numeric:
    abs_digits = abs(digits)
    return remainder * a_base / abs_digits + dollar


def DOLLARFR(decimal_dollar: Numeric, fraction: Numeric) -> Numeric:
    """Converts a dollar price, expressed as a decimal number, into a dollar price, expressed as a fraction"""
    if fraction < 0:
        return NUM_ERROR
    if not fraction:
        return ZERO_DIV_ERROR
    return dollar(decimal_dollar, fraction, dollar_fr)


def duration(
    settlement: date,
    maturity: date,
    coupon: Numeric,
    yld: Numeric,
    frequency: int,
    basis: int,
    is_mduration: bool,
) -> Numeric:
    dc = _DAY_COUNT[basis]
    dbc = dc.coup_days_bs(settlement, maturity, frequency)
    e = dc.coup_days(settlement, maturity, frequency)
    n = dc.coup_num(settlement, maturity, frequency)
    dsc = e - dbc
    x1 = dsc / e
    x2 = x1 + n - 1
    x3 = yld / frequency + 1
    x4 = x3**x2
    if not x4:
        return NUM_ERROR
    term1 = x2 * 100 / x4
    term3 = 100 / x4

    def aggr_function(acc, index):
        x5 = index - 1 + x1
        x6 = x3**x5
        if not x6:
            return NUM_ERROR
        x7 = (100 * coupon / frequency) / x6
        a, b = acc
        return a + x7 * x5, b + x7

    term2, term4 = aggr_between(1, int(n), aggr_function, (0, 0))
    term5 = term1 + term2
    term6 = term3 + term4
    if not term6:
        NUM_ERROR
    return (
        (term5 / term6) / frequency
        if not is_mduration
        else ((term5 / term6) / frequency) / x3
    )


@convert_args_to_pydatetime([0, 1])
def DURATION(
    settlement: DateValue,
    maturity: DateValue,
    coupon: Numeric,
    yld: Numeric,
    frequency: int,
    basis: int = 0,
) -> Numeric:
    """Returns the annual duration of a security with periodic interest payments"""
    if (
        coupon < 0
        or yld < 0
        or frequency not in [1, 2, 4]
        or basis not in _DAY_COUNT
        or settlement >= maturity
    ):
        return NUM_ERROR

    return duration(settlement, maturity, coupon, yld, frequency, basis, False)


def EFFECT(nominal_rate: Numeric, npery: Numeric) -> Numeric:
    """Returns the effective annual interest rate"""
    if nominal_rate <= 0 or npery < 1:
        return NUM_ERROR
    periods = math.floor(npery)
    return (nominal_rate / periods + 1) ** periods - 1


def FV(
    rate: Numeric, nper: int, pmt: Numeric, pv: Numeric = 0, _type: int = 0
) -> Numeric:
    """Returns the future value of an investment"""
    if _type not in [0, 1]:
        return NUM_ERROR
    if rate == -1 and _type == PaymentDue.BeginningOfPeriod:
        return -(pv * fv_factor(rate, nper))
    elif rate == -1 and _type == PaymentDue.EndOfPeriod:
        return -(pv * fv_factor(rate, nper + pmt))
    else:
        return pyxirr.fv(
            rate, nper, pmt, pv, pmt_at_beginning=_type == PaymentDue.BeginningOfPeriod
        )


def FVSCHEDULE(principal: Numeric, schedule: CellRange) -> Numeric:
    """Returns the future value of an initial principal after applying a series of compound interest rates"""
    return reduce(
        lambda x, y: x * (y + 1), _flatten_range(schedule, none_to_zero=True), principal
    )


def int_rate(
    settlement: date,
    maturity: date,
    investment: Numeric,
    redemption: Numeric,
    basis: int,
):
    dim, b = get_common_factors(settlement, maturity, basis)
    return (redemption - investment) / investment * b / dim


@convert_args_to_pydatetime([0, 1])
def INTRATE(
    settlement: DateValue,
    maturity: DateValue,
    investment: Numeric,
    redemption: Numeric,
    basis: int = 0,
) -> Numeric:
    """Returns the interest rate for a fully invested security"""
    if (
        maturity <= settlement
        or investment <= 0
        or redemption <= 0
        or basis not in _DAY_COUNT
    ):
        return NUM_ERROR
    return int_rate(settlement, maturity, investment, redemption, basis)


def IRR(values: CellRange, guess: Numeric = 0.1) -> Numeric:
    """Returns the internal rate of return for a series of cash flows"""
    return pyxirr.irr(_flatten_range(values), guess=guess)


def ISPMT(rate: Numeric, per: Numeric, nper: int, pv: Numeric) -> Numeric:
    """Calculates the interest paid during a specific period of an investment"""
    if per < 1 or per > nper or nper <= 0:
        return NUM_ERROR

    return ispmt(rate, per, nper, pv)


@convert_args_to_pydatetime([0, 1])
def MDURATION(
    settlement: DateValue,
    maturity: DateValue,
    coupon: Numeric,
    yld: Numeric,
    frequency: int,
    basis: int = 0,
) -> Numeric:
    """Returns the Macauley modified duration for a security with an assumed par value of $100"""
    if (
        coupon < 0
        or yld < 0
        or frequency not in [1, 2, 4]
        or basis not in _DAY_COUNT
        or settlement >= maturity
    ):
        return NUM_ERROR

    return duration(settlement, maturity, coupon, yld, frequency, basis, True)


def MIRR(values: CellRange, finance_rate: Numeric, reinvest_rate: Numeric) -> Numeric:
    """Returns the internal rate of return where positive and negative cash flows are financed at different rates"""
    return pyxirr.mirr(_flatten_range(values), finance_rate, reinvest_rate)


def nominal(effectRate: Numeric, npery: Numeric) -> Numeric:
    periods = math.floor(npery)
    return ((effectRate + 1) ** (1 / periods) - 1) * periods


def NOMINAL(effect_rate: Numeric, npery: int) -> Numeric:
    """Returns the annual nominal interest rate"""
    if effect_rate <= 0 or npery < 1:
        return NUM_ERROR
    return nominal(effect_rate, npery)


def NPER(rate: Numeric, pmt: Numeric, pv, fv: Numeric = 0, _type: int = 0) -> Numeric:
    """Returns the number of periods for an investment"""
    return pyxirr.nper(
        rate, pmt, pv, fv, pmt_at_beginning=_type == PaymentDue.BeginningOfPeriod
    )


def NPV(rate: Numeric, value1: Numeric, *values: tuple[Numeric]) -> Numeric:
    """Returns the net present value of an investment based on a series of periodic cash flows and a discount rate"""
    return pyxirr.npv(rate, [value1, *values], start_from_zero=False)


def coup_number(
    mat: date, settl: date, num_months: int, basis: int, is_whole_number: bool
) -> Numeric:
    my, mm, md = mat.year, mat.month, mat.day
    sy, sm, sd = settl.year, settl.month, settl.day
    coupons_temp = 0 if is_whole_number else 1
    end_of_month_temp = last_day_of_month(my, mm, md)
    end_of_month = (
        last_day_of_month(sy, sm, sd)
        if not end_of_month_temp and mm != 2 and md > 28 and md < days_of_month(my, mm)
        else end_of_month_temp
    )
    start_date = change_month(settl, 0, end_of_month)
    coupons = coupons_temp + 1 if settl < start_date else coupons_temp
    dat = change_month(start_date, num_months, end_of_month)
    result = dates_aggregate1(
        dat, mat, num_months, lambda pcd, ncd: 1, coupons, end_of_month
    )[2]
    return result


def days_between_not_neg(dc, start_date: date, end_date: date) -> int:
    result = dc.days_between(start_date, end_date, NumDenumPosition.Numerator)
    return result if result > 0 else 0


def days_between_not_neg_psa_hack(start_date: date, end_date: date) -> int:
    result = date_diff360us(start_date, end_date, Method360Us.ModifyBothDates)
    return result if result > 0 else 0


def days_between_not_neg_with_hack(
    dc, start_date: date, end_date: date, basis: int
) -> int:
    return (
        days_between_not_neg_psa_hack(start_date, end_date)
        if basis == DayCountBasis.UsPsa30_360
        else days_between_not_neg(dc, start_date, end_date)
    )


def odd_fprice(
    settlement: date,
    maturity: date,
    issue: date,
    first_coupon: date,
    rate: Numeric,
    yld: Numeric,
    redemption: Numeric,
    frequency: int,
    basis: int,
) -> Numeric:
    dc = _DAY_COUNT[basis]
    num_months = freq2months(frequency)
    num_months_neg = -num_months
    e = dc.coup_days(settlement, first_coupon, frequency)
    n = dc.coup_num(settlement, maturity, frequency)
    m = frequency
    dfc = days_between_not_neg(dc, issue, first_coupon)
    if dfc < e:
        dsc = days_between_not_neg(dc, settlement, first_coupon)
        a = days_between_not_neg(dc, issue, settlement)
        x = yld / m + 1
        y = dsc / e
        p1 = x
        p3 = p1 ** (n - 1 + y)
        term1 = redemption / p3
        term2 = 100.0 * rate / m * dfc / e / (p1**y)
        term3 = aggr_between(
            2,
            int(n),
            (lambda acc, index: acc + 100 * rate / m / (p1 ** (index - 1 + y))),
            0,
        )
        p2 = rate / m
        term4 = a / e * p2 * 100
        return term1 + term2 + term3 - term4
    else:
        nc = dc.coup_num(issue, first_coupon, frequency)
        late_coupon = [first_coupon]

        def aggr_function(acc, index):
            early_coupon = change_month(late_coupon[0], num_months_neg, False)
            nl = (
                days_between_not_neg(dc, early_coupon, late_coupon[0])
                if basis == DayCountBasis.ActualActual
                else e
            )
            dci = (
                nl if index > 1 else days_between_not_neg(dc, issue, not late_coupon[0])
            )
            start_date = issue if issue > early_coupon else early_coupon
            end_date = settlement if settlement < late_coupon[0] else late_coupon[0]
            a = days_between_not_neg(dc, start_date, end_date)
            late_coupon[0] = early_coupon
            dcnl, anl = acc
            return dcnl + dci / nl, anl + a / nl

        dcnl, anl = aggr_between(nc, 1, aggr_function, (0, 0))
        if basis == DayCountBasis.Actual360 or basis == DayCountBasis.Actual365:
            dat = dc.coup_ncd(settlement, first_coupon, frequency)
            dsc = days_between_not_neg(dc, settlement, dat)
        else:
            dat = dc.coup_pcd(settlement, first_coupon, frequency)
            a = dc.days_between(dat, settlement, NumDenumPosition.Numerator)
            dsc = e - a
        nq = coup_number(first_coupon, settlement, num_months, basis, True)
        n = dc.coup_num(first_coupon, maturity, frequency)
        x = yld / m + 1
        y = dsc / e
        p1 = x
        p3 = p1 ** (y + nq + n)
        term1 = redemption / p3
        term2 = 100.0 * rate / m * dcnl / p1 ** (nq + y)
        term3 = aggr_between(
            1,
            int(n),
            lambda acc, index: acc + 100.0 * rate / m / p1 ** (index + nq + y),
            0,
        )
        term4 = 100.0 * rate / m * anl
        return term1 + term2 + term3 - term4


@convert_args_to_pydatetime([0, 1, 2, 3])
def ODDFPRICE(
    settlement: DateValue,
    maturity: DateValue,
    issue: DateValue,
    first_coupon: DateValue,
    rate: Numeric,
    yld: Numeric,
    redemption: Numeric,
    frequency: int,
    basis: int = 0,
) -> Numeric:
    """Returns the price per $100 face value of a security with an odd first period"""
    if (
        rate < 0
        or yld < 0
        or basis not in _DAY_COUNT
        or frequency not in [1, 2, 4]
        or not maturity > first_coupon > settlement > issue
    ):
        return NUM_ERROR

    return odd_fprice(
        settlement,
        maturity,
        issue,
        first_coupon,
        rate,
        yld,
        redemption,
        frequency,
        basis,
    )


def odd_f_yield(
    settlement: date,
    maturity: date,
    issue: date,
    first_coupon: date,
    rate: Numeric,
    pr: Numeric,
    redemption: Numeric,
    frequency: int,
    basis: int,
) -> Numeric:
    dc = _DAY_COUNT[basis]
    years = dc.days_between(settlement, maturity, NumDenumPosition.Numerator)
    px = pr - 100
    num = rate * years * 100 - px
    denum = px / 4 + years * px / 2 + years * 100
    guess = num / denum
    return root(
        lambda yld: pr
        - odd_fprice(
            settlement,
            maturity,
            issue,
            first_coupon,
            rate,
            yld,
            redemption,
            frequency,
            basis,
        ),
        guess,
    ).x


@convert_args_to_pydatetime([0, 1, 2, 3])
def ODDFYIELD(
    settlement: DateValue,
    maturity: DateValue,
    issue: DateValue,
    first_coupon: DateValue,
    rate: Numeric,
    pr: Numeric,
    redemption: Numeric,
    frequency: int,
    basis: int = 0,
) -> Numeric:
    """Returns the yield of a security with an odd first period"""
    if (
        rate < 0
        or pr < 0
        or basis not in _DAY_COUNT
        or frequency not in [1, 2, 4]
        or not maturity > first_coupon > settlement > issue
    ):
        return NUM_ERROR

    return odd_f_yield(
        settlement,
        maturity,
        issue,
        first_coupon,
        rate,
        pr,
        redemption,
        frequency,
        basis,
    )


def odd_l_func(
    settlement: date,
    maturity: date,
    last_interest: date,
    rate: Numeric,
    pr_or_yld: Numeric,
    redemption: Numeric,
    frequency: int,
    basis: int,
    is_lprice: bool,
) -> Numeric:
    dc = _DAY_COUNT[basis]
    m = frequency
    num_months = 12 // frequency
    last_coupon = last_interest
    nc = dc.coup_num(last_coupon, maturity, frequency)
    early_coupon = [last_coupon]

    def aggr_function(acc, index):
        late_coupon = change_month(early_coupon[0], num_months, False)
        nl = days_between_not_neg_with_hack(dc, early_coupon[0], late_coupon, basis)
        dci = (
            nl
            if index < int(nc)
            else days_between_not_neg_with_hack(dc, early_coupon[0], maturity, basis)
        )
        if late_coupon < settlement:
            a = dci
        elif early_coupon[0] < settlement:
            a = days_between_not_neg(dc, early_coupon[0], settlement)
        else:
            a = 0
        start_date = settlement if settlement > early_coupon[0] else early_coupon[0]
        end_date = maturity if maturity < late_coupon else late_coupon
        dsc = days_between_not_neg(dc, start_date, end_date)
        early_coupon[0] = late_coupon
        dcnl, anl, dscnl = acc
        return dcnl + dci / nl, anl + a / nl, dscnl + dsc / nl

    dcnl, anl, dscnl = aggr_between(1, int(nc), aggr_function, (0, 0, 0))
    x = 100 * rate / m
    term1 = dcnl * x + redemption
    if is_lprice:
        return term1 / (dscnl * pr_or_yld / m + 1) - anl * x
    else:
        term2 = anl * x + pr_or_yld
        return (term1 - term2) / term2 * m / dscnl


@convert_args_to_pydatetime([0, 1, 2])
def ODDLPRICE(
    settlement: DateValue,
    maturity: DateValue,
    last_interest: DateValue,
    rate: Numeric,
    yld: Numeric,
    redemption: Numeric,
    frequency: int,
    basis: int = 0,
) -> Numeric:
    """Returns the price per $100 face value of a security with an odd last period"""

    if (
        rate < 0
        or yld < 0
        or basis not in _DAY_COUNT
        or frequency not in [1, 2, 4]
        or not maturity > settlement > last_interest
    ):
        return NUM_ERROR

    return odd_l_func(
        settlement,
        maturity,
        last_interest,
        rate,
        yld,
        redemption,
        frequency,
        basis,
        True,
    )


@convert_args_to_pydatetime([0, 1, 2])
def ODDLYIELD(
    settlement: DateValue,
    maturity: DateValue,
    last_interest: DateValue,
    rate: Numeric,
    pr: Numeric,
    redemption: Numeric,
    frequency: int,
    basis: int = 0,
) -> Numeric:
    """Returns the yield of a security with an odd last period"""
    if (
        rate < 0
        or pr <= 0
        or basis not in _DAY_COUNT
        or frequency not in [1, 2, 4]
        or not maturity > settlement > last_interest
    ):
        return NUM_ERROR

    return odd_l_func(
        settlement,
        maturity,
        last_interest,
        rate,
        pr,
        redemption,
        frequency,
        basis,
        False,
    )


def PDURATION(rate: Numeric, pv: Numeric, fv: Numeric) -> Numeric:
    """Returns the number of periods required by an investment to reach a specified value"""
    if rate <= 0 or pv <= 0 or fv <= 0:
        return NUM_ERROR
    return (math.log(fv) - math.log(pv)) / math.log(1 + rate)


def PMT(
    rate: Numeric, nper: int, pv: Numeric, fv: Numeric = 0, _type: int = 0
) -> Numeric:
    """Returns the periodic payment for an annuity"""
    return pyxirr.pmt(
        rate, nper, pv, fv, pmt_at_beginning=_type == PaymentDue.BeginningOfPeriod
    )


def get_price_yield_factors(
    settlement: date, maturity: date, frequency: int, basis: int
) -> Numeric:
    dc = _DAY_COUNT[basis]
    n = dc.coup_num(settlement, maturity, frequency)
    pcd = dc.coup_pcd(settlement, maturity, frequency)
    a = dc.days_between(pcd, settlement, NumDenumPosition.Numerator)
    e = dc.coup_days(settlement, maturity, frequency)
    return n, pcd, a, e, e - a


def price(
    settlement: date,
    maturity: date,
    rate: Numeric,
    yld: Numeric,
    redemption: Numeric,
    frequency: int,
    basis: int,
) -> Numeric:
    n, pcd, a, e, dsc = get_price_yield_factors(settlement, maturity, frequency, basis)
    coupon = 100 * rate / frequency
    accr_int = 100 * rate / frequency * a / e

    def pv_factor(k):
        return (1 + yld / frequency) ** (k - 1 + dsc / e)

    pv_of_redemption = redemption / pv_factor(n)
    pv_of_coupons = 0
    for k in range(1, int(n) + 1):
        pv_of_coupons += coupon / pv_factor(k)
    if n == 1:
        return (redemption + coupon) / (1 + dsc / e * yld / frequency) - accr_int
    else:
        return pv_of_redemption + pv_of_coupons - accr_int


@convert_args_to_pydatetime([0, 1])
def PRICE(
    settlement: DateValue,
    maturity: DateValue,
    rate: Numeric,
    yld: Numeric,
    redemption: Numeric,
    frequency: int,
    basis: int = 0,
) -> Numeric:
    """Returns the price per $100 face value of a security that pays periodic interest"""
    if (
        rate < 0
        or yld < 0
        or basis not in _DAY_COUNT
        or frequency not in [1, 2, 4]
        or settlement >= maturity
    ):
        return NUM_ERROR

    return price(settlement, maturity, rate, yld, redemption, frequency, basis)


def price_disc(
    settlement: date, maturity: date, discount: Numeric, redemption: Numeric, basis: int
) -> Numeric:
    dim, b = get_common_factors(settlement, maturity, basis)
    return redemption - discount * redemption * dim / b


@convert_args_to_pydatetime([0, 1])
def PRICEDISC(
    settlement: DateValue,
    maturity: DateValue,
    discount: Numeric,
    redemption: Numeric,
    basis: int = 0,
) -> Numeric:
    """Returns the price per $100 face value of a discounted security"""
    if (
        discount <= 0
        or redemption <= 0
        or basis not in _DAY_COUNT
        or settlement >= maturity
    ):
        return NUM_ERROR
    return price_disc(settlement, maturity, discount, redemption, basis)


def get_mat_factors(
    settlement: date, maturity: date, issue: date, basis: int
) -> Numeric:
    dc = _DAY_COUNT[basis]
    b = dc.days_in_year(issue, settlement)
    dim = dc.days_between(issue, maturity, NumDenumPosition.Numerator)
    a = dc.days_between(issue, settlement, NumDenumPosition.Numerator)
    return b, dim, a, dim - a


def price_mat(
    settlement: date,
    maturity: date,
    issue: date,
    rate: Numeric,
    yld: Numeric,
    basis: int,
) -> Numeric:
    b, dim, a, dsm = get_mat_factors(settlement, maturity, issue, basis)
    return (100 + (dim / b * rate * 100)) / (1 + (dsm / b * yld)) - a / b * rate * 100


@convert_args_to_pydatetime([0, 1, 2])
def PRICEMAT(
    settlement: DateValue,
    maturity: DateValue,
    issue: DateValue,
    rate: Numeric,
    yld: Numeric,
    basis: int = 0,
) -> Numeric:
    """Returns the price per $100 face value of a security that pays interest at maturity"""
    if rate < 0 or yld < 0 or basis not in _DAY_COUNT or settlement >= maturity:
        return NUM_ERROR

    return price_mat(settlement, maturity, issue, rate, yld, basis)


def PV(
    rate: Numeric, nper: int, pmt: Numeric, fv: Numeric = 0, _type: int = 0
) -> Numeric:
    """Returns the present value of an investment"""
    if rate <= -1 or nper <= 0 or _type not in [0, 1]:
        return NUM_ERROR

    return pyxirr.pv(
        rate, nper, pmt, fv, pmt_at_beginning=_type == PaymentDue.BeginningOfPeriod
    )


def RATE(
    nper: int,
    pmt: Numeric,
    pv: Numeric,
    fv: Numeric = 0,
    _type: int = 0,
    guess: Numeric = 0.1,
) -> Numeric:
    """Returns the interest rate per period of an annuity"""

    def have_right_signs(x, y, z):
        return (
            not (sign(x) == sign(y) and sign(y) == sign(z))
            and not (sign(x) == sign(y) and z == 0)
            and not (sign(x) == sign(z) and y == 0)
            and not (sign(y) == sign(z) and x == 0)
        )

    if (
        nper <= 0
        or _type not in [0, 1]
        or (pmt == 0 and pv == 0)
        or not have_right_signs(pmt, pv, fv)
    ):
        return NUM_ERROR

    return pyxirr.rate(
        nper,
        pmt,
        pv,
        fv,
        pmt_at_beginning=_type == PaymentDue.BeginningOfPeriod,
        guess=guess,
    )


def received(
    settlement: date, maturity: date, investment: Numeric, discount: Numeric, basis: int
) -> Numeric:
    dim, b = get_common_factors(settlement, maturity, basis)
    discount_factor = discount * dim / b
    return investment / (1 - discount_factor)


@convert_args_to_pydatetime([0, 1])
def RECEIVED(
    settlement: DateValue,
    maturity: DateValue,
    investment: Numeric,
    discount: Numeric,
    basis: int = 0,
) -> Numeric:
    """Returns the amount received at maturity for a fully invested security"""
    dc = _DAY_COUNT[basis]
    dim = dc.days_between(settlement, maturity, NumDenumPosition.Numerator)
    b = dc.days_in_year(settlement, maturity)
    discount_factor = discount * dim / b
    if (
        discount_factor >= 1
        or maturity <= settlement
        or investment <= 0
        or discount <= 0
    ):
        return NUM_ERROR
    return received(settlement, maturity, investment, discount, basis)


def RRI(nper: int, pv: Numeric, fv: Numeric) -> Numeric:
    """Returns an equivalent interest rate for the growth of an investment"""
    if nper <= 0:
        return NUM_ERROR
    if fv == pv:
        return 0
    if pv == 0 or fv / pv < 0:
        return NUM_ERROR
    return (fv / pv) ** (1 / nper) - 1


def SLN(cost: Numeric, salvage: Numeric, life: int) -> Numeric:
    """Returns the straight-line depreciation of an asset for one period"""
    if cost < 0 or salvage < 0 or life <= 0:
        return NUM_ERROR
    return sln(cost, salvage, life)


def SYD(cost: Numeric, salvage: Numeric, life: int, per: int) -> Numeric:
    """Returns the sum-of-years' digits depreciation of an asset for a specified period"""
    if cost < 0 or salvage < 0 or life <= 0 or not 0 < per <= life:
        return NUM_ERROR
    return syd(cost, salvage, life, per)


def get_dsm(settlement: date, maturity: date, basis: int) -> int:
    dc = _DAY_COUNT[basis]
    return dc.days_between(settlement, maturity, NumDenumPosition.Numerator)


def t_bill_eq(settlement: date, maturity: date, discount: Numeric) -> Numeric:
    dsm = get_dsm(settlement, maturity, DayCountBasis.Actual360)
    if dsm > 182:
        price = (100 - discount * 100 * dsm / 360) / 100
        days = 366 if dsm == 366 else 365
        term = (dsm / days) ** 2 - (2 * dsm / days - 1) * (1 - 1 / price)
        return 2 * (term**0.5 - dsm / days) / (2 * dsm / days - 1)
    else:
        # This is the algo in the docs, but it is valid just above 182
        return 365 * discount / (360 - discount * dsm)


@convert_args_to_pydatetime([0, 1])
def TBILLEQ(settlement: DateValue, maturity: DateValue, discount: Numeric) -> Numeric:
    """Returns the bond-equivalent yield for a Treasury bill"""
    if discount <= 0 or settlement > maturity:
        return NUM_ERROR

    if settlement + relativedelta(years=1) < maturity:
        return NUM_ERROR

    return t_bill_eq(settlement, maturity, discount)


def t_bill_price(settlement: date, maturity: date, discount: Numeric) -> Numeric:
    dsm = get_dsm(settlement, maturity, DayCountBasis.ActualActual)
    return 100 * (1 - discount * dsm / 360)


@convert_args_to_pydatetime([0, 1])
def TBILLPRICE(
    settlement: DateValue, maturity: DateValue, discount: Numeric
) -> Numeric:
    """Returns the price per $100 face value for a Treasury bill"""
    if discount <= 0 or settlement > maturity:
        return NUM_ERROR

    if settlement + relativedelta(years=1) < maturity:
        return NUM_ERROR

    return t_bill_price(settlement, maturity, discount)


def t_bill_yield(settlement: date, maturity: date, pr: Numeric):
    dsm = get_dsm(settlement, maturity, DayCountBasis.ActualActual)
    return (100 - pr) / pr * 360 / dsm


@convert_args_to_pydatetime([0, 1])
def TBILLYIELD(settlement: DateValue, maturity: DateValue, pr: Numeric) -> Numeric:
    """Returns the yield for a Treasury bill"""
    if pr <= 0 or settlement > maturity:
        return NUM_ERROR

    if settlement + relativedelta(years=1) < maturity:
        return NUM_ERROR

    return t_bill_yield(settlement, maturity, pr)


def vdb(
    cost: Numeric,
    salvage: Numeric,
    life: Numeric,
    start_period: Numeric,
    end_period: Numeric,
    factor: Numeric,
    no_switch: bool,
):
    return depr_between_periods(
        cost, salvage, life, start_period, end_period, factor, not no_switch
    )


def VDB(
    cost: Numeric,
    salvage: Numeric,
    life: Numeric,
    start_period: Numeric,
    end_period: Numeric,
    factor: Numeric = 2,
    no_switch: BooleanValue = FALSE,
) -> Numeric:
    """Returns the depreciation of an asset for a specified or partial period by using a declining balance method"""
    if (
        cost < 0
        or salvage < 0
        or life <= 0
        or factor <= 0
        or start_period > life
        or end_period > life
        or start_period > end_period
        or end_period <= 0
    ):
        return NUM_ERROR

    return vdb(cost, salvage, life, start_period, end_period, factor, no_switch)


@convert_args_to_pydatetime([1])
def XIRR(values: CellRange, dates: CellRange, guess: Numeric = 0.1) -> Numeric:
    """Returns the internal rate of return for a schedule of cash flows that is not necessarily periodic"""
    return pyxirr.xirr(dates, _flatten_range(values), guess=guess)


@convert_args_to_pydatetime([2])
def XNPV(rate: Numeric, values: CellRange, dates: CellRange) -> Numeric:
    """Returns the net present value for a schedule of cash flows that is not necessarily periodic"""
    return pyxirr.xnpv(rate, dates, _flatten_range(values))


def yield_func(
    settlement: date,
    maturity: date,
    rate: Numeric,
    pr: Numeric,
    redemption: Numeric,
    frequency: int,
    basis: int,
) -> Numeric:
    n, pcd, a, e, dsr = get_price_yield_factors(settlement, maturity, frequency, basis)
    if n <= 1:
        k = (redemption / 100 + rate / frequency) / (
            pr / 100 + (a / e * rate / frequency)
        ) - 1
        return k * frequency * e / dsr
    return root(
        lambda yld: price(settlement, maturity, rate, yld, redemption, frequency, basis)
        - pr,
        0.05,
    ).x


@convert_args_to_pydatetime([0, 1])
def YIELD(
    settlement: DateValue,
    maturity: DateValue,
    rate: Numeric,
    pr: Numeric,
    redemption: Numeric,
    frequency: int,
    basis: int = 0,
) -> Numeric:
    """Returns the yield on a security that pays periodic interest"""
    if (
        rate < 0
        or pr <= 0
        or redemption <= 0
        or basis not in _DAY_COUNT
        or frequency not in [1, 2, 4]
        or settlement >= maturity
    ):
        return NUM_ERROR

    return yield_func(settlement, maturity, rate, pr, redemption, frequency, basis)


def yield_disc(
    settlement: date, maturity: date, pr: Numeric, redemption: Numeric, basis: int
) -> Numeric:
    dim, b = get_common_factors(settlement, maturity, basis)
    return (redemption - pr) / pr * b / dim


@convert_args_to_pydatetime([0, 1])
def YIELDDISC(
    settlement: DateValue,
    maturity: DateValue,
    pr: Numeric,
    redemption: Numeric,
    basis: int = 0,
) -> Numeric:
    """Returns the annual yield for a discounted security; for example, a Treasury bill"""
    if pr <= 0 or redemption <= 0 or basis not in _DAY_COUNT or settlement >= maturity:
        return NUM_ERROR

    return yield_disc(settlement, maturity, pr, redemption, basis)


def yield_mat(
    settlement: date,
    maturity: date,
    issue: date,
    rate: Numeric,
    pr: Numeric,
    basis: int,
) -> Numeric:
    b, dim, a, dsm = get_mat_factors(settlement, maturity, issue, basis)
    term1 = dim / b * rate + 1 - pr / 100 - a / b * rate
    term2 = pr / 100 + a / b * rate
    return term1 / term2 * b / dsm


@convert_args_to_pydatetime([0, 1, 2])
def YIELDMAT(
    settlement: DateValue,
    maturity: DateValue,
    issue: DateValue,
    rate: Numeric,
    pr: Numeric,
    basis: int = 0,
) -> Numeric:
    """Returns the annual yield of a security that pays interest at maturity"""
    if (
        pr <= 0
        or rate < 0
        or basis not in _DAY_COUNT
        or settlement >= maturity
        or issue >= maturity
        or issue >= settlement
    ):
        return NUM_ERROR

    return yield_mat(settlement, maturity, issue, rate, pr, basis)
