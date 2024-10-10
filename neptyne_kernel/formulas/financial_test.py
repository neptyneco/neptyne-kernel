from datetime import date, datetime

import pytest

from ..cell_range import CellRange
from ..spreadsheet_datetime import SpreadsheetDate
from .boolean import FALSE, TRUE
from .date_time import DATE
from .financial import (
    ACCRINT,
    ACCRINTM,
    AMORDEGRC,
    AMORLINC,
    COUPDAYBS,
    COUPDAYS,
    COUPDAYSNC,
    COUPNCD,
    COUPNUM,
    COUPPCD,
    CUMIPMT,
    CUMPRINC,
    DB,
    DDB,
    DISC,
    DOLLARDE,
    DOLLARFR,
    DURATION,
    EFFECT,
    FV,
    FVSCHEDULE,
    INTRATE,
    IPMT,
    IRR,
    ISPMT,
    MDURATION,
    MIRR,
    NOMINAL,
    NPER,
    NPV,
    ODDFPRICE,
    ODDFYIELD,
    ODDLPRICE,
    ODDLYIELD,
    PDURATION,
    PMT,
    PPMT,
    PRICE,
    PRICEDISC,
    PRICEMAT,
    PV,
    RATE,
    RECEIVED,
    RRI,
    SLN,
    SYD,
    TBILLEQ,
    TBILLPRICE,
    TBILLYIELD,
    VDB,
    XIRR,
    XNPV,
    YIELD,
    YIELDDISC,
    YIELDMAT,
)


@pytest.mark.parametrize(
    "issue, first_interest, settlement, rate, par, frequency, basis, calc_method, result",
    [
        (
            date(2012, 6, 1),
            datetime(2012, 12, 31),
            datetime(2012, 8, 1),
            0.1,
            1000,
            2,
            0,
            TRUE,
            16.666667,
        ),
        (39508, 39691, 39569, 0.1, 1000, 2, 0, TRUE, 16.666667),
        (DATE(2008, 3, 5), 39691, 39569, 0.1, 1000, 2, 0, FALSE, 15.555556),
        (DATE(2008, 4, 5), 39691, 39569, 0.1, 1000, 2, 0, TRUE, 7.2222222),
        (
            DATE(2012, 6, 1),
            DATE(2012, 12, 31),
            DATE(2012, 8, 1),
            0.1,
            1000,
            2,
            0,
            TRUE,
            16.666667,
        ),
        (
            DATE(2001, 2, 28),
            DATE(2001, 8, 31),
            DATE(2001, 5, 1),
            0.1,
            1500,
            2,
            4,
            TRUE,
            26.25,
        ),
        (
            DATE(2001, 2, 28),
            DATE(2001, 8, 31),
            DATE(2001, 5, 1),
            0.1,
            1500,
            1,
            4,
            TRUE,
            26.25,
        ),
        (
            DATE(2001, 2, 28),
            DATE(2001, 8, 31),
            DATE(2001, 5, 1),
            0.1,
            1500,
            4,
            4,
            TRUE,
            25.416667,
        ),  # 26.25),
        (
            DATE(2012, 1, 1),
            DATE(2012, 6, 1),
            DATE(2012, 3, 29),
            0.05,
            1000,
            2,
            0,
            TRUE,
            12.22222222,
        ),
        (
            DATE(1990, 3, 4),
            DATE(1993, 3, 31),
            DATE(1992, 3, 4),
            0.07,
            10000,
            2,
            0,
            TRUE,
            1401.944444,
        ),
        (
            DATE(1990, 3, 4),
            DATE(1993, 3, 31),
            DATE(1992, 3, 4),
            0.07,
            10000,
            2,
            0,
            FALSE,
            -348.0555556,
        ),
        (
            DATE(1990, 3, 4),
            DATE(1993, 3, 31),
            DATE(1992, 3, 4),
            0.07,
            10000,
            4,
            0,
            TRUE,
            1401.944444,
        ),
        (
            DATE(1990, 3, 4),
            DATE(1993, 3, 31),
            DATE(1992, 3, 4),
            0.07,
            10000,
            4,
            0,
            FALSE,
            -523.0555556,
        ),
        (
            DATE(1990, 3, 4),
            DATE(1993, 3, 31),
            DATE(1992, 3, 4),
            0.07,
            10000,
            4,
            1,
            TRUE,
            1390.277778,
        ),
        (
            DATE(1990, 3, 4),
            DATE(1993, 3, 31),
            DATE(1992, 3, 4),
            0.07,
            10000,
            4,
            1,
            FALSE,
            -534.7222222,
        ),
        (
            DATE(1990, 3, 4),
            DATE(1993, 3, 31),
            DATE(1992, 3, 4),
            0.07,
            10000,
            4,
            2,
            TRUE,
            1390.277778,
        ),
        (
            DATE(1990, 3, 4),
            DATE(1993, 3, 31),
            DATE(1992, 3, 4),
            0.07,
            10000,
            4,
            2,
            FALSE,
            -534.7222222,
        ),
        (
            DATE(1990, 3, 4),
            DATE(1993, 3, 31),
            DATE(1992, 3, 4),
            0.07,
            10000,
            4,
            3,
            TRUE,
            1397.60274,
        ),
        (
            DATE(1990, 3, 4),
            DATE(1993, 3, 31),
            DATE(1992, 3, 4),
            0.07,
            10000,
            4,
            3,
            FALSE,
            -527.3972603,
        ),
        (
            DATE(1990, 3, 4),
            DATE(1993, 3, 31),
            DATE(1992, 3, 4),
            0.07,
            10000,
            4,
            4,
            TRUE,
            1400,
        ),
        (
            DATE(1990, 3, 4),
            DATE(1993, 3, 31),
            DATE(1992, 3, 4),
            0.07,
            10000,
            4,
            4,
            FALSE,
            -525,
        ),
    ],
)
def test_ACCRINT(
    issue, first_interest, settlement, rate, par, frequency, basis, calc_method, result
):
    assert ACCRINT(
        issue, first_interest, settlement, rate, par, frequency, basis, calc_method
    ) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "issue, settlement, rate, par, basis, result",
    [
        (datetime(2012, 1, 1), datetime(2012, 12, 31), 0.04, 1000, 0, 40),
        (39539, 39614, 0.1, 1000, 3, 20.54794521),
        (DATE(2012, 1, 1), DATE(2012, 12, 31), 0.04, 1000, 0, 40),
    ],
)
def test_ACCRINTM(issue, settlement, rate, par, basis, result):
    assert ACCRINTM(issue, settlement, rate, par, basis) == pytest.approx(
        result, rel=1e-3
    )


@pytest.mark.parametrize(
    "cost, date_purchased, first_period, salvage, period, rate, basis, result",
    [
        (2400, 39679, 39813, 300, 1, 0.15, 1, 776),
    ],
)
def test_AMORDEGRC(
    cost, date_purchased, first_period, salvage, period, rate, basis, result
):
    assert AMORDEGRC(
        cost, date_purchased, first_period, salvage, period, rate, basis
    ) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "cost, date_purchased, first_period, salvage, period, rate, basis, result",
    [
        (2400, 39679, 39813, 300, 1, 0.15, 1, 360),
    ],
)
def test_AMORLINC(
    cost, date_purchased, first_period, salvage, period, rate, basis, result
):
    assert AMORLINC(
        cost, date_purchased, first_period, salvage, period, rate, basis
    ) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, frequency, basis, result",
    [
        (DATE(2011, 1, 25), DATE(2011, 11, 15), 2, 1, 71),
        (date(2011, 1, 25), date(2011, 11, 15), 2, 1, 71),
    ],
)
def test_COUPDAYBS(settlement, maturity, frequency, basis, result):
    assert COUPDAYBS(settlement, maturity, frequency, basis) == pytest.approx(
        result, rel=1e-3
    )


@pytest.mark.parametrize(
    "settlement, maturity, frequency, basis, result",
    [
        (DATE(2011, 1, 25), DATE(2011, 11, 15), 2, 1, 181),
        (datetime(2011, 1, 25), date(2011, 11, 15), 2, 1, 181),
    ],
)
def test_COUPDAYS(settlement, maturity, frequency, basis, result):
    assert COUPDAYS(settlement, maturity, frequency, basis) == pytest.approx(
        result, rel=1e-3
    )


@pytest.mark.parametrize(
    "settlement, maturity, frequency, basis, result",
    [
        (DATE(2011, 1, 25), DATE(2011, 11, 15), 2, 1, 110),
        (date(2011, 1, 25), datetime(2011, 11, 15), 2, 1, 110),
    ],
)
def test_COUPDAYSNC(settlement, maturity, frequency, basis, result):
    assert COUPDAYSNC(settlement, maturity, frequency, basis) == pytest.approx(
        result, rel=1e-3
    )


@pytest.mark.parametrize(
    "settlement, maturity, frequency, basis, result",
    [
        (
            datetime(2011, 1, 25),
            datetime(2011, 11, 15),
            2,
            1,
            SpreadsheetDate(datetime(2011, 5, 15).date()),
        ),
        (
            DATE(2011, 1, 25),
            DATE(2011, 11, 15),
            2,
            1,
            SpreadsheetDate(datetime(2011, 5, 15).date()),
        ),
    ],
)
def test_COUPNCD(settlement, maturity, frequency, basis, result):
    assert COUPNCD(settlement, maturity, frequency, basis) == result


@pytest.mark.parametrize(
    "settlement, maturity, frequency, basis, result",
    [
        (date(2007, 1, 25), datetime(2008, 11, 15), 2, 1, 4),
        (DATE(2007, 1, 25), DATE(2008, 11, 15), 2, 1, 4),
    ],
)
def test_COUPNUM(settlement, maturity, frequency, basis, result):
    assert COUPNUM(settlement, maturity, frequency, basis) == result


@pytest.mark.parametrize(
    "settlement, maturity, frequency, basis, result",
    [
        (
            datetime(2011, 1, 25),
            datetime(2011, 11, 15),
            2,
            1,
            SpreadsheetDate(datetime(2010, 11, 15).date()),
        ),
        (
            DATE(2011, 1, 25),
            DATE(2011, 11, 15),
            2,
            1,
            SpreadsheetDate(datetime(2010, 11, 15).date()),
        ),
    ],
)
def test_COUPPCD(settlement, maturity, frequency, basis, result):
    assert COUPPCD(settlement, maturity, frequency, basis) == result


@pytest.mark.parametrize(
    "rate, per, nper, pv, fv, _type, result",
    [
        (0.1 / 12, 1, 3 * 12, 8000, 0, 0, -66.67),
        (0.1, 3, 3, 8000, 0, 0, -292.45),
    ],
)
def test_IPMT(rate, per, nper, pv, fv, _type, result):
    assert IPMT(rate, per, nper, pv, fv, _type) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "rate, nper, pv, start_period, end_period, _type, result",
    [
        (0.09 / 12, 30 * 12, 125000, 13, 24, 0, -11135.23213),
    ],
)
def test_CUMIPMT(rate, nper, pv, start_period, end_period, _type, result):
    assert CUMIPMT(rate, nper, pv, start_period, end_period, _type) == pytest.approx(
        result, rel=1e-3
    )


@pytest.mark.parametrize(
    "rate, per, nper, pv, fv, _type, result",
    [
        (0.1 / 12, 1, 2 * 12, 2000, 0, 0, -75.62),
        (0.08, 10, 10, 200000, 0, 0, -27598.05),
    ],
)
def test_PPMT(rate, per, nper, pv, fv, _type, result):
    assert PPMT(rate, per, nper, pv, fv, _type) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "rate, nper, pv, start_period, end_period, _type, result",
    [
        (0.09 / 12, 30 * 12, 125000, 13, 24, 0, -934.1071234),
        (0.09 / 12, 30 * 12, 125000, 1, 1, 0, -68.27827118),
    ],
)
def test_CUMPRINC(rate, nper, pv, start_period, end_period, _type, result):
    assert CUMPRINC(rate, nper, pv, start_period, end_period, _type) == pytest.approx(
        result, rel=1e-3
    )


@pytest.mark.parametrize(
    "cost, salvage, life, period, month, result",
    [
        (1000000, 100000, 6, 1, 7, 186083.33),
        (1000000, 100000, 6, 2, 7, 259639.42),
        (1000000, 100000, 6, 3, 7, 176814.44),
        (1000000, 100000, 6, 4, 7, 120410.64),
        (1000000, 100000, 6, 5, 7, 81999.64),
        (1000000, 100000, 6, 6, 7, 55841.76),
        (1000000, 100000, 6, 7, 7, 15845.10),
        (750000, 60000, 6, 1, 12, 258000),
        (750000, 60000, 6, 1, 10, 215000),
        (750000, 60000, 6, 5, 12, 47778.78),
        (750000, 60000, 12, 12, 12, 14032.98536),
        (750000, 60000, 11, 11, 12, 15505.5227),
        (750000, 60000, 12, 11, 12, 17324.67328),
        (750000, 60000, 12, 1, 12, 142500),
    ],
)
def test_DB(cost, salvage, life, period, month, result):
    assert DB(cost, salvage, life, period, month) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "cost, salvage, life, period, factor, result",
    [
        (2400, 300, 10 * 365, 1, 2, 1.315068493),
        (2400, 300, 10 * 12, 1, 2, 40),
        (2400, 300, 10, 1, 2, 480),
        (2400, 300, 10, 2, 1.5, 306),
        (2400, 300, 10, 10, 2, 22.1225472),
    ],
)
def test_DDB(cost, salvage, life, period, factor, result):
    assert DDB(cost, salvage, life, period, factor) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, pr, redemption, basis, result",
    [
        (datetime(2018, 1, 7), date(2048, 1, 1), 97.975, 100, 1, 0.000675416),
        (DATE(2018, 1, 7), DATE(2048, 1, 1), 97.975, 100, 1, 0.000675416),
        (DATE(2010, 1, 2), DATE(2039, 12, 31), 90, 100, 0, 0.003333642),
    ],
)
def test_DISC(settlement, maturity, pr, redemption, basis, result):
    assert DISC(settlement, maturity, pr, redemption, basis) == pytest.approx(
        result, rel=1e-3
    )


@pytest.mark.parametrize(
    "fractional_dollar, fraction, result",
    [
        (1.02, 16, 1.125),
        (1.1, 32, 1.3125),
        (100.10, 32, 100.3125),
    ],
)
def test_DOLLARDE(fractional_dollar, fraction, result):
    assert DOLLARDE(fractional_dollar, fraction) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "decimal_dollar, fraction, result",
    [
        (1.125, 16, 1.02),
        (1.125, 32, 1.04),
    ],
)
def test_DOLLARFR(decimal_dollar, fraction, result):
    assert DOLLARFR(decimal_dollar, fraction) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, coupon, yld, frequency, basis, result",
    [
        (datetime(2010, 1, 2), datetime(2039, 12, 31), 0.08, 0.09, 2, 1, 10.94757102),
        (DATE(2010, 1, 2), DATE(2039, 12, 31), 0.08, 0.09, 2, 1, 10.94757102),
        (DATE(2010, 1, 2), DATE(2039, 12, 31), 3, 1.2, 2, 0, 1.327777778),
    ],
)
def test_DURATION(settlement, maturity, coupon, yld, frequency, basis, result):
    assert DURATION(
        settlement, maturity, coupon, yld, frequency, basis
    ) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "nominal_rate, npery, result",
    [
        (0.0525, 4, 0.0535427),
        (0.99, 12, 1.589016751),
    ],
)
def test_EFFECT(nominal_rate, npery, result):
    assert EFFECT(nominal_rate, npery) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "rate, nper, pmt, pv, _type, result",
    [
        (0.06 / 12, 10, -200, -500, 1, 2581.40),
        (0.12 / 12, 12, -1000, 0, 0, 12682.50),
        (0.11 / 12, 35, -2000, 0, 1, 82846.25),
        (0.06 / 12, 12, -100, -1000, 1, 2301.40),
    ],
)
def test_FV(rate, nper, pmt, pv, _type, result):
    assert FV(rate, nper, pmt, pv, _type) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "principal, schedule, result",
    [
        (1, [0.09, 0.11, 0.1], 1.3309),
        (10000, [0.1, 0.95, 0.9, 0.85], 75396.75),
    ],
)
def test_FVSCHEDULE(principal, schedule, result):
    assert FVSCHEDULE(principal, schedule) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, investment, redemption, basis, result",
    [
        (date(2008, 2, 15), date(2008, 5, 15), 1000000, 1014420, 2, 0.05768),
        (DATE(2008, 2, 15), DATE(2008, 5, 15), 1000000, 1014420, 2, 0.05768),
        (DATE(2010, 1, 2), DATE(2019, 12, 31), 90, 140, 2, 0.054794521),
    ],
)
def test_INTRATE(settlement, maturity, investment, redemption, basis, result):
    assert INTRATE(
        settlement, maturity, investment, redemption, basis
    ) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "values, guess, result",
    [
        (CellRange([-70000, 12000, 15000, 18000, 21000]), 0.1, -0.021244848),
        (CellRange([-70000, 12000, 15000, 18000, 21000, 26000]), 0.1, 0.086630948),
        (CellRange([-70000, 12000, 15000]), -0.1, -0.443506941),
    ],
)
def test_IRR(values, guess, result):
    assert IRR(values, guess) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "rate, per, nper, pv, result",
    [
        (0.15, 2, 5, 1000, -90),
    ],
)
def test_ISPMT(rate, per, nper, pv, result):
    assert ISPMT(rate, per, nper, pv) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, coupon, yld, frequency, basis, result",
    [
        (date(2008, 1, 1), date(2016, 1, 1), 0.08, 0.09, 2, 1, 5.736),
        (DATE(2008, 1, 1), DATE(2016, 1, 1), 0.08, 0.09, 2, 1, 5.736),
        (DATE(2010, 1, 2), DATE(2039, 12, 31), 3, 1.2, 2, 0, 0.829861111),
    ],
)
def test_MDURATION(settlement, maturity, coupon, yld, frequency, basis, result):
    assert MDURATION(
        settlement, maturity, coupon, yld, frequency, basis
    ) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "values, finance_rate, reinvest_rate, result",
    [
        (
            CellRange([-120000, 39000, 30000, 21000, 37000, 46000]),
            0.1,
            0.12,
            0.12609413,
        ),
        (CellRange([-120000, 39000, 30000, 21000]), 0.1, 0.12, -0.048044655),
        (
            CellRange([-120000, 39000, 30000, 21000, 37000, 46000]),
            0.1,
            0.14,
            0.134759111,
        ),
    ],
)
def test_MIRR(values, finance_rate, reinvest_rate, result):
    assert MIRR(values, finance_rate, reinvest_rate) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "effect_rate, npery, result",
    [
        (0.053543, 4, 0.05250032),
        (0.1, 4, 0.09645475634),
        (0.03, 2, 0.02977831302),
    ],
)
def test_NOMINAL(effect_rate, npery, result):
    assert NOMINAL(effect_rate, npery) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "rate, pmt, pv, fv, _type, result",
    [
        (0.12 / 12, -100, -1000, 10000, 1, 59.6738657),
        (0.12 / 12, -100, -1000, 10000, 0, 60.0821229),
        (0.12 / 12, -100, -1000, 0, 0, -9.57859404),
    ],
)
def test_NPER(rate, pmt, pv, fv, _type, result):
    assert NPER(rate, pmt, pv, fv, _type) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "rate, values, result",
    [
        (0.1, (-10000, 3000, 4200, 6800), 1188.44),
        (0.08, (-40000, 8000, 9200, 10000, 12000, 14500), 1779.686625),
    ],
)
def test_NPV(rate, values, result):
    assert NPV(rate, *values) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, issue, first_coupon, rate, yld, redemption,frequency, basis, result",
    [
        (
            datetime(2008, 11, 11),
            date(2021, 3, 1),
            date(2008, 10, 15),
            datetime(2009, 3, 1),
            0.0785,
            0.0625,
            100,
            2,
            1,
            113.6,
        ),
        (
            DATE(2008, 11, 11),
            DATE(2021, 3, 1),
            DATE(2008, 10, 15),
            DATE(2009, 3, 1),
            0.0785,
            0.0625,
            100,
            2,
            1,
            113.6,
        ),
        (
            DATE(2019, 2, 1),
            DATE(2022, 2, 15),
            DATE(2018, 12, 1),
            DATE(2019, 2, 15),
            0.05,
            0.06,
            100,
            2,
            0,
            97.26007079,
        ),
    ],
)
def test_ODDFPRICE(
    settlement,
    maturity,
    issue,
    first_coupon,
    rate,
    yld,
    redemption,
    frequency,
    basis,
    result,
):
    assert ODDFPRICE(
        settlement,
        maturity,
        issue,
        first_coupon,
        rate,
        yld,
        redemption,
        frequency,
        basis,
    ) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, issue, first_coupon, rate, pr, redemption, frequency, basis, result",
    [
        (
            datetime(2019, 2, 1),
            date(2022, 2, 15),
            DATE(2018, 12, 1),
            date(2019, 2, 15),
            0.05,
            97,
            100,
            2,
            0,
            0.060966715,
        ),
        (
            DATE(2019, 2, 1),
            DATE(2022, 2, 15),
            DATE(2018, 12, 1),
            DATE(2019, 2, 15),
            0.05,
            97,
            100,
            2,
            0,
            0.060966715,
        ),
        (
            DATE(2008, 11, 11),
            DATE(2021, 3, 1),
            DATE(2008, 10, 15),
            DATE(2009, 3, 1),
            0.0575,
            84.5,
            100,
            2,
            0,
            0.0772,
        ),
    ],
)
def test_ODDFYIELD(
    settlement,
    maturity,
    issue,
    first_coupon,
    rate,
    pr,
    redemption,
    frequency,
    basis,
    result,
):
    assert ODDFYIELD(
        settlement,
        maturity,
        issue,
        first_coupon,
        rate,
        pr,
        redemption,
        frequency,
        basis,
    ) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, last_interest, rate, yld, redemption, frequency, basis, result",
    [
        (
            date(2008, 2, 7),
            datetime(2008, 6, 15),
            date(2007, 10, 15),
            0.0375,
            0.0405,
            100,
            2,
            0,
            99.88,
        ),
        (
            DATE(2008, 2, 7),
            DATE(2008, 6, 15),
            DATE(2007, 10, 15),
            0.0375,
            0.0405,
            100,
            2,
            0,
            99.88,
        ),
    ],
)
def test_ODDLPRICE(
    settlement, maturity, last_interest, rate, yld, redemption, frequency, basis, result
):
    assert ODDLPRICE(
        settlement, maturity, last_interest, rate, yld, redemption, frequency, basis
    ) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, last_interest, rate, pr, redemption, frequency, basis, result",
    [
        (
            datetime(2008, 4, 20),
            datetime(2008, 6, 15),
            datetime(2007, 12, 24),
            0.0375,
            99.875,
            100,
            2,
            0,
            0.0452,
        ),
        (
            DATE(2008, 4, 20),
            DATE(2008, 6, 15),
            DATE(2007, 12, 24),
            0.0375,
            99.875,
            100,
            2,
            0,
            0.0452,
        ),
        (
            DATE(2018, 2, 5),
            DATE(2018, 6, 15),
            DATE(2017, 10, 15),
            0.05,
            99.5,
            100,
            2,
            0,
            0.063196633,
        ),
    ],
)
def test_ODDLYIELD(
    settlement, maturity, last_interest, rate, pr, redemption, frequency, basis, result
):
    assert ODDLYIELD(
        settlement, maturity, last_interest, rate, pr, redemption, frequency, basis
    ) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "rate, pv, fv, result",
    [
        (0.025, 2000, 2200, 3.86),
        (0.025 / 12, 1000, 1200, 87.6),
    ],
)
def test_PDURATION(rate, pv, fv, result):
    assert PDURATION(rate, pv, fv) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "rate, nper, pv, fv, _type, result",
    [
        (0.08 / 12, 10, 10000, 0, 0, -1037.03),
        (0.08 / 12, 10, 10000, 0, 1, -1030.16),
        (0.08 / 12, -10, 10000, 1, 1, 964.042178),
        (0.06 / 12, 18 * 12, 0, 50000, 0, -129.0811609),
    ],
)
def test_PMT(rate, nper, pv, fv, _type, result):
    assert PMT(rate, nper, pv, fv, _type) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, rate, yld, redemption, frequency, basis, result",
    [
        (datetime(2008, 2, 15), date(2017, 11, 15), 0.0575, 0.065, 100, 2, 0, 94.63),
        (DATE(2008, 2, 15), DATE(2017, 11, 15), 0.0575, 0.065, 100, 2, 0, 94.63),
    ],
)
def test_PRICE(settlement, maturity, rate, yld, redemption, frequency, basis, result):
    assert PRICE(
        settlement, maturity, rate, yld, redemption, frequency, basis
    ) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, discount, redemption, basis, result",
    [
        (datetime(2008, 2, 16), DATE(2008, 3, 1), 0.0525, 100, 2, 99.8),
        (DATE(2008, 2, 16), DATE(2008, 3, 1), 0.0525, 100, 2, 99.8),
    ],
)
def test_PRICEDISC(settlement, maturity, discount, redemption, basis, result):
    assert PRICEDISC(
        settlement, maturity, discount, redemption, basis
    ) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, issue, rate, yld, basis, result",
    [
        (
            datetime(2008, 2, 15),
            date(2008, 4, 13),
            DATE(2007, 11, 11),
            0.061,
            0.061,
            0,
            99.98,
        ),
        (
            DATE(2008, 2, 15),
            DATE(2008, 4, 13),
            DATE(2007, 11, 11),
            0.061,
            0.061,
            0,
            99.98,
        ),
        (
            DATE(2010, 1, 2),
            DATE(2039, 12, 31),
            DATE(2010, 1, 1),
            3,
            1.2,
            0,
            245.1347719,
        ),
    ],
)
def test_PRICEMAT(settlement, maturity, issue, rate, yld, basis, result):
    assert PRICEMAT(settlement, maturity, issue, rate, yld, basis) == pytest.approx(
        result, rel=1e-3
    )


@pytest.mark.parametrize(
    "rate, nper, pmt, fv, _type, result",
    [
        (0.08 / 12, 20 * 12, 500, 0, 0, -59777.15),
    ],
)
def test_PV(rate, nper, pmt, fv, _type, result):
    assert PV(rate, nper, pmt, fv, _type) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "nper, pmt, pv, fv, _type, guess, result",
    [
        (4 * 12, -200, 8000, 0, 0, 0.1, 0.007701472),
    ],
)
def test_RATE(nper, pmt, pv, fv, _type, guess, result):
    assert RATE(nper, pmt, pv, fv, _type, guess) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, investment, discount, basis, result",
    [
        (date(2008, 2, 15), datetime(2008, 5, 15), 1000000, 0.0575, 2, 1014584.65),
        (DATE(2008, 2, 15), DATE(2008, 5, 15), 1000000, 0.0575, 2, 1014584.65),
    ],
)
def test_RECEIVED(settlement, maturity, investment, discount, basis, result):
    assert RECEIVED(settlement, maturity, investment, discount, basis) == pytest.approx(
        result, rel=1e-3
    )


@pytest.mark.parametrize(
    "nper, pv, fv, result",
    [
        (96, 10000, 11000, 0.0009933),
        (10.5, 10, 3, -0.1083343751),
        (3, 2, 4, 0.2599210499),
        (1, 10, 0, -1),
    ],
)
def test_RRI(nper, pv, fv, result):
    assert RRI(nper, pv, fv) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "cost, salvage, life, result",
    [
        (30000, 7500, 10, 2250),
        (750000, 60000, 6, 115000),
    ],
)
def test_SLN(cost, salvage, life, result):
    assert SLN(cost, salvage, life) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "cost, salvage, life, per, result",
    [
        (30000, 7500, 10, 1, 4090.91),
        (30000, 7500, 10, 10, 409.09),
    ],
)
def test_SYD(cost, salvage, life, per, result):
    assert SYD(cost, salvage, life, per) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, discount, result",
    [
        (datetime(2008, 3, 31), date(2008, 6, 1), 0.0914, 0.0942),
        (DATE(2008, 3, 31), DATE(2008, 6, 1), 0.0914, 0.0942),
        (DATE(2010, 1, 2), DATE(2010, 12, 31), 0.09, 0.097971075),
    ],
)
def test_TBILLEQ(settlement, maturity, discount, result):
    assert TBILLEQ(settlement, maturity, discount) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, discount, result",
    [
        (date(2008, 3, 31), datetime(2008, 6, 1), 0.09, 98.45),
        (DATE(2008, 3, 31), DATE(2008, 6, 1), 0.09, 98.45),
        (DATE(2010, 1, 2), DATE(2010, 12, 31), 0.0125, 98.73958333),
    ],
)
def test_TBILLPRICE(settlement, maturity, discount, result):
    assert TBILLPRICE(settlement, maturity, discount) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, discount, result",
    [
        (date(2008, 3, 31), date(2008, 6, 1), 98.45, 0.0914),
        (DATE(2008, 3, 31), DATE(2008, 6, 1), 98.45, 0.0914),
        (DATE(2010, 1, 2), DATE(2010, 12, 31), 398.45, -0.742837172),
    ],
)
def test_TBILLYIELD(settlement, maturity, discount, result):
    assert TBILLYIELD(settlement, maturity, discount) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "cost, salvage, life, start_period, end_period, factor, no_switch, result",
    [
        (100, 10, 20, 10, 11, 2, TRUE, 3.486784401),
        (100, 33, 20, 10, 11, 2, FALSE, 1.86784401),
    ],
)
def test_VDB(cost, salvage, life, start_period, end_period, factor, no_switch, result):
    assert VDB(
        cost, salvage, life, start_period, end_period, factor, no_switch
    ) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "values, dates, guess, result",
    [
        (
            CellRange([-10000, 2750, 4250, 3250, 2750]),
            CellRange(
                [
                    datetime(2008, 1, 1),
                    date(2008, 3, 1),
                    DATE(2008, 10, 30),
                    datetime(2009, 2, 15),
                    date(2009, 4, 1),
                ]
            ),
            0.1,
            0.373362535,
        ),
        (
            CellRange([-10000, 2750, 4250, 3250, 2750]),
            CellRange(
                [
                    DATE(2008, 1, 1),
                    DATE(2008, 3, 1),
                    DATE(2008, 10, 30),
                    DATE(2009, 2, 15),
                    DATE(2009, 4, 1),
                ]
            ),
            0.1,
            0.373362535,
        ),
        (
            CellRange([-4000, 200, 250, 300]),
            CellRange(
                [
                    DATE(2012, 1, 1),
                    DATE(2012, 6, 23),
                    DATE(2013, 5, 12),
                    DATE(2014, 2, 9),
                ]
            ),
            0.09,
            -0.644085538,
        ),
    ],
)
def test_XIRR(values, dates, guess, result):
    assert XIRR(values, dates, guess) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "rate, values, dates, result",
    [
        (
            0.09,
            CellRange([-10000, 2750, 4250, 3250, 2750]),
            CellRange(
                [
                    date(2008, 1, 1),
                    DATE(2008, 3, 1),
                    datetime(2008, 10, 30),
                    date(2009, 2, 15),
                    DATE(2009, 4, 1),
                ]
            ),
            2086.65,
        ),
        (
            0.09,
            CellRange([-10000, 2750, 4250, 3250, 2750]),
            CellRange(
                [
                    DATE(2008, 1, 1),
                    DATE(2008, 3, 1),
                    DATE(2008, 10, 30),
                    DATE(2009, 2, 15),
                    DATE(2009, 4, 1),
                ]
            ),
            2086.65,
        ),
    ],
)
def test_XNPV(rate, values, dates, result):
    assert XNPV(rate, values, dates) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, rate, pr, redemption, frequency, basis, result",
    [
        (datetime(2008, 2, 15), date(2016, 11, 15), 0.0575, 95.04287, 100, 2, 0, 0.065),
        (DATE(2008, 2, 15), DATE(2016, 11, 15), 0.0575, 95.04287, 100, 2, 0, 0.065),
        (DATE(2010, 1, 2), DATE(2039, 12, 31), 3, 93.45, 100, 2, 0, 3.187600803),
    ],
)
def test_YIELD(settlement, maturity, rate, pr, redemption, frequency, basis, result):
    assert YIELD(
        settlement, maturity, rate, pr, redemption, frequency, basis
    ) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "settlement, maturity, pr, redemption, basis, result",
    [
        (datetime(2008, 2, 16), date(2008, 3, 1), 99.795, 100, 2, 0.052823),
        (DATE(2008, 2, 16), DATE(2008, 3, 1), 99.795, 100, 2, 0.052823),
        (DATE(2010, 1, 2), DATE(2010, 12, 31), 98.45, 100, 0, 0.015787888),
    ],
)
def test_YIELDDISC(settlement, maturity, pr, redemption, basis, result):
    assert YIELDDISC(settlement, maturity, pr, redemption, basis) == pytest.approx(
        result, rel=1e-3
    )


@pytest.mark.parametrize(
    "settlement, maturity, issue, rate, pr, basis, result",
    [
        (
            DATE(2008, 2, 15),
            datetime(2008, 11, 3),
            date(2007, 11, 8),
            0.0625,
            100.0123,
            0,
            0.061288715,
        ),
        (
            DATE(2008, 2, 15),
            DATE(2008, 11, 3),
            DATE(2007, 11, 8),
            0.0625,
            100.0123,
            0,
            0.061288715,
        ),
        (
            DATE(2010, 1, 2),
            DATE(2039, 12, 31),
            DATE(2010, 1, 1),
            3,
            100.47,
            0,
            2.961248382,
        ),
    ],
)
def test_YIELDMAT(settlement, maturity, issue, rate, pr, basis, result):
    assert YIELDMAT(settlement, maturity, issue, rate, pr, basis) == pytest.approx(
        result, rel=1e-3
    )
