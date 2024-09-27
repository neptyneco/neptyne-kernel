import pytest

from ..cell_range import CellRange
from ..spreadsheet_error import NA_ERROR, NUM_ERROR, VALUE_ERROR
from .engineering import (
    BESSELI,
    BESSELJ,
    BESSELK,
    BESSELY,
    BIN2DEC,
    BIN2HEX,
    BIN2OCT,
    BITAND,
    BITLSHIFT,
    BITOR,
    BITRSHIFT,
    BITXOR,
    COMPLEX,
    CONVERT,
    DEC2BIN,
    DEC2HEX,
    DEC2OCT,
    DELTA,
    ERF,
    ERFC,
    GESTEP,
    HEX2BIN,
    HEX2DEC,
    HEX2OCT,
    IMABS,
    IMAGINARY,
    IMARGUMENT,
    IMCONJUGATE,
    IMCOS,
    IMCOSH,
    IMCOT,
    IMCSC,
    IMCSCH,
    IMDIV,
    IMEXP,
    IMLN,
    IMLOG2,
    IMLOG10,
    IMPOWER,
    IMPRODUCT,
    IMREAL,
    IMSEC,
    IMSECH,
    IMSIN,
    IMSINH,
    IMSQRT,
    IMSUB,
    IMSUM,
    IMTAN,
    OCT2BIN,
    OCT2DEC,
    OCT2HEX,
    str2complex,
)
from .test_helpers import approx_or_error


@pytest.mark.parametrize(
    "x, n, result",
    [
        (1.5, 1, 0.981666428),
        (3.45, 4, 0.651416873),
        (3.45, 4.333, 0.651416873060081),
        (-3.45, 4, 0.651416873060081),
    ],
)
def test_BESSELI(x, n, result):
    assert BESSELI(x, n) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, n, result",
    [
        (1.9, 2, 0.329925829),
        (3.45, 4, 0.196772639864984),
        (3.45, 4.333, 0.196772639864984),
        (-3.45, 4, 0.196772639864984),
    ],
)
def test_BESSELJ(x, n, result):
    assert BESSELJ(x, n) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, n, result",
    [
        (1.5, 1, 0.277387804),
        (3.45, 4, 0.144803466373734),
        (3.45, 4.333, 0.144803466373734),
    ],
)
def test_BESSELK(x, n, result):
    assert BESSELK(x, n) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, n, result",
    [
        (2.5, 1, 0.145918138),
        (3.45, 4, -0.679848116844476),
        (3.45, 4.333, -0.679848116844476),
        (2.6, 3, -0.705956708152387),
    ],
)
def test_BESSELY(x, n, result):
    assert BESSELY(x, n) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "number, result",
    [
        ("120000100", NUM_ERROR),
        ("10000100", 132),
        ("10011110101010110101010101010101", NUM_ERROR),
        ("-10010101", NUM_ERROR),
        ("1111111111", -1),
        ("1100111001", -199),
    ],
)
def test_BIN2DEC(number, result):
    assert BIN2DEC(number) == result


@pytest.mark.parametrize(
    "number, places, result",
    [
        ("11111011", 4, "00FB"),
        ("1110", None, "E"),
        ("1110", 0, NUM_ERROR),
        ("1110", 2, "0E"),
        ("1111111111", None, "FFFFFFFFFF"),
        ("1111111111", -4, NUM_ERROR),
    ],
)
def test_BIN2HEX(number, places, result):
    assert BIN2HEX(number, places) == result


@pytest.mark.parametrize(
    "number, places, result",
    [
        ("1001", 3, "011"),
        ("1100100", None, "144"),
        ("1100100", 6, "000144"),
        ("1111111111", None, "7777777777"),
        ("1111111111", -4, NUM_ERROR),
        ("11111111111111111111", None, NUM_ERROR),
    ],
)
def test_BIN2OCT(number, places, result):
    assert BIN2OCT(number, places) == result


@pytest.mark.parametrize(
    "number, places, result",
    [
        ("F", 8, "00001111"),
        ("B7", None, "10110111"),
        ("FFFFFFFFFF", None, "1111111111"),
    ],
)
def test_HEX2BIN(number, places, result):
    assert HEX2BIN(number, places) == result


@pytest.mark.parametrize(
    "number, result",
    [
        ("A5", 165),
        ("FFFFFFFF5B", -165),
        ("3DA408B9", 1034160313),
    ],
)
def test_HEX2DEC(number, result):
    assert HEX2DEC(number) == result


@pytest.mark.parametrize(
    "number, places, result",
    [
        ("F", 3, "017"),
        ("3B4E", None, "35516"),
        ("FFFFFFFF00", None, "7777777400"),
    ],
)
def test_HEX2OCT(number, places, result):
    assert HEX2OCT(number, places) == result


@pytest.mark.parametrize(
    "number, places, result",
    [
        ("3", 3, "011"),
        ("7777777000", None, "1000000000"),
    ],
)
def test_OCT2BIN(number, places, result):
    assert OCT2BIN(number, places) == result


@pytest.mark.parametrize(
    "number, places, result",
    [
        ("100", 4, "0040"),
        ("7777777533", None, "FFFFFFFF5B"),
    ],
)
def test_OCT2HEX(number, places, result):
    assert OCT2HEX(number, places) == result


@pytest.mark.parametrize(
    "number, result",
    [
        ("54", 44),
        ("7777777533", -165),
    ],
)
def test_OCT2DEC(number, result):
    assert OCT2DEC(number) == result


@pytest.mark.parametrize(
    "number, places, result",
    [
        (9, 4, "1001"),
        (-100, None, "1110011100"),
        (-513, None, NUM_ERROR),
    ],
)
def test_DEC2BIN(number, places, result):
    assert DEC2BIN(number, places) == result


@pytest.mark.parametrize(
    "number, places, result",
    [
        (100, 4, "0064"),
        (-54, None, "FFFFFFFFCA"),
        (28, None, "1C"),
        (64, 1, NUM_ERROR),
    ],
)
def test_DEC2HEX(number, places, result):
    assert DEC2HEX(number, places) == result


@pytest.mark.parametrize(
    "number, places, result",
    [
        (58, 3, "072"),
        (-100, None, "7777777634"),
    ],
)
def test_DEC2OCT(number, places, result):
    assert DEC2OCT(number, places) == result


@pytest.mark.parametrize(
    "number1, number2, result",
    [
        (1, 5, 1),
        (13, 25, 9),
        (6, 10, 2),
        (-4, -2, NUM_ERROR),
    ],
)
def test_BITAND(number1, number2, result):
    assert BITAND(number1, number2) == result


@pytest.mark.parametrize(
    "number1, number2, result",
    [
        (4, 2, 16),
        (6, 3, 48),
        (-4, -2, NUM_ERROR),
        (4, -2, 1),
    ],
)
def test_BITLSHIFT(number1, number2, result):
    assert BITLSHIFT(number1, number2) == result


@pytest.mark.parametrize(
    "number1, number2, result",
    [
        (23, 10, 31),
        (6, 3, 7),
        (-4, -2, NUM_ERROR),
    ],
)
def test_BITOR(number1, number2, result):
    assert BITOR(number1, number2) == result


@pytest.mark.parametrize(
    "number1, number2, result",
    [
        (13, 2, 3),
        (15, 3, 1),
        (-4, -2, NUM_ERROR),
        (4, -2, 16),
    ],
)
def test_BITRSHIFT(number1, number2, result):
    assert BITRSHIFT(number1, number2) == result


@pytest.mark.parametrize(
    "number1, number2, result",
    [
        (5, 3, 6),
        (6, 10, 12),
        (-4, -2, NUM_ERROR),
    ],
)
def test_BITXOR(number1, number2, result):
    assert BITXOR(number1, number2) == result


@pytest.mark.parametrize(
    "number1, number2, result",
    [
        (5, 4, 0),
        (5, 5, 1),
        (0.5, 0, 0),
    ],
)
def test_DELTA(number1, number2, result):
    assert DELTA(number1, number2) == result


@pytest.mark.parametrize(
    "lower_limit, upper_limit, result",
    [
        (0.5, 1.5, 0.445605269),
        (2.8, 13, 7.50132e-05),
        (0.745, None, 0.70792892),
        (1, None, 0.84270079),
    ],
)
def test_ERF(lower_limit, upper_limit, result):
    assert ERF(lower_limit, upper_limit) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, result",
    [
        (0.745, 0.70792892),
        (1, 0.84270079),
    ],
)
def test_ERF_PRECISE(x, result):
    assert ERF.PRECISE(x) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, result",
    [
        (0.5, 0.479500122),
        (2.8, 7.50132e-05),
        (0.745, 0.29207108),
        (1, 0.157299207),
    ],
)
def test_ERFC(x, result):
    assert ERFC(x) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, result",
    [
        (0.5, 0.479500122),
        (1, 0.157299207),
    ],
)
def test_ERFC_PRECISE(x, result):
    assert ERFC.PRECISE(x) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "number, from_unit, to_unit, result",
    [
        (1, "lbm", "kg", 0.4535924),
        (68, "F", "C", 20),
        (2.5, "ft", "sec", NA_ERROR),
        (3.5, "mi", "yd", 6160),
        (15.6, "day", "hr", 374.4),
        (-20, "grain", "g", -1.29598),
        (-10, "C", "F", 14),
        (1, "dyn", "e", NA_ERROR),
        (8, "barrel", "kL", 1.27190),
        (1, "kstone", "lbm", NA_ERROR),
        (256, "Gibit", "Mibyte", 32768),
        (256, "Gbit", "Mbyte", 32000),
        (CONVERT(100, "ft", "m"), "ft", "m", 9.290304),
        (6, "C", "F", 42.8),
        (6, "tsp", "tbs", 2),
        (6, "gal", "l", 22.71741274),
        (6, "mi", "km", 9.656064),
        (6, "km", "mi", 3.728227153),
        (6, "in", "ft", 0.5),
        (6, "cm", "in", 2.362204724),
    ],
)
def test_CONVERT(number, from_unit, to_unit, result):
    assert CONVERT(number, from_unit, to_unit) == approx_or_error(result)


@pytest.mark.parametrize(
    "number, step, result",
    [
        (5, 4, 1),
        (5, 5, 1),
        (-4, -5, 1),
        (-1, 0, 0),
        (-0.5, 0, 0),
        (0, 0, 1),
        (0.5, 0, 1),
        (7.6, 7.6001, 0),
    ],
)
def test_GESTEP(number, step, result):
    assert GESTEP(number, step) == result


@pytest.mark.parametrize(
    "real_num, i_num, suffix, result",
    [
        (3, 4, "i", "3+4i"),
        (3, 4, "j", "3+4j"),
        (0, 1, "i", "i"),
        (1, 0, "i", "1"),
        (0, -1, "i", "-i"),
        (3, -4, "j", "3-4j"),
        (0, 0, "i", "0"),
        (-2, 0, "j", "-2"),
        (2, 0, "j", "2"),
        (2.2, -3.3, "i", "2.2-3.3i"),
        (2.2, -3.3, "k", VALUE_ERROR),
    ],
)
def test_COMPLEX(real_num, i_num, suffix, result):
    assert COMPLEX(real_num, i_num, suffix) == result


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("5+12j", 13),
        ("12.34-56.78i", 58.1055),
        ("2-3j", 3.6056),
    ],
)
def test_IMABS(inumber, result):
    assert IMABS(inumber) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("5+12j", 12),
        ("12.34-56.78i", -56.78),
        ("2-3j", -3),
        ("3+4i", 4),
        ("0-j", -1),
        ("4", 0),
    ],
)
def test_IMAGINARY(inumber, result):
    assert IMAGINARY(inumber) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("5+12j", 1.176005207),
        ("12.34-56.78i", -1.356794138),
        ("2-3j", -0.982793723),
        ("3+4i", 0.92729522),
        ("0-j", -1.570796327),
        ("4", 0),
    ],
)
def test_IMARGUMENT(inumber, result):
    assert IMARGUMENT(inumber) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("5+12j", "5-12j"),
        ("12.34-56.78i", "12.34+56.78i"),
        ("2-3j", "2+3j"),
        ("0-j", "j"),
        ("4", "4"),
    ],
)
def test_IMCONJUGATE(inumber, result):
    assert IMCONJUGATE(inumber) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("1+j", "0.833730025131149-0.988897705762865j"),
        ("1.2-3.4j", "5.43490853562577+13.9483036139884j"),
        ("-j", "1.54308063481524"),
        ("-1", "0.54030230586814"),
    ],
)
def test_IMCOS(inumber, result):
    assert complex(IMCOS(inumber)) == pytest.approx(str2complex(result)[0], rel=1e-3)


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("1.2-3.4j", "-1.75053852987314+0.385729418228941j"),
        ("-j", "0.54030230586814"),
        ("-1", "1.54308063481524"),
    ],
)
def test_IMCOSH(inumber, result):
    assert complex(IMCOSH(inumber)) == pytest.approx(str2complex(result)[0], rel=1e-3)


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("1.2-3.4j", "0.0015021589255484+0.998357637173032j"),
        ("-j", "1.31303528549933j"),
        ("-1", "-0.642092615934331"),
        (0, NUM_ERROR),
    ],
)
def test_IMCOT(inumber, result):
    assert (
        complex(IMCOT(inumber)) == pytest.approx(str2complex(result)[0], rel=1e-3)
        if isinstance(result, str)
        else result
    )


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("1.2-3.4j", "0.0621774637507711+0.024119540185495j"),
        ("-j", "0.850918128239322j"),
        ("-1", "-1.18839510577812"),
        (0, NUM_ERROR),
    ],
)
def test_IMCSC(inumber, result):
    assert (
        complex(IMCSC(inumber)) == pytest.approx(str2complex(result)[0], rel=1e-3)
        if isinstance(result, str)
        else result
    )


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("1.2-3.4j", "-0.622647059064193-0.197415259991078j"),
        ("-j", "1.18839510577812j"),
        (0, NUM_ERROR),
    ],
)
def test_IMCSCH(inumber, result):
    assert (
        complex(IMCSCH(inumber)) == pytest.approx(str2complex(result)[0], rel=1e-3)
        if isinstance(result, str)
        else result
    )


@pytest.mark.parametrize(
    "inumber1, inumber2, result",
    [
        ("-238+240j", "10+24j", "5+12j"),
        ("9+8j", "1+2i", "5-2j"),
        ("9+8j", "0j", NUM_ERROR),
    ],
)
def test_IMDIV(inumber1, inumber2, result):
    assert (
        complex(IMDIV(inumber1, inumber2))
        == pytest.approx(str2complex(result)[0], rel=1e-3)
        if isinstance(result, str)
        else result
    )


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("1.2-3.4j", "-3.20988304005418+0.848426337294029j"),
        ("-j", "0.54030230586814-0.841470984807897j"),
        (-1, 0.367879441171442),
    ],
)
def test_IMEXP(inumber, result):
    assert complex(IMEXP(inumber)) == pytest.approx(str2complex(result)[0], rel=1e-3)


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("1.2-3.4j", "1.28247467873077-1.23150371234085j"),
        ("-j", "-1.5707963267949j"),
        ("-1", "3.14159265358979i"),
        ("-1e20-j", "46.0517018598809-3.14159265358979j"),
        ("2-3j", "1.28247467873077-0.982793723247329j"),
    ],
)
def test_IMLN(inumber, result):
    assert complex(IMLN(inumber)) == pytest.approx(str2complex(result)[0], rel=1e-3)


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("1.2-3.4j", "0.556971676153418-0.534835266713002j"),
        ("-j", "-0.682188176920921j"),
        ("-1", "1.36437635384184i"),
        ("-1e20-j", "20-1.36437635384184j"),
        ("2-3j", "0.556971676153418-0.426821890855467j"),
    ],
)
def test_IMLOG10(inumber, result):
    assert complex(IMLOG10(inumber)) == pytest.approx(str2complex(result)[0], rel=1e-3)


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("1.2-3.4j", "1.85021985907055-1.7766842986305j"),
        ("-j", "-2.2661800709136j"),
        ("-1", "4.53236014182719i"),
        ("-1e20-j", "66.4385618977472-4.53236014182719j"),
        ("2-3j", "1.85021985907055-1.41787163074572j"),
    ],
)
def test_IMLOG2(inumber, result):
    assert complex(IMLOG2(inumber)) == pytest.approx(str2complex(result)[0], rel=1e-3)


@pytest.mark.parametrize(
    "inumber, number, result",
    [
        ("2+3j", 3, "-46+9.00000000000001j"),
        ("2+3j", 2, "-5+12j"),
        (2, 3, 8),
        ("2-3j", 3, "-46-9.00000000000001j"),
    ],
)
def test_IMPOWER(inumber, number, result):
    assert complex(IMPOWER(inumber, number)) == pytest.approx(
        str2complex(result)[0], rel=1e-3
    )


@pytest.mark.parametrize(
    "inumbers, result",
    [
        (("3+4j", "5-3j", "2+3j"), "21+103j"),
        (("3.4+5.6j", 7.8), "26.52+43.68j"),
        ((CellRange(["1+2j", "3-4j", "-5+6j"])), "-67+56i"),
        (("3+4j", "5-3j", "jjj"), NUM_ERROR),
    ],
)
def test_IMPRODUCT(inumbers, result):
    assert (
        complex(IMPRODUCT(*inumbers)) == str2complex(result)[0]
        if isinstance(result, str)
        else result
    )


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("6-9i", 6),
        ("5+12j", 5),
        ("12.34-56.78i", 12.34),
        ("2-3j", 2),
        ("3+4j", 3),
        ("0-j", 0),
        ("4", 4),
    ],
)
def test_IMREAL(inumber, result):
    assert IMREAL(inumber) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("1.2-3.4j", "0.0242528714657428-0.0622432580967562j"),
        ("-j", "0.648054273663885"),
        ("-1", "1.85081571768093"),
    ],
)
def test_IMSEC(inumber, result):
    assert complex(IMSEC(inumber)) == pytest.approx(str2complex(result)[0], rel=1e-3)


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("1.2-3.4j", "-0.544800698755068-0.120046290324579j"),
        ("-j", 1.85081571768093),
        ("-1", "0.648054273663885"),
    ],
)
def test_IMSECH(inumber, result):
    assert complex(IMSECH(inumber)) == pytest.approx(str2complex(result)[0], rel=1e-3)


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("1.2-3.4j", "13.979408806018-5.4228154724634j"),
        ("-j", "-1.1752011936438j"),
        ("-1", "-0.841470984807897"),
    ],
)
def test_IMSIN(inumber, result):
    assert complex(IMSIN(inumber)) == pytest.approx(str2complex(result)[0], rel=1e-3)


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("1.2-3.4j", "-1.45934451018103+0.462696919065088j"),
        ("-j", "-0.841470984807897j"),
        ("-1", "-1.1752011936438"),
    ],
)
def test_IMSINH(inumber, result):
    assert complex(IMSINH(inumber)) == pytest.approx(str2complex(result)[0], rel=1e-3)


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("1.2-3.4j", "1.55008891284726-1.0967112827595j"),
        ("-j", "0.707106781186547-0.707106781186547j"),
        ("-1", "j"),
    ],
)
def test_IMSQRT(inumber, result):
    assert complex(IMSQRT(inumber)) == pytest.approx(str2complex(result)[0], rel=1e-3)


@pytest.mark.parametrize(
    "inumber1, inumber2, result",
    [
        ("-238+240j", "10+24j", "-248+216j"),
        ("9+8j", "1+2i", "8+6j"),
        (3, -2, 5),
        ("a", 3, NUM_ERROR),
    ],
)
def test_IMSUB(inumber1, inumber2, result):
    assert (
        complex(IMSUB(inumber1, inumber2)) == str2complex(result)[0]
        if isinstance(result, str)
        else result
    )


@pytest.mark.parametrize(
    "inumbers, result",
    [
        (("3+4j", "5-3j"), "8+j"),
        (("3.4+5.6j", 7.8), "11.2+5.6j"),
        ((CellRange(["1+2j", "3-4j", "-5+6j"])), "-1+4j"),
        (("3+4j", "5-3j", "hello"), NUM_ERROR),
    ],
)
def test_IMSUM(inumbers, result):
    assert (
        complex(IMSUM(*inumbers)) == str2complex(result)[0]
        if isinstance(result, str)
        else result
    )


@pytest.mark.parametrize(
    "inumber, result",
    [
        ("1.2-3.4j", "0.00150710187580578-1.00164279698914j"),
        ("-j", "-0.761594155955765j"),
        (-1, -1.5574077246549),
    ],
)
def test_IMTAN(inumber, result):
    assert complex(IMTAN(inumber)) == pytest.approx(str2complex(result)[0], rel=1e-3)
