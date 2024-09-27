from datetime import date, datetime, timedelta

import pytest

from ..cell_range import CellRange
from ..spreadsheet_datetime import SpreadsheetDate, SpreadsheetDateTime
from ..spreadsheet_error import NA_ERROR, VALUE_ERROR, SpreadsheetError
from .boolean import FALSE, TRUE
from .stats import T
from .text import (
    ARRAYTOTEXT,
    ASC,
    BAHTTEXT,
    CHAR,
    CLEAN,
    CODE,
    CONCAT,
    CONCATENATE,
    DBCS,
    DOLLAR,
    EXACT,
    FIND,
    FINDB,
    FIXED,
    LEFT,
    LEFTB,
    LEN,
    LENB,
    LOWER,
    MID,
    MIDB,
    NUMBERVALUE,
    PHONETIC,
    PROPER,
    REPLACE,
    REPLACEB,
    REPT,
    RIGHT,
    RIGHTB,
    SEARCH,
    SEARCHB,
    SUBSTITUTE,
    TEXT,
    TEXTJOIN,
    TRIM,
    UNICHAR,
    UNICODE,
    UPPER,
    VALUE,
    VALUETOTEXT,
)


@pytest.mark.parametrize(
    "text, result",
    [
        ("グーグル", "ｸﾞｰｸﾞﾙ"),
        ("123", "123"),
        ("Hello world!", "Hello world!"),
        ("", ""),
    ],
)
def test_ASC(text, result):
    return ASC(text) == result


@pytest.mark.parametrize(
    "text, result",
    [
        ("A", 65),
        ("!", 33),
        ("", VALUE_ERROR),
        ("Ω", 937),
        ("APPLE", 65),
    ],
)
def test_CODE(text, result):
    return CODE(text) == result


TEST_CONCAT1 = CellRange(
    [
        ["A`s", "B`s"],
        ["a1", "b1"],
        ["a2", "b2"],
        ["a4", "b4"],
        ["a5", "b5"],
        ["a6", "b6"],
        ["a7", "b7"],
    ]
)
TEST_CONCAT2 = CellRange(
    [
        ["Data", "First Name", "Last name"],
        ["brook trout", "Andreas", "Hauser"],
        ["species", "Fourth", "Pine"],
        [32],
    ]
)


@pytest.mark.parametrize(
    "texts, result",
    [
        ((TEST_CONCAT1[:, 0], TEST_CONCAT1[:, 1]), "A`sa1a2a4a5a6a7B`sb1b2b4b5b6b7"),
        ((TEST_CONCAT1[1:, :],), "a1b1a2b2a4b4a5b5a6b6a7b7"),
        (
            (
                "Stream population for ",
                TEST_CONCAT2[1][0],
                " ",
                TEST_CONCAT2[2][0],
                " is ",
                TEST_CONCAT2[3][0],
                "/mile.",
            ),
            "Stream population for brook trout species is 32/mile.",
        ),
        ((TEST_CONCAT2[1][1], " ", TEST_CONCAT2[1][2]), "Andreas Hauser"),
        ((TEST_CONCAT2[1][2], ", ", TEST_CONCAT2[1][1]), "Hauser, Andreas"),
        ((TEST_CONCAT2[2][1], " & ", TEST_CONCAT2[2][2]), "Fourth & Pine"),
    ],
)
def test_CONCAT(texts, result):
    assert CONCAT(*texts) == result


@pytest.mark.parametrize(
    "texts,result",
    [
        (("1", "+", "2"), "1+2"),
        (("1", "+", "2"), "1+2"),
        (("pi = ", str(3.1415926)), "pi = 3.1415926"),
        ((str(TRUE), " is so true"), "TRUE is so true"),
    ],
)
def test_CONCATENATE(texts, result):
    assert CONCATENATE(*texts) == result


@pytest.mark.parametrize(
    "text, result",
    [
        ("EXCEL", "EXCEL"),
        ("ｸﾞｰｸﾞﾙ", "グーグル"),
    ],
)
def test_DBCS(text, result):
    return DBCS(text) == result


@pytest.mark.parametrize(
    "number, decimals, result",
    [
        (1234.567, 2, "$1,234.57"),
        (-1234.567, -2, "($1,200)"),
        (-0.123, 4, "($0.1230)"),
        (99.888, 2, "$99.89"),
    ],
)
def test_DOLLAR(number, decimals, result):
    assert DOLLAR(number, decimals) == result


@pytest.mark.parametrize(
    "text1, text2, result",
    [("word", "word", TRUE), ("Word", "word", FALSE), ("w ord", "word", FALSE)],
)
def test_EXACT(text1, text2, result):
    assert EXACT(text1, text2) == result


@pytest.mark.parametrize(
    "find_text, within_text, start_num, result",
    [
        ("M", "Miriam McGovern", 1, 1),
        ("m", "Miriam McGovern", 1, 6),
        ("M", "Miriam McGovern", 3, 8),
        ("M", "Miriam McGovern", 20, VALUE_ERROR),
        ("", "", 3, VALUE_ERROR),
        ("", "", 1, 1),
        ("W", "Miriam McGovern", 1, VALUE_ERROR),
    ],
)
def test_FIND(find_text, within_text, start_num, result):
    FIND(find_text, within_text, start_num) == result


@pytest.mark.parametrize(
    "find_text, within_text, start_num, result",
    [
        ("新", "农历新年", 4, 5),
        ("a", "ＬｉｂｒｅＯｆｆｉｃｅ　Ｃａｌｃ", 1, VALUE_ERROR),  # noqa: RUF001
        ("ａ", "ＬｉｂｒｅＯｆｆｉｃｅ　Ｃａｌｃ", 1, 27),  # noqa: RUF001
        ("ａ", "LibreOffice Ｃａｌｃ", 1, 15),  # noqa: RUF001
    ],
)
def test_FINDB(find_text, within_text, start_num, result):
    FINDB(find_text, within_text, start_num) == result


@pytest.mark.parametrize(
    "number, decimals, no_commas, result",
    [
        (1234.567, 1, FALSE, "1,234.6"),
        (1234.567, -1, FALSE, "1,230"),
        (-1234.567, -1, TRUE, "-1230"),
        (44.332, 2, FALSE, "44.33"),
    ],
)
def test_FIXED(number, decimals, no_commas, result):
    assert FIXED(number, decimals, no_commas) == result


@pytest.mark.parametrize("text, result", [("123", 3), ("", 0), (FALSE, 5)])
def test_LEN(text, result):
    assert LEN(str(text)) == result


@pytest.mark.parametrize("text, result", [("熊本", 4), ("Aeñ", 3)])
def test_LENB(text, result):
    assert LENB(str(text)) == result


@pytest.mark.parametrize(
    "old_text, start_num, num_chars, new_text, result",
    [
        ("abcdefghijk", 6, 5, "*", "abcde*k"),
        ("2009", 3, 2, "10", "2010"),
        ("123456", 1, 3, "@", "@456"),
    ],
)
def test_REPLACE(old_text, start_num, num_chars, new_text, result):
    assert REPLACE(old_text, start_num, num_chars, new_text) == result


@pytest.mark.parametrize(
    "old_text, start_num, num_chars, new_text, result",
    [
        ("中国香港", 5, 4, "北京", "中国北京"),
        ("中国", 1, 0, "?", "?中国"),
        ("中国", 1, 1, "?", "? 国"),
        ("中国", 1, 2, "?", "?国"),
        ("中国", 1, 3, "?", "? "),
        ("中国", 1, 4, "?", "?"),
        ("中国", 2, 0, "?", " ? 国"),
        ("中国", 2, 1, "?", " ?国"),
        ("中国", 2, 2, "?", " ? "),
        ("中国", 2, 3, "?", " ?"),
    ],
)
def test_REPLACEB(old_text, start_num, num_chars, new_text, result):
    assert REPLACEB(old_text, start_num, num_chars, new_text) == result


@pytest.mark.parametrize(
    "text, old_text, new_text, instance_num, result",
    [
        ("Sales Data", "Sales", "Cost", None, "Cost Data"),
        ("Quarter 1, 2008", "1", "2", 1, "Quarter 2, 2008"),
        ("Quarter 1, 2011", "1", "2", 3, "Quarter 1, 2012"),
        ("search for it", "search for", "Google", 2, "search for it"),
        ("January 2, 2012", 2, 3, "one", VALUE_ERROR),
    ],
)
def test_SUBSTITUTE(text, old_text, new_text, instance_num, result):
    assert SUBSTITUTE(text, old_text, new_text, instance_num) == result


@pytest.mark.parametrize(
    "text, num_chars, result",
    [
        ("Sale Price", 4, "Sale"),
        ("Sweden", 1, "S"),
        ("", 10, ""),
        ("100", 2, "10"),
        ("熊本", 2, "熊本"),
    ],
)
def test_LEFT(text, num_chars, result):
    assert LEFT(text, num_chars) == result


@pytest.mark.parametrize(
    "text, num_chars, result",
    [
        ("熊本", 3, "熊 "),
        ("Aeñ", 2, "Ae"),
        ("熊本", 2, "熊"),
        ("", 10, ""),
        ("100", 2, "10"),
        ("じゃぱん1", 2, "じ"),
    ],
)
def test_LEFTB(text, num_chars, result):
    assert LEFTB(text, num_chars) == result


@pytest.mark.parametrize(
    "text, num_chars, result",
    [("Sale Price", 5, "Price"), ("Sweden", 1, "n"), ("", 10, ""), ("200", 2, "00")],
)
def test_RIGHT(text, num_chars, result):
    assert RIGHT(text, num_chars) == result


@pytest.mark.parametrize(
    "text, num_chars, result",
    [
        ("Sale Price", 5, "Price"),
        ("Sweden", 1, "n"),
        ("", 10, ""),
        ("200", 2, "00"),
        ("Aeñ", 2, "eñ"),
        ("熊本", 2, "本"),
        ("中国", 1, " "),
        ("中国", 3, " 国"),
        ("中国", 2, "国"),
        ("中国", 4, "中国"),
        ("Привет, мир!", 2, "р!"),  # noqa: RUF001
    ],
)
def test_RIGHTB(text, num_chars, result):
    assert RIGHTB(text, num_chars) == result


@pytest.mark.parametrize(
    "text, start_num, num_chars, result",
    [
        ("Fluid Flow", 1, 5, "Fluid"),
        ("Fluid Flow", 7, 20, "Flow"),
        ("Fluid Flow", 20, 5, ""),
        ("100", 2, 2, "00"),
        ("String", 0, 3, VALUE_ERROR),
        ("String", 3, -1, VALUE_ERROR),
    ],
)
def test_MID(text, start_num, num_chars, result) -> str:
    assert MID(text, start_num, num_chars) == result


@pytest.mark.parametrize(
    "text, start_num, num_chars, result",
    [
        ("中国香港", 3, 4, "国香"),
        ("中国香港", 1, 24, "中国香港"),
        ("Fluid Flow", 7, 20, "Flow"),
        ("Fluid Flow", 20, 5, ""),
        ("100", 2, 2, "00"),
        ("String", 0, 3, VALUE_ERROR),
        ("String", 3, -1, VALUE_ERROR),
    ],
)
def test_MIDB(text, start_num, num_chars, result) -> str:
    assert MIDB(text, start_num, num_chars) == result


@pytest.mark.parametrize(
    "text, decimal_sep, group_sep, result",
    [
        ("2.500,27", ",", ".", 2500.27),
        ("3.5%", ".", ",", 0.035),
        ("    2.300,95       %%", ",", ".", 0.230095),
        ("10,15", ",", ".", 10.15),
        ("5%", ".", ",", 0.05),
        ("6.000", ",", ".", 6000),
        ("%%%", ".", ",", 0),
        ("", ".", ",", 0),
        ("4_250:50", ":", "_", 4250.50),
        ("1.200,70", ",", ".", 1200.7),
        ("    230095       %%", ".", "", VALUE_ERROR),
        ("    230095       %%", "", ",", VALUE_ERROR),
        (
            "            $                          0.05%%%               ",
            ".",
            ",",
            0.00000005,
        ),
    ],
)
def test_NUMBERVALUE(text, decimal_sep, group_sep, result):
    assert NUMBERVALUE(text, decimal_sep, group_sep) == result


@pytest.mark.parametrize("text, result", [("total", "TOTAL"), ("Yield", "YIELD")])
def test_UPPER(text, result):
    assert UPPER(text) == result


@pytest.mark.parametrize(
    "text, result", [("E. E. Cummings", "e. e. cummings"), ("Apt. 2B", "apt. 2b")]
)
def test_LOWER(text, result):
    assert LOWER(text) == result


@pytest.mark.parametrize(
    "text, result",
    [
        ("this is a TITLE", "This Is A Title"),
        ("2-way street", "2-Way Street"),
        ("76BudGet", "76Budget"),
    ],
)
def test_PROPER(text, result):
    assert PROPER(text) == result


ARRAYTOTEXT_TEST = CellRange(
    [[TRUE, VALUE_ERROR], [1234.01234, "Seattle"], ["Hello", 10]]
)


@pytest.mark.parametrize(
    "array, ret_format, result",
    [
        (ARRAYTOTEXT_TEST, 0, "TRUE, #VALUE!, 1234.01234, Seattle, Hello, 10"),
        (ARRAYTOTEXT_TEST, 1, '{TRUE,#VALUE!;1234.01234,"Seattle";"Hello",10}'),
        (ARRAYTOTEXT_TEST[0][0], 0, "TRUE"),
        (ARRAYTOTEXT_TEST[0][0], 1, "{TRUE}"),
        (ARRAYTOTEXT_TEST[2], 0, "Hello, 10"),
        (ARRAYTOTEXT_TEST[2], 1, '{"Hello",10}'),
    ],
)
def test_ARRAYTOTEXT(array, ret_format, result):
    assert ARRAYTOTEXT(array, ret_format) == result


@pytest.mark.parametrize(
    "number, result", [(65, "A"), (33, "!"), (0, VALUE_ERROR), (256, VALUE_ERROR)]
)
def test_CHAR(number, result):
    assert CHAR(number) == result


@pytest.mark.parametrize(
    "text, result",
    [
        (chr(9) + "Monthly report" + chr(10), "Monthly report"),
        ("", ""),
        ("Monthly report", "Monthly report"),
    ],
)
def test_CLEAN(text, result):
    assert CLEAN(text) == result


@pytest.mark.parametrize(
    "number, result",
    [
        (1234, "หนึ่งพันสองร้อยสามสิบสี่บาทถ้วน"),
        (132.3232, "หนึ่งร้อยสามสิบสองบาทสามสิบสองสตางค์"),
        (0, "ศูนย์บาทถ้วน"),
        (-1, "ลบหนึ่งบาทถ้วน"),
        (
            123123123.54543543543534,
            "หนึ่งร้อยยี่สิบสามล้านหนึ่งแสนสองหมื่นสามพันหนึ่งร้อยยี่สิบสามบาทห้าสิบห้าสตางค์",
        ),
    ],
)
def test_BAHTTEXT(number, result):
    assert BAHTTEXT(number) == result


@pytest.mark.parametrize(
    "reference, result",
    [
        ("さしすせそ", "ｻｼｽｾｿ"),
        (CellRange(["東京都"]), "東京都"),
        (CellRange([["じゃぱん1", "2", 3], ["4", "5", "6"]]), "ｼﾞｬﾊﾟﾝ1"),
    ],
)
def test_PHONETIC(reference, result):
    assert PHONETIC(reference) == result


@pytest.mark.parametrize(
    "text, number_times, result",
    [
        ("*-", 3, "*-*-*-"),
        ("-", 10, "----------"),
        ("*", -3, VALUE_ERROR),
        ("*", 0, ""),
    ],
)
def test_REPT(text, number_times, result):
    assert REPT(text, number_times) == result


@pytest.mark.parametrize(
    "value, result",
    [
        ("Rainfall", "Rainfall"),
        (19, ""),
        (TRUE, ""),
    ],
)
def test_T(value, result):
    assert T(value) == result


@pytest.mark.parametrize("number, result", [(66, "B"), (32, " "), (0, VALUE_ERROR)])
def test_UNICHAR(number, result):
    assert UNICHAR(number) == result


@pytest.mark.parametrize("text, result", [("B", 66), (" ", 32), ("じゃぱん", 12376)])
def test_UNICODE(text, result):
    assert UNICODE(text) == result


@pytest.mark.parametrize(
    "text, result",
    [
        ("", 0),
        ("hello", VALUE_ERROR),
        ("$1,000", 1000),
        ("16:48:00", 0.7),
        ("12:00:00", 0.5),
        ("2012-01-01", 40909),
        (NA_ERROR, NA_ERROR),
    ],
)
def test_VALUE(text, result):
    assert VALUE(text) == (
        result
        if isinstance(result, SpreadsheetError)
        else pytest.approx(result, rel=1e-3)
    )


@pytest.mark.parametrize(
    "value, ret_format, result",
    [
        (TRUE, 0, TRUE),
        (TRUE, 1, TRUE),
        (1234.01234, 0, 1234.01234),
        (1234.01234, 1, 1234.01234),
        ("Hello", 0, "Hello"),
        ("Hello", 1, '"Hello"'),
        (VALUE_ERROR, 0, VALUE_ERROR),
        (VALUE_ERROR, 1, VALUE_ERROR),
        ("Seattle", 0, "Seattle"),
        ("Seattle", 1, '"Seattle"'),
        (1234, 0, 1234),
        (1234, 1, 1234),
    ],
)
def test_VALUETOTEXT(value, ret_format, result):
    assert VALUETOTEXT(value, ret_format) == result


@pytest.mark.parametrize(
    "text, result",
    [
        (
            "\n\n\n\r\t                    hello           \r\n\t                           ",
            "hello",
        ),
    ],
)
def test_TRIM(text, result):
    assert TRIM(text) == result


TEXTJOIN_TEST1 = CellRange(
    [
        ["US Dollar"],
        ["Australian Dollar"],
        ["Chinese Yuan"],
        ["Hong Kong Dollar"],
        ["Israeli Shekel"],
        ["South Korean Won"],
        ["Russian Ruble"],
    ]
)
TEXTJOIN_TEST2 = CellRange(
    [
        ["a1", "b1"],
        ["a2", "b2"],
        ["", ""],
        ["a4", "b4"],
        ["a5", "b5"],
        ["a6", "b6"],
        ["a7", "b7"],
    ]
)
TEXTJOIN_TEST3 = CellRange(
    [
        ["Tulsa", "OK", "74133", "US"],
        ["Seattle", "WA", "98109", "US"],
        ["Iselin", "NJ", "08830", "US"],
        ["Fort Lauderdale", "FL", "33309", "US"],
        ["Tempe", "AZ", "85285", "US"],
        ["end", "", "", ""],
        [",", ",", ",", ";"],
    ]
)


@pytest.mark.parametrize(
    "args, result",
    [
        (
            (" ", TRUE, "The", "sun", "will", "come", "up", "tomorrow."),
            "The sun will come up tomorrow.",
        ),
        (
            (", ", TRUE, TEXTJOIN_TEST1),
            "US Dollar, Australian Dollar, Chinese Yuan, Hong Kong Dollar, Israeli Shekel, South Korean Won, Russian Ruble",
        ),
        (
            (", ", TRUE, TEXTJOIN_TEST2),
            "a1, b1, a2, b2, a4, b4, a5, b5, a6, b6, a7, b7",
        ),
        (
            (", ", FALSE, TEXTJOIN_TEST2),
            "a1, b1, a2, b2, , , a4, b4, a5, b5, a6, b6, a7, b7",
        ),
        (
            (TEXTJOIN_TEST3[6], TRUE, TEXTJOIN_TEST3[:6, :]),
            "Tulsa,OK,74133,US;Seattle,WA,98109,US;Iselin,NJ,08830,US;Fort Lauderdale,FL,33309,US;Tempe,AZ,85285,US;end",
        ),
    ],
)
def test_TEXTJOIN(args, result):
    delimiter, ignore_empty, text1, *texts = args
    assert TEXTJOIN(delimiter, ignore_empty, text1, *texts) == result


SEARCH_TEST1 = CellRange(
    [["Statements"], ["Profit Margin"], ["margin"], ['The "boss" is here.']]
)


@pytest.mark.parametrize(
    "find_text, within_text, start_num, result",
    [
        ("e", SEARCH_TEST1[0][0], 6, 7),
        (SEARCH_TEST1[2][0], SEARCH_TEST1[1][0], 1, 8),
        ('"', SEARCH_TEST1[3][0], 1, 5),
        ("o????n", "Excel function option tutolail", 1, 16),
        ("o*n", "Excel function option tutolail", 1, 13),
    ],
)
def test_SEARCH(find_text, within_text, start_num, result):
    assert SEARCH(find_text, within_text, start_num) == result


@pytest.mark.parametrize(
    "find_text, within_text, start_num, result",
    [
        ("ａ", "LibreOffice Ｃａｌｃ", 1, 15),  # noqa: RUF001
        ("国", "中国香港", 1, 3),
        ("国*", "中国香港", 1, 3),
    ],
)
def test_SEARCHB(find_text, within_text, start_num, result):
    assert SEARCHB(find_text, within_text, start_num) == result


@pytest.mark.parametrize(
    "value, format_text, result",
    [
        (12.34567, "###.##", "12.35"),
        (12, "000.00", "012.00"),
        (123456789, "#,###", "123,456,789"),
        (355, "YYYY-MM-DD HH:MM:SS", "1900-12-20 00:00:00"),
        (1.5, "# ?/?", "1 1/2"),
        ("xyz", "=== @ ===", "=== xyz ==="),
        (0.123456, "0.00%", "12.35%"),
        (123.456, "0.00E+#", "1.23E+2"),
        (-6789.4, "[$$-409]#,##0.00; -[$$-409]#,##0.00", " -$6,789.40"),
    ],
)
def test_TEXT_misc(value, format_text, result):
    assert TEXT(value, format_text) == result


@pytest.mark.parametrize(
    "value, format_text, result",
    [
        ("Hello", '"p"0;"m"0;"z"0;"t"@', "tHello"),
        ("Hello", '"num"0', "Hello"),
        (-1, '"p"0;"m"0;"z"0;"t"@', "m1"),
        (0, '"p"0;"m"0;"z"0;"t"@', "z0"),
        (1, '"p"0;"m"0;"z"0;"t"@', "p1"),
        (-1, '[<0]"p"0;"m"0;"z"0;"t"@', "p1"),
        (0, '[<0]"p"0;"m"0;"z"0;"t"@', "z0"),
        (1, '[<0]"p"0;"m"0;"z"0;"t"@', "z1"),
        (-1, '"p"0;[>0]"m"0;"z"0;"t"@', "-z1"),
        (0, '"p"0;[>0]"m"0;"z"0;"t"@', "z0"),
        (1, '"p"0;[>0]"m"0;"z"0;"t"@', "p1"),
        (-1, '[<0]"LT0";"ELSE"', "LT0"),
        (0, '[<0]"LT0";"ELSE"', "ELSE"),
        (1, '[<0]"LT0";"ELSE"', "ELSE"),
        (-1, '[<=0]"LTE0";"ELSE"', "LTE0"),
        (0, '[<=0]"LTE0";"ELSE"', "LTE0"),
        (1, '[<=0]"LTE0";"ELSE"', "ELSE"),
        (-1, '[>0]"GT0";"ELSE"', "ELSE"),
        (0, '[>0]"GT0";"ELSE"', "ELSE"),
        (1, '[>0]"GT0";"ELSE"', "GT0"),
        (-1, '[>=0]"GTE0";"ELSE"', "ELSE"),
        (0, '[>=0]"GTE0";"ELSE"', "GTE0"),
        (1, '[>=0]"GTE0";"ELSE"', "GTE0"),
        (-1, '[=0]"EQ0";"ELSE"', "ELSE"),
        (0, '[=0]"EQ0";"ELSE"', "EQ0"),
        (1, '[=0]"EQ0";"ELSE"', "ELSE"),
        (-1, '[<>0]"NEQ0";"ELSE"', "NEQ0"),
        (0, '[<>0]"NEQ0";"ELSE"', "ELSE"),
        (1, '[<>0]"NEQ0";"ELSE"', "NEQ0"),
    ],
)
def test_TEXT_condition(value, format_text, result):
    assert TEXT(value, format_text) == result


@pytest.mark.parametrize(
    "value, format_text, result",
    [
        (0, "??/??", " 0/1 "),
        (1.5, "??/??", " 3/2 "),
        (3.4, "??/??", "17/5 "),
        (4.3, "??/??", "43/10"),
        (0, "00/00", "00/01"),
        (1.5, "00/00", "03/02"),
        (3.4, "00/00", "17/05"),
        (4.3, "00/00", "43/10"),
        (0.00, '# ??/"a"?"a"0"a"', "0        a"),
        (0.10, '# ??/"a"?"a"0"a"', "0        a"),
        (0.12, '# ??/"a"?"a"0"a"', "  1/a8a0a"),
        (1.00, '# ??/"a"?"a"0"a"', "1        a"),
        (1.10, '# ??/"a"?"a"0"a"', "1  1/a9a0a"),
        (1.12, '# ??/"a"?"a"0"a"', "1  1/a8a0a"),
    ],
)
def test_TEXT_fraction_suffix(value, format_text, result):
    assert TEXT(value, format_text) == result


@pytest.mark.parametrize(
    "value, format_text, result",
    [
        (SpreadsheetDate(date(2000, 1, 1)), "d-mmm-yy", "1-Jan-00"),
        (
            SpreadsheetDateTime(
                datetime(2000, 1, 1, 12, 34, 56, tzinfo=SpreadsheetDateTime.TZ_INFO)
            ),
            "m/d/yyyy\\ h:mm:ss;@",
            "1/1/2000 12:34:56",
        ),
        (SpreadsheetDate(date(2010, 9, 26)), "yyyy-MMM-dd", "2010-Sep-26"),
        (SpreadsheetDate(date(2010, 9, 26)), "yyyy-MM-dd", "2010-09-26"),
        (SpreadsheetDate(date(2010, 9, 26)), "mm/dd/yyyy", "09/26/2010"),
        (SpreadsheetDate(date(2010, 9, 26)), "m/d/yy", "9/26/10"),
        (
            SpreadsheetDateTime(
                datetime(
                    2010, 9, 26, 12, 34, 56, 123000, tzinfo=SpreadsheetDateTime.TZ_INFO
                )
            ),
            "m/d/yy hh:mm:ss.000",
            "9/26/10 12:34:56.123",
        ),
        (
            SpreadsheetDateTime(
                datetime(
                    2010, 9, 26, 12, 34, 56, 123000, tzinfo=SpreadsheetDateTime.TZ_INFO
                )
            ),
            "YYYY-MM-DD HH:MM:SS",
            "2010-09-26 12:34:56",
        ),
        (
            SpreadsheetDateTime(
                datetime(2020, 1, 1, 14, 35, 55, tzinfo=SpreadsheetDateTime.TZ_INFO)
            ),
            "m/d/yyyy\\ h:mm:ss AM/PM;@",
            "1/1/2020 2:35:55 PM",
        ),
        (
            SpreadsheetDateTime(
                datetime(2020, 1, 1, 14, 35, 55, tzinfo=SpreadsheetDateTime.TZ_INFO)
            ),
            "m/d/yyyy\\ h:mm:ss aM/Pm;@",
            "1/1/2020 2:35:55 PM",
        ),
        (
            SpreadsheetDateTime(
                datetime(2020, 1, 1, 14, 35, 55, tzinfo=SpreadsheetDateTime.TZ_INFO)
            ),
            "m/d/yyyy\\ h:mm:ss am/PM;@",
            "1/1/2020 2:35:55 PM",
        ),
        (
            SpreadsheetDateTime(
                datetime(2020, 1, 1, 14, 35, 55, tzinfo=SpreadsheetDateTime.TZ_INFO)
            ),
            "m/d/yyyy\\ h:mm:ss A/P;@",
            "1/1/2020 2:35:55 P",
        ),
        (
            SpreadsheetDateTime(
                datetime(2020, 1, 1, 14, 35, 55, tzinfo=SpreadsheetDateTime.TZ_INFO)
            ),
            "m/d/yyyy\\ h:mm:ss a/P;@",
            "1/1/2020 2:35:55 p",
        ),
        (
            SpreadsheetDateTime(
                datetime(2020, 1, 1, 14, 35, 55, tzinfo=SpreadsheetDateTime.TZ_INFO)
            ),
            "m/d/yyyy\\ h:mm:ss A/p;@",
            "1/1/2020 2:35:55 P",
        ),
        (
            SpreadsheetDateTime(
                datetime(2020, 1, 1, 14, 35, 55, tzinfo=SpreadsheetDateTime.TZ_INFO)
            ),
            "m/d/yyyy\\ h:mm:ss;@",
            "1/1/2020 14:35:55",
        ),
        (
            SpreadsheetDateTime(
                datetime(2020, 1, 1, 14, 35, 55, tzinfo=SpreadsheetDateTime.TZ_INFO)
            ),
            "m/d/yyyy\\ hh:mm:ss AM/PM;@",
            "1/1/2020 02:35:55 PM",
        ),
        (
            SpreadsheetDateTime(
                datetime(2020, 1, 1, 16, 5, 6, tzinfo=SpreadsheetDateTime.TZ_INFO)
            ),
            "m/d/yyyy\\ h:m:s AM/PM;@",
            "1/1/2020 4:5:6 PM",
        ),
        (
            SpreadsheetDateTime(
                datetime(2017, 10, 16, 0, 0, 0, tzinfo=SpreadsheetDateTime.TZ_INFO)
            ),
            "dddd, MMMM d, yyyy",
            "Monday, October 16, 2017",
        ),
        (
            SpreadsheetDateTime(
                datetime(2017, 10, 16, 0, 0, 0, tzinfo=SpreadsheetDateTime.TZ_INFO)
            ),
            "dddd,,, MMMM d,, yyyy,,,,",
            "Monday, October 16, 2017,",
        ),
        (
            SpreadsheetDateTime(
                datetime(2020, 1, 1, 0, 35, 55, tzinfo=SpreadsheetDateTime.TZ_INFO)
            ),
            "m/d/yyyy\\ hh:mm:ss AM/PM;@",
            "1/1/2020 12:35:55 AM",
        ),
        (
            SpreadsheetDateTime(
                datetime(2020, 1, 1, 12, 35, 55, tzinfo=SpreadsheetDateTime.TZ_INFO)
            ),
            "m/d/yyyy\\ hh:mm:ss AM/PM;@",
            "1/1/2020 12:35:55 PM",
        ),
        # Date 1900
        ("0", "dd/mm/yyyy", "0"),
        (1, "dd/mm/yyyy", "01/01/1900"),
        (61, "dd/mm/yyyy", "01/03/1900"),
        (43648, "[$-409]d\\-mmm\\-yyyy;@", "2-Jul-2019"),
    ],
)
def test_TEXT_date(value, format_text, result):
    assert TEXT(value, format_text) == result


@pytest.mark.parametrize(
    "value, format_text, result",
    [
        # Fraction
        (1, "# ?/?", "1    "),
        (-1.2, "# ?/?", "-1 1/5"),
        (12.3, "# ?/?", "12 1/3"),
        (-12.34, "# ?/?", "-12 1/3"),
        (123.45, "# ?/?", "123 4/9"),
        (-123.456, "# ?/?", "-123 1/2"),
        (1234.567, "# ?/?", "1234 4/7"),
        (-1234.5678, "# ?/?", "-1234 4/7"),
        (12345.6789, "# ?/?", "12345 2/3"),
        (-12345.67891, "# ?/?", "-12345 2/3"),
        (1, "# ??/??", "1      "),
        (-1.2, "# ??/??", "-1  1/5 "),
        (12.3, "# ??/??", "12  3/10"),
        (-12.34, "# ??/??", "-12 17/50"),
        (123.45, "# ??/??", "123  9/20"),
        (-123.456, "# ??/??", "-123 26/57"),
        (1234.567, "# ??/??", "1234 55/97"),
        (-1234.5678, "# ??/??", "-1234 46/81"),
        (12345.6789, "# ??/??", "12345 55/81"),
        (-12345.67891, "# ??/??", "-12345 55/81"),
        (1, "# ???/???", "1        "),
        (-1.2, "# ???/???", "-1   1/5  "),
        (12.3, "# ???/???", "12   3/10 "),
        (-12.34, "# ???/???", "-12  17/50 "),
        (123.45, "# ???/???", "123   9/20 "),
        (-123.456, "# ???/???", "-123  57/125"),
        (1234.567, "# ???/???", "1234  55/97 "),
        (-1234.5678, "# ???/???", "-1234  67/118"),
        (12345.6789, "# ???/???", "12345  74/109"),
        (-12345.67891, "# ???/???", "-12345 573/844"),
        (1, "# ?/2", "1    "),
        (-1.2, "# ?/2", "-1    "),
        (12.3, "# ?/2", "12 1/2"),
        (-12.34, "# ?/2", "-12 1/2"),
        (123.45, "# ?/2", "123 1/2"),
        (-123.456, "# ?/2", "-123 1/2"),
        (1234.567, "# ?/2", "1234 1/2"),
        (-1234.5678, "# ?/2", "-1234 1/2"),
        (12345.6789, "# ?/2", "12345 1/2"),
        (-12345.67891, "# ?/2", "-12345 1/2"),
        (1, "# ?/4", "1    "),
        (-1.2, "# ?/4", "-1 1/4"),
        (12.3, "# ?/4", "12 1/4"),
        (-12.34, "# ?/4", "-12 1/4"),
        (123.45, "# ?/4", "123 2/4"),
        (-123.456, "# ?/4", "-123 2/4"),
        (1234.567, "# ?/4", "1234 2/4"),
        (-1234.5678, "# ?/4", "-1234 2/4"),
        (12345.6789, "# ?/4", "12345 3/4"),
        (-12345.67891, "# ?/4", "-12345 3/4"),
        (1, "# ?/8", "1    "),
        (-1.2, "# ?/8", "-1 2/8"),
        (12.3, "# ?/8", "12 2/8"),
        (-12.34, "# ?/8", "-12 3/8"),
        (123.45, "# ?/8", "123 4/8"),
        (-123.456, "# ?/8", "-123 4/8"),
        (1234.567, "# ?/8", "1234 5/8"),
        (-1234.5678, "# ?/8", "-1234 5/8"),
        (12345.6789, "# ?/8", "12345 5/8"),
        (-12345.67891, "# ?/8", "-12345 5/8"),
        (1, "# ??/16", "1      "),
        (-1.2, "# ??/16", "-1  3/16"),
        (12.3, "# ??/16", "12  5/16"),
        (-12.34, "# ??/16", "-12  5/16"),
        (123.45, "# ??/16", "123  7/16"),
        (-123.456, "# ??/16", "-123  7/16"),
        (1234.567, "# ??/16", "1234  9/16"),
        (-1234.5678, "# ??/16", "-1234  9/16"),
        (12345.6789, "# ??/16", "12345 11/16"),
        (-12345.67891, "# ??/16", "-12345 11/16"),
        (1, "# ?/10", "1     "),
        (-1.2, "# ?/10", "-1 2/10"),
        (12.3, "# ?/10", "12 3/10"),
        (-12.34, "# ?/10", "-12 3/10"),
        (123.45, "# ?/10", "123 5/10"),
        (-123.456, "# ?/10", "-123 5/10"),
        (1234.567, "# ?/10", "1234 6/10"),
        (-1234.5678, "# ?/10", "-1234 6/10"),
        (12345.6789, "# ?/10", "12345 7/10"),
        (-12345.67891, "# ?/10", "-12345 7/10"),
        (1, "# ??/100", "1       "),
        (-1.2, "# ??/100", "-1 20/100"),
        (12.3, "# ??/100", "12 30/100"),
        (-12.34, "# ??/100", "-12 34/100"),
        (123.45, "# ??/100", "123 45/100"),
        (-123.456, "# ??/100", "-123 46/100"),
        (1234.567, "# ??/100", "1234 57/100"),
        (-1234.5678, "# ??/100", "-1234 57/100"),
        (12345.6789, "# ??/100", "12345 68/100"),
        (-12345.67891, "# ??/100", "-12345 68/100"),
        (1, "??/??", " 1/1 "),
        (-1.2, "??/??", "- 6/5 "),
        (12.3, "??/??", "123/10"),
        (-12.34, "??/??", "-617/50"),
        (123.45, "??/??", "2469/20"),
        (-123.456, "??/??", "-7037/57"),
        (1234.567, "??/??", "119753/97"),
        (-1234.5678, "??/??", "-100000/81"),
        (12345.6789, "??/??", "1000000/81"),
        (-12345.67891, "??/??", "-1000000/81"),
        (0.3, "# ?/?", " 2/7"),
        (1.3, "# ?/?", "1 1/3"),
        (2.3, "# ?/?", "2 2/7"),
        (0.123251512342345, "# ??/?????????", " 480894/3901729  "),
        (0.123251512342345, "# ?? / ?????????", " 480894 / 3901729  "),
        (0.123251512342345, "# ??/?????????", " 480894/3901729  "),
        (0.123251512342345, "# ?? / ?????????", " 480894 / 3901729  "),
        (0, "0", "0"),
    ],
)
def test_TEXT_fraction(value, format_text, result):
    assert TEXT(value, format_text) == result


@pytest.mark.parametrize(
    "value, format_text, result",
    [
        (1469.07, "0,000,000.00", "0,001,469.07"),
    ],
)
def test_TEXT_thousands_separator(value, format_text, result):
    assert TEXT(value, format_text) == result


@pytest.mark.parametrize(
    "value, format_text, result",
    [
        (1234.56, "[$€-1809]# ##0.00", "€1 234.56"),
        (1234.56, "#,##0.00 [$EUR]", "1,234.56 EUR"),
    ],
)
def test_TEXT_currency(value, format_text, result):
    assert TEXT(value, format_text) == result


@pytest.mark.parametrize(
    "value, expected1, expected2, expected3, expected4",
    [
        (-1.23457e-13, "-12.3E-14", "-123.5E-15", "-1234.6E-16", "-123.5E-15"),
        (-12345.6789, "-1.2E+4", "-12.3E+3", "-1.2E+4", "-12345.7E+0"),
        (1.23457e-13, "12.3E-14", "123.5E-15", "1234.6E-16", "123.5E-15"),
        (1.23457e-12, "1.2E-12", "1.2E-12", "1.2E-12", "1234.6E-15"),
        (1.23457e-11, "12.3E-12", "12.3E-12", "12.3E-12", "12345.7E-15"),
        (1.23457e-10, "1.2E-10", "123.5E-12", "123.5E-12", "1.2E-10"),
        (1.23457e-09, "12.3E-10", "1.2E-9", "1234.6E-12", "12.3E-10"),
        (1.23457e-08, "1.2E-8", "12.3E-9", "1.2E-8", "123.5E-10"),
        (0.000000123457, "12.3E-8", "123.5E-9", "12.3E-8", "1234.6E-10"),
        (0.00000123457, "1.2E-6", "1.2E-6", "123.5E-8", "12345.7E-10"),
        (0.0000123457, "12.3E-6", "12.3E-6", "1234.6E-8", "1.2E-5"),
        (0.000123457, "1.2E-4", "123.5E-6", "1.2E-4", "12.3E-5"),
        (0.001234568, "12.3E-4", "1.2E-3", "12.3E-4", "123.5E-5"),
        (0.012345679, "1.2E-2", "12.3E-3", "123.5E-4", "1234.6E-5"),
        (0.123456789, "12.3E-2", "123.5E-3", "1234.6E-4", "12345.7E-5"),
        (1.23456789, "1.2E+0", "1.2E+0", "1.2E+0", "1.2E+0"),
        (12.3456789, "12.3E+0", "12.3E+0", "12.3E+0", "12.3E+0"),
        (123.456789, "1.2E+2", "123.5E+0", "123.5E+0", "123.5E+0"),
        (1234.56789, "12.3E+2", "1.2E+3", "1234.6E+0", "1234.6E+0"),
        (12345.6789, "1.2E+4", "12.3E+3", "1.2E+4", "12345.7E+0"),
        (123456.789, "12.3E+4", "123.5E+3", "12.3E+4", "1.2E+5"),
        (1234567.89, "1.2E+6", "1.2E+6", "123.5E+4", "12.3E+5"),
        (12345678.9, "12.3E+6", "12.3E+6", "1234.6E+4", "123.5E+5"),
        (123456789, "1.2E+8", "123.5E+6", "1.2E+8", "1234.6E+5"),
        (1234567890, "12.3E+8", "1.2E+9", "12.3E+8", "12345.7E+5"),
        (12345678900, "1.2E+10", "12.3E+9", "123.5E+8", "1.2E+10"),
        (123456789000, "12.3E+10", "123.5E+9", "1234.6E+8", "12.3E+10"),
        (1234567890000, "1.2E+12", "1.2E+12", "1.2E+12", "123.5E+10"),
        (12345678900000, "12.3E+12", "12.3E+12", "12.3E+12", "1234.6E+10"),
        (123456789000000, "1.2E+14", "123.5E+12", "123.5E+12", "12345.7E+10"),
        (1234567890000000, "12.3E+14", "1.2E+15", "1234.6E+12", "1.2E+15"),
        (12345678900000000, "1.2E+16", "12.3E+15", "1.2E+16", "12.3E+15"),
        (123456789000000000, "12.3E+16", "123.5E+15", "12.3E+16", "123.5E+15"),
        (1234567890000000000, "1.2E+18", "1.2E+18", "123.5E+16", "1234.6E+15"),
        (12345678900000000000, "12.3E+18", "12.3E+18", "1234.6E+16", "12345.7E+15"),
        (123456789000000000000, "1.2E+20", "123.5E+18", "1.2E+20", "1.2E+20"),
        (1234567890000000000000, "12.3E+20", "1.2E+21", "12.3E+20", "12.3E+20"),
        (12345678900000000000000, "1.2E+22", "12.3E+21", "123.5E+20", "123.5E+20"),
        (123456789000000000000000, "12.3E+22", "123.5E+21", "1234.6E+20", "1234.6E+20"),
        (1234567890000000000000000, "1.2E+24", "1.2E+24", "1.2E+24", "12345.7E+20"),
        (12345678900000000000000000, "12.3E+24", "12.3E+24", "12.3E+24", "1.2E+25"),
        (123456789000000000000000000, "1.2E+26", "123.5E+24", "123.5E+24", "12.3E+25"),
        (
            1234567890000000000000000000,
            "12.3E+26",
            "1.2E+27",
            "1234.6E+24",
            "123.5E+25",
        ),
        (12345678900000000000000000000, "1.2E+28", "12.3E+27", "1.2E+28", "1234.6E+25"),
        (
            123456789000000000000000000000,
            "12.3E+28",
            "123.5E+27",
            "12.3E+28",
            "12345.7E+25",
        ),
        (1234567890000000000000000000000, "1.2E+30", "1.2E+30", "123.5E+28", "1.2E+30"),
        (
            12345678900000000000000000000000,
            "12.3E+30",
            "12.3E+30",
            "1234.6E+28",
            "12.3E+30",
        ),
    ],
)
def test_TEXT_exponents(value, expected1, expected2, expected3, expected4):
    assert TEXT(value, "#0.0E+0") == expected1
    assert TEXT(value, "##0.0E+0") == expected2
    assert TEXT(value, "###0.0E+0") == expected3
    assert TEXT(value, "####0.0E+0") == expected4


@pytest.mark.parametrize(
    "value, expected1, expected2, expected3, expected4, expected5, expected6, expected7",
    [
        (0.99, ".0000", ".0000", ".0010", "1.0", "1", "1", ".99"),
        (1.2345, ".0000", ".0000", ".0012", "1.2", "1", "1", "1.23"),
        (12.345, ".0000", ".0000", ".0123", "12.3", "12", "12", "12.35"),
        (123.456, ".0000", ".0001", ".1235", "123.5", "123", "123", "123.46"),
        (1234, ".0000", ".0012", "1.2340", "1,234.0", "1,234", "1,234", "1,234.00"),
        (
            12345,
            ".0000",
            ".0123",
            "12.3450",
            "12,345.0",
            "12,345",
            "12,345",
            "12,345.00",
        ),
        (
            123456,
            ".0001",
            ".1235",
            "123.4560",
            "123,456.0",
            "123,456",
            "123,456",
            "123,456.00",
        ),
        (
            1234567,
            ".0012",
            "1.2346",
            "1234.5670",
            "1,234,567.0",
            "1,234,567",
            "1,234,567",
            "1,234,567.00",
        ),
        (
            12345678,
            ".0123",
            "12.3457",
            "12345.6780",
            "12,345,678.0",
            "12,345,678",
            "12,345,678",
            "12,345,678.00",
        ),
        (
            123456789,
            ".1235",
            "123.4568",
            "123456.7890",
            "123,456,789.0",
            "123,456,789",
            "123,456,789",
            "123,456,789.00",
        ),
        (
            1234567890,
            "1.2346",
            "1234.5679",
            "1234567.8900",
            "1,234,567,890.0",
            "1,234,567,890",
            "1,234,567,890",
            "1,234,567,890.00",
        ),
        (
            12345678901,
            "12.3457",
            "12345.6789",
            "12345678.9010",
            "12,345,678,901.0",
            "12,345,678,901",
            "12,345,678,901",
            "12,345,678,901.00",
        ),
        (
            123456789012,
            "123.4568",
            "123456.7890",
            "123456789.0120",
            "123,456,789,012.0",
            "123,456,789,012",
            "123,456,789,012",
            "123,456,789,012.00",
        ),
        (4321, ".0000", ".0043", "4.3210", "4,321.0", "4,321", "4,321", "4,321.00"),
        (
            4321234,
            ".0043",
            "4.3212",
            "4321.2340",
            "4,321,234.0",
            "4,321,234",
            "4,321,234",
            "4,321,234.00",
        ),
    ],
)
def test_TEXT_commas(
    value, expected1, expected2, expected3, expected4, expected5, expected6, expected7
):
    assert TEXT(value, "#.0000,,,") == expected1
    assert TEXT(value, "#.0000,,") == expected2
    assert TEXT(value, "#.0000,") == expected3
    assert TEXT(value, "#,##0.0") == expected4
    assert TEXT(value, "###,##0") == expected5
    assert TEXT(value, "###,###") == expected6
    assert TEXT(value, "#,###.00") == expected7


@pytest.mark.parametrize(
    "value, expected1, expected2, expected3, expected4, expected5",
    [
        (0.0, " . ", "  .  ", "   .   ", "   . 0 ", "   .  "),
        (0.1, " .1", "  .1 ", "   .1  ", "   .10 ", "   .1 "),
        (0.12, " .1", "  .12", "   .12 ", "   .12 ", "   .12 "),
        (0.123, " .1", "  .12", "   .123", "   .123", "   .123"),
        (1.0, "1. ", " 1.  ", "  1.   ", "  1. 0 ", "  1.  "),
        (1.1, "1.1", " 1.1 ", "  1.1  ", "  1.10 ", "  1.1 "),
        (1.12, "1.1", " 1.12", "  1.12 ", "  1.12 ", "  1.12 "),
        (1.123, "1.1", " 1.12", "  1.123", "  1.123", "  1.123"),
    ],
)
def test_TEXT_number(value, expected1, expected2, expected3, expected4, expected5):
    assert TEXT(value, "?.?") == expected1
    assert TEXT(value, "??.??") == expected2
    assert TEXT(value, "???.???") == expected3
    assert TEXT(value, "???.?0?") == expected4
    assert TEXT(value, "???.?#?") == expected5


@pytest.mark.parametrize(
    "value, format_text, result",
    [
        (0, "[hh]:mm", "00:00"),
        (1, "[hh]:mm", "24:00"),
        (1.5, "[hh]:mm", "36:00"),
    ],
)
def test_TEXT_duration(value, format_text, result):
    assert TEXT(value, format_text) == result


@pytest.mark.parametrize(
    "value, format_text, result",
    [
        (timedelta(hours=100), "[hh]:mm:ss", "100:00:00"),
        (timedelta(hours=100), "[mm]:ss", "6000:00"),
        (
            timedelta(milliseconds=100 * 60 * 60 * 1000 + 123),
            "[mm]:ss.000",
            "6000:00.123",
        ),
        (timedelta(days=1, hours=2, minutes=31, seconds=45), "[hh]:mm:ss", "26:31:45"),
        (
            timedelta(days=1, hours=2, minutes=31, seconds=44, milliseconds=500),
            "[hh]:mm:ss",
            "26:31:45",
        ),
        (
            timedelta(days=1, hours=2, minutes=31, seconds=44, milliseconds=500),
            "[hh]:mm:ss.000",
            "26:31:44.500",
        ),
        (
            timedelta(days=-1, hours=-2, minutes=-31, seconds=-45),
            "[hh]:mm:ss",
            "-26:31:45",
        ),
        (
            timedelta(days=0, hours=-2, minutes=-31, seconds=-45),
            "[hh]:mm:ss",
            "-02:31:45",
        ),
        (
            timedelta(days=0, hours=-2, minutes=-31, seconds=-44, milliseconds=-500),
            "[hh]:mm:ss",
            "-02:31:45",
        ),
        (
            timedelta(days=0, hours=-2, minutes=-31, seconds=-44, milliseconds=-500),
            "[hh]:mm:ss.000",
            "-02:31:44.500",
        ),
    ],
)
def test_TEXT_duration_from_timedelta(value, format_text, result):
    assert TEXT(value.total_seconds() / 86400, format_text) == result
