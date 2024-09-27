import datetime
import re
from itertools import cycle

import jaconv
from bahttext import bahttext

from ..cell_range import CellRange
from ..spreadsheet_error import VALUE_ERROR, SpreadsheetError
from .boolean import FALSE, BooleanValue
from .date_time import SpreadsheetDate, SpreadsheetDateTime, SpreadsheetTime
from .helpers import (
    CellValue,
    Numeric,
    SimpleCellValue,
    _flatten_range,
    re_from_wildcard,
    round_to_decimals,
    to_number,
)
from .text_formatter import formatter

__all__ = [
    "ASC",
    "ARRAYTOTEXT",
    "BAHTTEXT",
    "CHAR",
    "CLEAN",
    "CODE",
    "CONCAT",
    "CONCATENATE",
    "DBCS",
    "DOLLAR",
    "EXACT",
    "FIND",
    "FINDB",
    "FIXED",
    "LEFT",
    "LEFTB",
    "LEN",
    "LENB",
    "LOWER",
    "MID",
    "MIDB",
    "NUMBERVALUE",
    "PHONETIC",
    "PROPER",
    "REPLACE",
    "REPLACEB",
    "REPT",
    "RIGHT",
    "RIGHTB",
    "SEARCH",
    "SEARCHB",
    "SUBSTITUTE",
    "TEXT",
    "TEXTJOIN",
    "TRIM",
    "UNICHAR",
    "UNICODE",
    "UPPER",
    "VALUE",
    "VALUETOTEXT",
]


_decimal_sep = "."
_group_sep = ","


def _num_dbcs_bytes(c: str) -> int:
    return 2 if len(c.encode()) > 2 else 1


def ASC(text: str) -> str:
    """Changes full-width (double-byte) English letters or katakana within a character string to half-width (single-byte) characters"""
    return jaconv.z2h(text)


def ARRAYTOTEXT(array: CellValue, ret_format: int = 0):
    """Returns an array of text values from any specified range"""
    if ret_format not in [0, 1]:
        return VALUE_ERROR
    if ret_format == 0:
        if isinstance(array, CellRange):
            return ", ".join(str(el) for el in _flatten_range(array))
        else:
            return str(array)
    else:
        if isinstance(array, CellRange):
            if isinstance(array[0], CellRange):
                text_array = []
                for row in array:
                    text_array.append(
                        ",".join(
                            f'"{el}"' if isinstance(el, str) else str(el) for el in row
                        )
                    )
                return "{" + ";".join(text_array) + "}"
            else:
                return (
                    "{"
                    + ",".join(
                        f'"{el}"' if isinstance(el, str) else str(el) for el in array
                    )
                    + "}"
                )
        else:
            return (
                "{" + f'"{array}"' + "}"
                if isinstance(array, str)
                else "{" + str(array) + "}"
            )


def BAHTTEXT(number: int | float) -> str:
    """Converts a number to text, using the ß (baht) currency format"""
    return bahttext(number)


def CHAR(number: int) -> str:
    """Returns the character specified by the code number"""
    if 1 <= number <= 255:
        return chr(number)
    else:
        return VALUE_ERROR


def CLEAN(text: str) -> str:
    """Removes all nonprintable characters from text"""
    return "".join(ch for ch in text if ord(ch) >= 32)


def CODE(text: str) -> int:
    """Returns a numeric code for the first character in a text string"""
    return ord(text[0]) if text else VALUE_ERROR


def CONCAT(text1: CellValue, *texts: tuple[CellValue]) -> str:
    result = ""
    for txt in [text1, *texts]:
        result += (
            "".join(str(t) for t in _flatten_range(txt))
            if isinstance(txt, CellRange)
            else str(txt)
        )
    return result


def CONCATENATE(text1: str, *texts: tuple[str]) -> str:
    """Joins several text items into one text item"""
    return "".join(value for value in [text1, *texts])


def DBCS(text: str) -> str:
    """Changes half-width (single-byte) English letters or katakana within a character string to full-width (double-byte) characters"""
    return jaconv.h2z(text)


def DOLLAR(number: Numeric, decimals: int = 2) -> str:
    """Converts a number to text, using the $ (dollar) currency format"""
    number, decimals = round_to_decimals(number, decimals)

    result = f"${abs(number):,.{decimals}f}"
    if number < 0:
        result = f"({result})"
    return result


def EXACT(text1: str, text2: str):
    """Checks to see if two text values are identical"""
    return BooleanValue(text1 == text2)


def _find(
    find_text: str, within_text: str, start_num: int = 1, wildcard_allowed: bool = False
):
    """Finds one text value within another"""
    if start_num >= len(within_text) or start_num < 1:
        return VALUE_ERROR
    if not find_text:
        return 1
    if wildcard_allowed and "*" in find_text or "?" in find_text:
        regexp = re_from_wildcard(find_text)
        for match in re.finditer(regexp, within_text):
            s = match.span()
            return s[0] + 1

    result = within_text.find(find_text, start_num - 1)
    if result == -1:
        return VALUE_ERROR
    return result + 1


def _findb(
    find_text: str, within_text: str, start_num: int = 1, wildcard_allowed: bool = False
):
    """Finds one text value within another"""
    if start_num < 1:
        return VALUE_ERROR
    if not find_text:
        return 1
    idx = 0
    i = 0
    while idx < start_num - 1:
        idx += _num_dbcs_bytes(within_text[i])
        i += 1
    if idx >= len(within_text):
        return VALUE_ERROR

    result = -1
    if wildcard_allowed and "*" in find_text or "?" in find_text:
        regexp = re_from_wildcard(find_text)
        for match in re.finditer(regexp, within_text):
            s = match.span()
            result = s[0]
            break

    if result == -1:
        result = within_text.find(find_text, i)
    if result == -1:
        return VALUE_ERROR

    return idx + sum(_num_dbcs_bytes(ch) for ch in within_text[i:result]) + 1


def FIND(find_text: str, within_text: str, start_num: int = 1) -> int:
    """Finds one text value within another (case-sensitive)"""
    return _find(find_text, within_text, start_num)


def FINDB(find_text: str, within_text: str, start_num: int = 1) -> int:
    """Finds one text value within another (case-sensitive)"""
    return _findb(find_text, within_text, start_num)


def FIXED(number: Numeric, decimals: int = 2, no_commas: BooleanValue = FALSE) -> str:
    """"""
    number, decimals = round_to_decimals(number, decimals)
    comma = "" if no_commas else ","
    return f"{number:{comma}.{decimals}f}"


def LEFT(text: str, num_chars: int = 1) -> str:
    """Returns the leftmost characters from a text value"""
    if num_chars == 0:
        return ""
    elif num_chars < 0:
        return VALUE_ERROR
    return text[:num_chars]


def _leftb(text: str, num_chars: int = 1) -> str:
    if num_chars == 0:
        return ""
    elif num_chars < 0:
        return VALUE_ERROR

    if not text:
        return ""
    i = -1

    if num_chars >= _lenb(text):
        return text

    while i < len(text):
        i += 1
        if num_chars == 0:
            return text[:i]
        if num_chars == -1:
            return text[: i - 1] + " "
        num_chars -= _num_dbcs_bytes(text[i])


def LEFTB(text: str, num_chars: int = 1) -> str:
    """Returns the leftmost characters from a text value"""
    return _leftb(text, num_chars)


def LEN(text: str) -> int:
    """Returns the number of characters in a text string"""
    return len(text)


def _lenb(text: str) -> int:
    return sum(_num_dbcs_bytes(t) for t in text)


def LENB(text: str) -> int:
    """Returns the number of characters in a text string"""
    return _lenb(text)


def LOWER(text: str) -> str:
    """Converts text to lowercase"""
    return text.lower()


def MID(text: str, start_num: int, num_chars: int) -> str:
    """Returns a specific number of characters from a text string starting at the position you specify"""
    if start_num < 1 or num_chars < 0:
        return VALUE_ERROR
    return text[start_num - 1 : start_num - 1 + num_chars]


def MIDB(text: str, start_num: int, num_chars: int) -> str:
    """Returns a specific number of characters from a text string starting at the position you specify"""
    if start_num < 1 or num_chars < 0:
        return VALUE_ERROR

    text = _leftb(text, start_num + num_chars - 1)
    ncnt = _lenb(text) - start_num + 1
    return _rightb(text, max(ncnt, 0))


def _str2number(
    text: str, decimal_sep: str = _decimal_sep, group_sep: str = _group_sep
) -> Numeric:
    text = "".join(ch for ch in text if ch not in [" ", "\t", "\n", "\r"])
    if text.startswith("$"):
        text = text[1:]
    idx_percent = -1
    while abs(idx_percent) <= len(text) and text[idx_percent] == "%":
        idx_percent -= 1
    text = text[: len(text) + idx_percent + 1]

    if len(decimal_sep) > 1:
        decimal_sep = decimal_sep[0]
    text_ar = text.split(decimal_sep)
    if len(group_sep) > 1:
        group_sep = group_sep[0]
    text_ar[0] = text_ar[0].replace(group_sep, "")

    text = decimal_sep.join(text_ar).replace(decimal_sep, ".")

    fac = 10 ** ((idx_percent + 1) * 2)
    if not text:
        return 0
    value = to_number(text) * fac
    return value


def NUMBERVALUE(
    text: str, decimal_sep: str = _decimal_sep, group_sep: str = _group_sep
) -> Numeric:
    """Converts text to number in a locale-independent manner"""
    if not decimal_sep:
        return VALUE_ERROR
    if not group_sep:
        return VALUE_ERROR

    return _str2number(text, decimal_sep, group_sep)


def PHONETIC(reference: str | CellRange) -> str:
    """Extracts the phonetic (furigana) characters from a text string"""
    if isinstance(reference, CellRange):
        if isinstance(reference[0], CellRange):
            value = str(reference[0][0])
        else:
            value = str(reference[0])
    else:
        value = reference
    return jaconv.hira2hkata(value)


def PROPER(text: str) -> str:
    """Capitalizes the first letter in each word of a text value"""
    return text.title()


def REPLACE(old_text: str, start_num: int, num_chars: int, new_text: str) -> str:
    """Replaces part of a text string, based on the number of characters you specify, with a different text string"""
    if start_num < 1 or num_chars < 0:
        return VALUE_ERROR
    return old_text[: start_num - 1] + new_text + old_text[start_num - 1 + num_chars :]


def REPLACEB(old_text: str, start_num: int, num_chars: int, new_text: str) -> str:
    """Replaces part of a text string, based on the number of characters you specify, with a different text string"""
    nlen = _lenb(old_text)
    if (
        start_num < 1
        or start_num > nlen
        or num_chars < 0
        or start_num + num_chars - 1 > nlen
    ):
        return VALUE_ERROR
    head = _leftb(old_text, start_num - 1)
    tail = _rightb(old_text, nlen - start_num - num_chars + 1)
    return head + new_text + tail


def REPT(text: str, number_times: int) -> str:
    """Repeats text a given number of times"""
    if number_times < 0:
        return VALUE_ERROR
    if not number_times:
        return ""
    return text * number_times


def RIGHT(text: str, num_chars: int = 1) -> str:
    """Returns the rightmost characters from a text value"""
    if num_chars == 0:
        return ""
    elif num_chars < 0:
        return VALUE_ERROR
    return text[-num_chars:]


def _rightb(text: str, num_chars: int = 1) -> str:
    """Returns the rightmost characters from a text value"""
    if num_chars == 0:
        return ""
    elif num_chars < 0:
        return VALUE_ERROR

    if not text:
        return ""

    if num_chars >= _lenb(text):
        return text

    i = len(text)

    while i >= 0:
        i -= 1
        if num_chars == 0:
            return text[i + 1 :]
        if num_chars == -1:
            return " " + text[i + 2 :]
        num_chars -= _num_dbcs_bytes(text[i])


def RIGHTB(text: str, num_chars: int = 1) -> str:
    """Returns the rightmost characters from a text value"""
    return _rightb(text, num_chars)


def SEARCH(find_text: str, within_text: str, start_num: int = 1) -> int:
    """Finds one text value within another (not case-sensitive)"""
    return _find(find_text.lower(), within_text.lower(), start_num, True)


def SEARCHB(find_text: str, within_text: str, start_num: int = 1) -> int:
    """Finds one text value within another (not case-sensitive)"""
    return _findb(find_text.lower(), within_text.lower(), start_num, True)


def SUBSTITUTE(text: str, old_text: str, new_text: str, instance_num=None) -> str:
    """Substitutes new text for old text in a text string"""
    if instance_num is None:
        return text.replace(old_text, new_text)
    try:
        instance_num = int(instance_num)
    except ValueError:
        return VALUE_ERROR
    start = text.find(old_text)
    while start >= 0 and instance_num > 1:
        start = text.find(old_text, start + len(old_text))
        instance_num -= 1
    if start > -1:
        return text[:start] + new_text + text[start + len(old_text) :]
    else:
        return text


# TODO add TEXT function
# T function is located in stats.py


def TEXTJOIN(
    delimiter: str | CellRange,
    ignore_empty: BooleanValue,
    text1: str | CellRange,
    *texts: tuple[str | CellRange],
) -> str:
    """Combines the text from multiple ranges and/or strings, and includes a delimiter you specify between each text value that will be combined. If the delimiter is an empty text string, this function will effectively concatenate the ranges."""

    seps = cycle(delimiter if isinstance(delimiter, CellRange) else (delimiter,))
    txts = ""
    for t in [text1, *texts]:
        if isinstance(t, CellRange):
            for tt in _flatten_range(t):
                if tt or not ignore_empty:
                    txts += str(tt)
                    sep = next(seps)
                    txts += sep
        elif t or not ignore_empty:
            txts += str(t)
            sep = next(seps)
            txts += sep
    return txts[: -len(sep)]


def TRIM(text: str) -> str:
    """Removes spaces from text"""
    return text.strip()


def UNICHAR(number: int) -> str:
    """Returns the Unicode character that is references by the given numeric value"""
    if not number:
        return VALUE_ERROR
    try:
        return chr(number)
    except Exception as e:
        return SpreadsheetError.from_python_exception(e, "", None)


def UNICODE(text: str) -> int:
    """Returns the Unicode character that is references by the given numeric value"""
    if not text:
        return VALUE_ERROR
    try:
        return ord(text[0])
    except ValueError:
        return VALUE_ERROR


def UPPER(text: str) -> str:
    """Converts text to uppercase"""
    return text.upper()


def _str2datetime_serial(text: str) -> Numeric:
    """Converts date/time string to serial representation"""
    try:
        datetime.date.fromisoformat(text)
        return SpreadsheetDate(text)
    except ValueError:
        try:
            datetime.time.fromisoformat(text)
            return SpreadsheetTime(text)
        except ValueError:
            try:
                return SpreadsheetDateTime(text)
            except ValueError:
                return VALUE_ERROR


def VALUE(text: str) -> CellValue:
    """Converts a text argument to a number"""
    if not isinstance(text, str):
        return text

    result = _str2number(text)
    if not isinstance(result, SpreadsheetError):
        return result

    return _str2datetime_serial(text)


def VALUETOTEXT(value: SimpleCellValue, ret_format: int = 0) -> str:
    """Returns text from any specified value"""
    if ret_format not in [0, 1]:
        return VALUE_ERROR
    if isinstance(value, str) and ret_format == 1:
        return f'"{value}"'
    return value


def TEXT(value: Numeric, format_text: str) -> str:
    """Formats a number and converts it to text"""
    return formatter.run(value, format_text)
