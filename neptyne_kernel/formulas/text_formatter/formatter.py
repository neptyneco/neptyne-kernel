import calendar
from datetime import datetime, timedelta
from math import floor, log10
from typing import Any

from ...spreadsheet_datetime import excel2datetime
from ..helpers import round_half_up
from .decimal_section import DecimalSection
from .evaluator import get_section
from .number_format import NumberFormat
from .section import Section, SectionType
from .token import is_date_part, is_general, is_placeholder

# Formatter is adapted from the C# repo: https://github.com/andersnm/ExcelNumberFormat


def run(value: Any, node: [str | Section]):
    if isinstance(node, str):
        fmt = NumberFormat(node)
        if not fmt.is_valid:
            return str(value)

        node = get_section(fmt.sections, value)
        if not node:
            return str(value)

    match node.section_type:
        case SectionType.Number:
            # Hide sign under certain conditions and section index
            number = float(value)
            if (node.index == 0 and node.condition) or node.index == 1:
                number = abs(number)

            return format_number_str(number, node.number)

        case SectionType.Date:
            d = excel2datetime(value)
            return format_date(d, node.general_text_date_duration_parts)

        case SectionType.Duration:
            if isinstance(value, timedelta):
                return format_timespan(value, node.general_text_date_duration_parts)
            else:
                return format_timespan(
                    timedelta(days=value), node.general_text_date_duration_parts
                )

        case SectionType.General | SectionType.Text:
            return format_general_text(
                str(value), node.general_text_date_duration_parts
            )

        case SectionType.Exponential:
            return format_exponential(float(value), node)

        case SectionType.Fraction:
            return format_fraction(float(value), node)


def look_ahead_date_part(tokens: list[str], from_ind: int, starts_with: str) -> bool:
    starts_with = starts_with.lower()
    for token in tokens[from_ind:]:
        if token.lower().startswith(starts_with):
            return True
        if is_date_part(token):
            return False
    return False


def look_back_date_part(tokens: list[str], from_ind: int, starts_with: str) -> bool:
    starts_with = starts_with.lower()
    for token in tokens[from_ind::-1]:
        if token.lower().startswith(starts_with):
            return True
        if is_date_part(token):
            return False
    return False


def contains_ampm(tokens: list[str]) -> bool:
    for token in tokens:
        token = token.lower()
        if token == "am/pm" or token == "a/p":
            return True
    return False


def format_date(date: datetime, tokens: list[str]) -> str:
    has_ampm = contains_ampm(tokens)
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        ltoken = token.lower()
        if ltoken.startswith("y"):
            # year
            digits = len(ltoken)
            if digits < 2:
                digits = 2
            elif digits == 3:
                digits = 4

            year = date.year
            if digits == 2:
                year = year % 100

            result.append(f"{year:>0{digits}}")

        elif ltoken.startswith("m"):
            # If  "m" or "mm" code is used immediately after the "h" or "hh" code (for hours) or immediately before
            # the "ss" code (for seconds), the application shall display minutes instead of the month.
            digits = len(ltoken)
            if look_back_date_part(tokens, i - 1, "h") or look_ahead_date_part(
                tokens, i + 1, "s"
            ):
                result.append(f"{date.minute:>0{digits}}")
            else:
                if digits == 3:
                    month = calendar.month_abbr[date.month]
                elif digits == 4:
                    month = calendar.month_name[date.month]
                elif digits == 5:
                    month = calendar.month_name[date.month][0]
                else:
                    month = f"{date.month:>0{digits}}"
                result.append(month)

        elif ltoken.startswith("d"):
            digits = len(ltoken)
            if digits == 3:
                # Sun-Sat
                day = calendar.day_abbr[date.weekday()]
            elif digits == 4:
                # Sunday-Saturday
                day = calendar.day_name[date.weekday()]
            else:
                day = f"{date.day:>0{digits}}"
            result.append(day)
        elif ltoken.startswith("h"):
            digits = len(ltoken)
            hour = (
                f"{(date.hour + 11) % 12 + 1:>0{digits}}"
                if has_ampm
                else f"{date.hour:>0{digits}}"
            )
            result.append(hour)

        elif ltoken.startswith("s"):
            result.append(f"{date.second:>0{len(ltoken)}}")

        elif ltoken == "am/pm":
            result.append(date.strftime("%p").upper())
        elif ltoken == "a/p":
            ampm = date.strftime("%p")[0]
            result.append(ampm.upper() if token[0].isupper() else ampm.lower())
        elif ltoken.startswith(".0"):
            value = date.microsecond // 1000
            result.append(f".{value:>0{len(token) - 1}}")
        elif token == "/":
            result.append(token)
        elif token == ",":
            while i < len(tokens) - 1 and tokens[i + 1] == ",":
                i += 1
            result.append(token)
        else:
            format_literal(token, result)
        i += 1
    return "".join(result)


def format_general_text(text: str, tokens: list[str]) -> str:
    result = []
    for i, token in enumerate(tokens):
        if is_general(token) or token == "@":
            result.append(text)
        else:
            format_literal(token, result)
    return "".join(result)


def format_number_str(value: float, fmt: DecimalSection) -> str:
    thousand_sep = fmt.thousand_sep
    value = value / fmt.thousand_divisor
    value = value * fmt.percent_multiplier

    result = []
    format_number(
        value,
        fmt.before_decimal,
        fmt.decimal_sep,
        fmt.after_decimal,
        thousand_sep,
        result,
    )
    return "".join(result)


def format_number(
    value: float,
    before_decimal: list[str],
    decimal_separator: bool,
    after_decimal: list[str],
    thousand_sep: bool,
    result: list[str],
):
    signitificant_digits = 0
    if after_decimal:
        signitificant_digits = get_digit_count(after_decimal)

    values = f"{abs(value):.{signitificant_digits}f}".split(".")
    thousands_str = values[0]
    decimal_str = values[1].rstrip("0") if len(values) > 1 else ""

    if value < 0:
        result.append("-")

    if before_decimal:
        format_thousands(thousands_str, thousand_sep, False, before_decimal, result)

    if decimal_separator:
        result.append(".")

    if after_decimal:
        format_decimals(decimal_str, after_decimal, result)


def format_thousands(
    value_str: str,
    thousand_sep: bool,
    significant_zero: bool,
    tokens: list[str],
    result: list[str],
):
    significant = False
    format_digits = get_digit_count(tokens)
    value_str = f"{value_str:>0{format_digits}}"

    # Print literals occurring before any placeholders
    token_ind = 0
    while token_ind < len(tokens):
        token = tokens[token_ind]
        if is_placeholder(token):
            break
        else:
            format_literal(token, result)
        token_ind += 1

    # Print value digits until there are as many digits remaining as there are placeholders
    digit_ind = 0
    while digit_ind < (len(value_str) - format_digits):
        significant = True
        result.append(value_str[digit_ind])

        if thousand_sep:
            format_thousands_separator(value_str, digit_ind, result)
        digit_ind += 1
    # Print remaining value digits and format literals
    for token in tokens[token_ind:]:
        if is_placeholder(token):
            c = value_str[digit_ind]
            if c != "0" or (significant_zero and digit_ind == len(value_str) - 1):
                significant = True

            format_placeholder(token, c, significant, result)

            if thousand_sep and (significant or token == "0"):
                format_thousands_separator(value_str, digit_ind, result)

            digit_ind += 1
        else:
            format_literal(token, result)


def format_thousands_separator(value_str: str, digit: int, result: list[str]):
    position_in_tens = len(value_str) - 1 - digit
    if position_in_tens > 0 and (position_in_tens % 3) == 0:
        result.append(",")


def format_decimals(value_str: str, tokens: list[str], result: list[str]):
    unpadded_digits = len(value_str)
    format_digits = get_digit_count(tokens)
    value_str = f"{value_str:<0{format_digits}}"

    # Print all format digits
    value_ind = 0
    for tokenIndex, token in enumerate(tokens):
        if is_placeholder(token):
            c = value_str[value_ind]
            significant = value_ind < unpadded_digits
            format_placeholder(token, c, significant, result)
            value_ind += 1
        else:
            format_literal(token, result)


def format_exponential(value: float, fmt: Section):
    # The application shall display a number to the right of
    # the "E" symbol that corresponds to the number of places that
    # the decimal point was moved.

    base_digits = 0
    if fmt.exponential.before_decimal:
        base_digits = get_digit_count(fmt.exponential.before_decimal)

    exponent = floor(log10(abs(value)))
    mantissa = value / 10**exponent

    shift = abs(exponent) % base_digits
    if shift > 0:
        if exponent < 0:
            shift = base_digits - shift

        mantissa *= 10**shift
        exponent -= shift

    result = []
    format_number(
        mantissa,
        fmt.exponential.before_decimal,
        fmt.exponential.decimal_sep,
        fmt.exponential.after_decimal,
        False,
        result,
    )

    result.append(fmt.exponential.exponential_token[0])

    if fmt.exponential.exponential_token[1] == "+" and exponent >= 0:
        result.append("+")

    elif exponent < 0:
        result.append("-")

    format_thousands(str(abs(exponent)), False, False, fmt.exponential.power, result)
    return "".join(result)


def format_fraction(value: float, fmt: Section) -> str:
    integral = 0

    sign = value < 0

    if fmt.fraction.integer_part is not None:
        integral = int(value)
        value = abs(value - integral)

    if fmt.fraction.denominator_constant:
        denominator = fmt.fraction.denominator_constant
        rr = round(value * denominator)
        b = floor(rr / denominator)
        numerator = int(rr - b * denominator)
    else:
        denominatorDigits = min(get_digit_count(fmt.fraction.denominator), 7)
        numerator, denominator = get_fraction(value, 10**denominatorDigits - 1)

    # Don't hide fraction if at least one zero in the numerator format
    numerator_zeros = get_zero_count(fmt.fraction.numerator)
    hideFraction = (
        fmt.fraction.integer_part is not None
        and numerator == 0
        and numerator_zeros == 0
    )

    result = []

    if sign:
        result.append("-")

    # Print integer part with significant zero if fraction part is hidden
    if fmt.fraction.integer_part is not None:
        format_thousands(
            f"{abs(integral)}",
            False,
            hideFraction,
            fmt.fraction.integer_part,
            result,
        )

    numerator_str = f"{abs(numerator)}"
    denominator_str = f"{abs(denominator)}"

    fraction = []

    format_thousands(numerator_str, False, True, fmt.fraction.numerator, fraction)

    fraction.append("/")

    if fmt.fraction.denominator_prefix:
        format_thousands("", False, False, fmt.fraction.denominator_prefix, fraction)

    if fmt.fraction.denominator_constant != 0:
        fraction.append(str(fmt.fraction.denominator_constant))
    else:
        format_denominator(denominator_str, fmt.fraction.denominator, fraction)

    if fmt.fraction.denominator_suffix:
        format_thousands("", False, False, fmt.fraction.denominator_suffix, fraction)

    if hideFraction:
        result.append(" " * sum(len(f) for f in fraction))
    else:
        result.append("".join(fraction))

    if fmt.fraction.fraction_suffix:
        format_thousands("", False, False, fmt.fraction.fraction_suffix, result)

    return "".join(result)


def get_fraction(x: float, D: int) -> tuple[int, int]:
    sgn = -1 if x < 0 else 1
    B = x * sgn
    P_2 = 0.0
    P_1 = 1.0
    P = 0.0
    Q_2 = 1.0
    Q_1 = 0.0
    Q = 0.0
    A = floor(B)
    while Q_1 < D:
        A = floor(B)
        P = A * P_1 + P_2
        Q = A * Q_1 + Q_2
        if (B - A) < 0.00000005:
            break
        B = 1 / (B - A)
        P_2 = P_1
        P_1 = P
        Q_2 = Q_1
        Q_1 = Q
    if Q > D:
        if Q_1 > D:
            Q = Q_2
            P = P_2
        else:
            Q = Q_1
            P = P_1

    return int(sgn * P), int(Q)


def format_denominator(value_str: str, tokens: list[str], result: list[str]):
    format_digits = get_digit_count(tokens)
    value_str = f"{value_str:>0{format_digits}}"

    significant = False
    value_ind = 0
    for token_ind, token in enumerate(tokens):
        if value_ind < len(value_str):
            c, value_ind = get_left_aligned_value_digit(
                token, value_str, value_ind, significant
            )
            if c != "0":
                significant = True
        else:
            c = "0"
            significant = False

        format_placeholder(token, c, significant, result)


def get_left_aligned_value_digit(
    token: str, value_str: str, start_ind: int, significant: bool
) -> tuple[str, int]:
    value_ind = start_ind
    if value_ind < len(value_str):
        c = value_str[value_ind]
        value_ind += 1
        if c != "0":
            significant = True

        if token == "?" and not significant:
            # Eat insignificant zeros to left align denominator
            while value_ind < len(value_str):
                c = value_str[value_ind]
                value_ind += 1
                if c != "0":
                    break
    else:
        c = "0"

    return c, value_ind


def format_placeholder(token: str, c: str, significant: bool, result: list[str]):
    if token == "0":
        if significant:
            result.append(c)
        else:
            result.append("0")
    elif token == "#":
        if significant:
            result.append(c)
    elif token == "?":
        if significant:
            result.append(c)
        else:
            result.append(" ")


def get_digit_count(tokens: list[str]) -> int:
    return sum(is_placeholder(token) for token in tokens)


def get_zero_count(tokens: list[str]) -> int:
    return tokens.count("0")


def format_literal(token: str, result: list[str]):
    literal = ""
    if token != ",":
        # skip commas
        if len(token) == 2 and (token[0] == "*" or token[0] == "\\"):
            literal = token[1]
        elif len(token) == 2 and token[0] == "_":
            literal = " "
        elif token.startswith('"'):
            literal = token[1:-1]
        else:
            literal = token
    result.append(literal)


def format_timespan(timespan: timedelta, tokens: list[str]) -> str:
    result = []
    contains_milliseconds = False
    for token in tokens[::-1]:
        if token.startswith(".0"):
            contains_milliseconds = True
            break

    for token in tokens:
        ltoken = token.lower()
        if ltoken.startswith("m"):
            value = timespan.seconds // 60 % 60
            result.append(f"{value:0{len(token)}d}")
        elif ltoken.startswith("s"):
            # If format does not include ms, then include ms in seconds and round before printing
            format_ms = 0 if contains_milliseconds else timespan.microseconds / 1e6
            value = round_half_up(timespan.seconds % 60 + format_ms)
            result.append(f"{value:0{len(token)}d}")
        elif ltoken.startswith("[h"):
            if timespan.days < 0:
                timespan = -timespan
                sgn = "-"
            else:
                sgn = ""
            total_sec = timespan.total_seconds()
            value = int(total_sec) // 3600
            result.append(f"{sgn}{value:0{len(token) - 2}d}")
            timespan = timedelta(seconds=total_sec % 3600)
        elif ltoken.startswith("[m"):
            value = int(timespan.total_seconds()) // 60
            result.append(f"{value:0{len(token) - 2}d}")
            timespan = timedelta(
                seconds=timespan.seconds % 60, microseconds=timespan.microseconds
            )
        elif ltoken.startswith("[s"):
            value = int(timespan.total_seconds())
            result.append(f"{value:0{len(token) - 2}d}")
            timespan = timedelta(microseconds=timespan.microseconds)
        elif token.startswith(".0"):
            value = timespan.microseconds // 1000
            result.append(f".{value:0{len(token) - 1}d}")
        else:
            format_literal(token, result)
    return "".join(result)
