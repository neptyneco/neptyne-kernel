from .condition import Condition
from .decimal_section import DecimalSection
from .exponential_section import ExponentialSection
from .fraction_section import FractionSection
from .section import Section
from .section_type import SectionType
from .token import (
    is_date_part,
    is_duration_part,
    is_general,
    is_number_literal,
    is_placeholder,
)
from .tokenizer import Tokenizer


def parse_sections(format_str: str) -> tuple[list[Section], bool]:
    tokenizer = Tokenizer(format_str)
    sections = []
    syntax_error = False
    while True:
        section, section_syntax_error = parse_section(tokenizer, len(sections))

        if section_syntax_error:
            syntax_error = True
        if not section:
            break

        sections.append(section)

    return sections, syntax_error


def parse_section(reader: Tokenizer, index: int) -> tuple[Section | None, bool]:
    has_date_parts: bool = False
    has_duration_parts: bool = False
    has_general_part: bool = False
    has_text_part: bool = False
    has_placeholders: bool = False
    condition: Condition | None = None
    color: str = ""
    tokens: list[str] = []

    syntax_error: bool = False
    while (read_token_result := read_token(reader)) and read_token_result[0]:
        token, syntax_error = read_token_result
        if token == ";":
            break

        has_placeholders |= is_placeholder(token)

        if is_date_part(token):
            has_date_parts = True
            has_duration_parts |= is_duration_part(token)
            tokens.append(token)
        elif is_general(token):
            has_general_part = True
            tokens.append(token)
        elif token == "@":
            has_text_part = True
            tokens.append(token)
        elif token.startswith("["):
            # Does not add to tokens. Absolute/elapsed time tokens
            # also start with '[', but handled as date part above
            expression = token[1:-1]
            parse_condition = try_parse_condition(expression)
            if parse_condition[0]:
                condition = parse_condition[1]
            elif (parse_color := try_parse_color(expression)) and parse_color[0]:
                color = parse_color[1]
            elif (
                parse_currency_symbol := try_parse_currency_symbol(expression)
            ) and parse_currency_symbol[0]:
                tokens.append('"' + parse_currency_symbol[1] + '"')
        else:
            tokens.append(token)

    if syntax_error or not tokens:
        return None, syntax_error

    if (
        (has_date_parts and (has_general_part or has_text_part))
        or (has_general_part and (has_date_parts or has_text_part))
        or (has_text_part and (has_general_part or has_date_parts))
    ):
        # Cannot mix date, general and/or text parts
        return None, True

    fraction: FractionSection = None
    exponential: ExponentialSection = None
    number: DecimalSection = None
    general_text_date_duration: list[str] = []

    if has_date_parts:
        section_type = SectionType.Duration if has_duration_parts else SectionType.Date
        general_text_date_duration = parse_milliseconds(tokens)
    elif has_general_part:
        section_type = SectionType.General
        general_text_date_duration = tokens
    elif has_text_part or not has_placeholders:
        section_type = SectionType.Text
        general_text_date_duration = tokens
    elif fraction := FractionSection.try_parse(tokens):
        section_type = SectionType.Fraction
    elif exponential := ExponentialSection.try_parse(tokens):
        section_type = SectionType.Exponential
    elif number := DecimalSection.try_parse(tokens):
        section_type = SectionType.Number
    else:
        # Unable to parse format string
        return None, True

    return (
        Section(
            section_type=section_type,
            index=index,
            color=color,
            condition=condition,
            fraction=fraction,
            exponential=exponential,
            number=number,
            general_text_date_duration_parts=general_text_date_duration,
        ),
        syntax_error,
    )


# Parses as many placeholders and literals needed to format a number with optional decimals.
# Returns number of tokens parsed, or 0 if the tokens didn't form a number.


def parse_number_tokens(tokens: list[str]) -> tuple[int, list[str], bool, list[str]]:
    before_decimal = []
    after_decimal = []
    decimal_sep = False

    remainder: list[str] = []
    for index, _ in enumerate(tokens):
        token = tokens[index]
        if token == "." and not before_decimal:
            decimal_sep = True
            before_decimal = tokens[:index]
            remainder = []
        elif is_number_literal(token):
            remainder.append(token)
        elif not token.startswith("["):
            break

    if remainder:
        if before_decimal:
            after_decimal = remainder
        else:
            before_decimal = remainder

    return index, before_decimal, decimal_sep, after_decimal


def parse_milliseconds(tokens: list[str]) -> list[str]:
    # if tokens form .0 through .000.., combine to single subsecond token
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == ".":
            zeros = 0
            while i + 1 < len(tokens) and tokens[i + 1] == "0":
                i += 1
                zeros += 1

            if zeros > 0:
                result.append("." + "".join(["0"] * zeros))
            else:
                result.append(".")
        else:
            result.append(token)
        i += 1
    return result


def read_token(reader: Tokenizer) -> tuple[str, bool]:
    offset = reader.pos
    if (
        read_literal(reader)
        or reader.read_enclosed("[", "]")
        or
        # Symbols
        reader.read_one_of("=#?,!&%+-$€£0123456789{}():;/.@ ")
        or reader.read_string("e+", True)
        or reader.read_string("e-", True)
        or reader.read_string("General", True)
        or
        # Date
        reader.read_string("am/pm", True)
        or reader.read_string("a/p", True)
        or reader.read_one_or_more("y")
        or reader.read_one_or_more("Y")
        or reader.read_one_or_more("m")
        or reader.read_one_or_more("M")
        or reader.read_one_or_more("d")
        or reader.read_one_or_more("D")
        or reader.read_one_or_more("h")
        or reader.read_one_or_more("H")
        or reader.read_one_or_more("s")
        or reader.read_one_or_more("S")
        or reader.read_one_or_more("g")
        or reader.read_one_or_more("G")
    ):
        syntax_error = False
        length = reader.pos - offset
        return reader.substring(offset, length), syntax_error

    syntax_error = reader.pos < reader.length()
    return "", syntax_error


def read_literal(reader: Tokenizer) -> bool:
    peek = reader.peek()
    if peek == "\\" or peek == "*" or peek == "_":
        reader.advance(2)
        return True
    elif reader.read_enclosed('"', '"'):
        return True
    return False


def try_parse_condition(token: str) -> tuple[bool, Condition | None]:
    tokenizer = Tokenizer(token)

    if (
        tokenizer.read_string("<=")
        or tokenizer.read_string("<>")
        or tokenizer.read_string("<")
        or tokenizer.read_string(">=")
        or tokenizer.read_string(">")
        or tokenizer.read_string("=")
    ):
        condition_pos = tokenizer.pos
        op = tokenizer.substring(0, condition_pos)

        if read_condition_value(tokenizer):
            value_str = tokenizer.substring(
                condition_pos, tokenizer.pos - condition_pos
            )
            result = Condition(op=op, value=float(value_str))
            return True, result

    return False, None


def read_condition_value(tokenizer: Tokenizer) -> bool:
    tokenizer.read_string("-")
    while tokenizer.read_one_of("0123456789"):
        pass

    if tokenizer.read_string("."):
        while tokenizer.read_one_of("0123456789"):
            pass

    if tokenizer.read_string("e+", True) or tokenizer.read_string("e-", True):
        if tokenizer.read_one_of("0123456789"):
            while tokenizer.read_one_of("0123456789"):
                pass
        else:
            return False

    return True


def try_parse_color(token: str) -> tuple[bool, str | None]:
    tokenizer = Tokenizer(token)
    if (
        tokenizer.read_string("black", True)
        or tokenizer.read_string("blue", True)
        or tokenizer.read_string("cyan", True)
        or tokenizer.read_string("green", True)
        or tokenizer.read_string("magenta", True)
        or tokenizer.read_string("red", True)
        or tokenizer.read_string("white", True)
        or tokenizer.read_string("yellow", True)
    ):
        return True, tokenizer.substring(0, tokenizer.pos)

    return False, None


def try_parse_currency_symbol(token: str) -> tuple[bool, str | None]:
    if not token or not token.startswith("$"):
        return False, None

    return True, token[1 : token.index("-")] if "-" in token else token[1:]
