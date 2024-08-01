LITERAL_TOKENS = (
    ",",
    "!",
    "&",
    "%",
    "+",
    "-",
    "$",
    "€",
    "£",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "{",
    "}",
    "(",
    ")",
    " ",
)
LITERAL_TOKENS_START = ("_", "\\", '"', "*")
DATE_PART_TOKENS_START = ("y", "m", "d", "s", "h")


def is_exponent(token: str) -> bool:
    token = token.lower()
    return token in ["e+", "e-"]


def is_literal(token: str):
    return token.startswith(LITERAL_TOKENS_START) or token in LITERAL_TOKENS


def is_number_literal(token: str):
    return is_placeholder(token) or is_literal(token) or token == "."


def is_placeholder(token: str):
    return token in ("0", "#", "?")


def is_general(token: str):
    return token.lower() == "general"


def is_date_part(token: str):
    ltoken = token.lower()
    return (
        ltoken.startswith(DATE_PART_TOKENS_START)
        or (ltoken.startswith("g") and not is_general(ltoken))
        or ltoken == "am/pm"
        or ltoken == "a/p"
        or is_duration_part(ltoken)
    )


def is_duration_part(token: str):
    ltoken = token.lower()
    return len(ltoken) >= 2 and ltoken[0] == "[" and ltoken[1] in ("h", "m", "s")


def is_digit_09(token: str):
    return token.isdigit()


def is_digit_19(token: str) -> bool:
    return is_digit_09(token) and token != "0"
