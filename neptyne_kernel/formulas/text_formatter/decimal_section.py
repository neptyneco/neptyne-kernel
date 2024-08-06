from dataclasses import dataclass
from typing import Optional

from .token import is_placeholder


@dataclass
class DecimalSection:
    thousand_sep: bool
    thousand_divisor: float
    percent_multiplier: float
    before_decimal: list[str]
    decimal_sep: bool
    after_decimal: list[str]

    @classmethod
    def try_parse(cls, tokens: list[str]) -> Optional["DecimalSection"]:
        from .parser import parse_number_tokens

        if (parse_result := parse_number_tokens(tokens)) and parse_result[0] == (
            len(tokens) - 1
        ):
            before_decimal, decimal_sep, after_decimal = parse_result[1:]
            divisor, thousand_sep = DecimalSection.get_trailing_commas_divisor(tokens)
            multiplier = DecimalSection.get_percent_multiplier(tokens)

            return cls(
                thousand_sep,
                divisor,
                multiplier,
                before_decimal,
                decimal_sep,
                after_decimal,
            )

    @staticmethod
    def get_percent_multiplier(tokens: list[str]) -> float:
        # If there is a percentage literal in the part list, multiply the result by 100
        for token in tokens:
            if token == "%":
                return 100
        return 1

    @staticmethod
    def get_trailing_commas_divisor(tokens: list[str]) -> tuple[float, bool]:
        # This parses all comma literals in the part list:
        # Each comma after the last digit placeholder divides the result by 1000.
        # If there are any other commas, display the result with thousand separators.

        has_last_placeholder = False
        divisor = 1.0

        for j, _ in enumerate(tokens):
            token_ind = len(tokens) - 1 - j
            token = tokens[token_ind]

            if not has_last_placeholder:
                if is_placeholder(token):
                    # Each trailing comma multiplies the divisor by 1000
                    for k in range(token_ind + 1, len(tokens)):
                        token = tokens[k]
                        if token == ",":
                            divisor *= 1000.0
                        else:
                            break

                    # Continue scanning backwards from the last digit placeholder,
                    # but now look for a thousand separator comma
                    has_last_placeholder = True

            elif token == ",":
                return divisor, True

        return divisor, False
