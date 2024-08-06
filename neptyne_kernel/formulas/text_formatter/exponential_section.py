from dataclasses import dataclass
from typing import Optional

from .token import is_exponent


@dataclass
class ExponentialSection:
    before_decimal: list[str]
    decimal_sep: bool
    after_decimal: list[str]
    exponential_token: str
    power: list[str]

    @classmethod
    def try_parse(cls, tokens: list[str]) -> Optional["ExponentialSection"]:
        from .parser import parse_number_tokens

        (
            part_count,
            before_decimal,
            decimal_sep,
            after_decimal,
        ) = parse_number_tokens(tokens)

        if part_count == 0:
            return

        position = part_count
        if position < len(tokens) and is_exponent(tokens[position]):
            exponential_token = tokens[position]
            position += 1
        else:
            return

        return cls(
            before_decimal,
            decimal_sep,
            after_decimal,
            exponential_token,
            tokens[position:],
        )
