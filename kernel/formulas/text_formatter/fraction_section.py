from dataclasses import dataclass
from typing import Optional

from .token import is_digit_09, is_digit_19, is_placeholder


@dataclass
class FractionSection:
    integer_part: list[str]
    numerator: list[str]
    denominator_prefix: list[str]
    denominator: list[str]
    denominator_constant: int
    denominator_suffix: list[str]
    fraction_suffix: list[str]

    @classmethod
    def try_parse(cls, tokens: list[str]) -> Optional["FractionSection"]:
        numerator_parts: list[str] = []
        denominator_parts: list[str] = []

        for i, _ in enumerate(tokens):
            part = tokens[i]
            if part == "/":
                numerator_parts = tokens[:i]
                denominator_parts = tokens[i + 1 :]
                break

        if not numerator_parts:
            return

        integer_part, numerator_part = FractionSection.get_numerator(numerator_parts)

        (
            success,
            denominator_prefix,
            denominator_part,
            denominator_constant,
            denominator_suffix,
            fraction_suffix,
        ) = FractionSection.try_get_denominator(denominator_parts)
        if not success:
            return

        return cls(
            integer_part,
            numerator_part,
            denominator_prefix,
            denominator_part,
            denominator_constant,
            denominator_suffix,
            fraction_suffix,
        )

    @staticmethod
    def get_numerator(tokens: list[str]) -> tuple[list[str], list[str]]:
        has_placeholder = False
        has_space = False
        has_integer_part = False
        numerator_ind = -1
        ind = len(tokens) - 1
        while ind >= 0:
            token = tokens[ind]
            if is_placeholder(token):
                has_placeholder = True
                if has_space:
                    has_integer_part = True
                    break
            elif has_placeholder and not has_space:
                # First time we get here marks the end of the integer part
                has_space = True
                numerator_ind = ind + 1
            ind -= 1

        return (
            (tokens[:numerator_ind], tokens[numerator_ind:])
            if has_integer_part
            else (None, tokens)
        )

    @staticmethod
    def try_get_denominator(
        tokens: list[str],
    ) -> tuple[bool, list[str], list[str], int, list[str], list[str]]:
        ind = 0
        has_placeholder = False
        has_constant = False

        constant = []

        # Read literals until the first number placeholder or digit
        while ind < len(tokens):
            token = tokens[ind]
            if is_placeholder(token):
                has_placeholder = True
                break
            elif is_digit_19(token):
                has_constant = True
                break
            ind += 1

        if not has_placeholder and not has_constant:
            return False, [], [], 0, [], []

        # The denominator starts here, keep the index

        denominator_ind = ind

        # Read placeholders or digits in sequence
        while ind < len(tokens):
            token = tokens[ind]
            if not (has_placeholder and is_placeholder(token)):
                if has_constant and is_digit_09(token):
                    constant.append(token)
                else:
                    break
            ind += 1

        # 'index' is now at the first token after the denominator placeholders.
        # The remaining, if anything, is to be treated in one or two parts:
        # Any ultimately terminating literals are considered the "Fraction suffix".
        # Anything between the denominator and the fraction suffix is the "Denominator suffix".
        # Placeholders in the denominator suffix are treated as insignificant zeros.

        # Scan backwards to determine the fraction suffix
        fraction_suffix_ind = len(tokens)
        while fraction_suffix_ind > ind:
            token = tokens[fraction_suffix_ind - 1]
            if is_placeholder(token):
                break

            fraction_suffix_ind -= 1

        # Finally, extract the detected token ranges

        denominator_prefix = tokens[:denominator_ind] if denominator_ind > 0 else []
        denominator_constant = int("".join(constant)) if has_constant else 0
        denominator_part = tokens[denominator_ind:ind]
        denominator_suffix = (
            tokens[ind:fraction_suffix_ind] if ind < fraction_suffix_ind else []
        )
        fraction_suffix = (
            tokens[fraction_suffix_ind:] if fraction_suffix_ind < len(tokens) else []
        )

        return (
            True,
            denominator_prefix,
            denominator_part,
            denominator_constant,
            denominator_suffix,
            fraction_suffix,
        )
