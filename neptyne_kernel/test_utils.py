from .cell_address import Address, Range


def a1(s: str) -> Address | Range:
    if ":" in s:
        return Range.from_a1(s)
    else:
        return Address.from_a1(s)
