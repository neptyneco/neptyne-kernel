import json
import re
from dataclasses import dataclass, replace
from typing import Any, Iterator, Union

CoordAddr = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Address:
    """Address points to a cell within a sheet. It can be parsed from A1 notation, but always
    refers to a specific sheet by unique ID, so will infer the default sheet ID if none is
    specified. It is a frozen dataclass so that it can be used as a key in the Dash map"""

    column: int
    row: int
    sheet: int

    @classmethod
    def from_r1c1(cls, r1c1: str, sheet: int = 0) -> "Address":
        if "!" in r1c1:
            raise ValueError("Address can only be instantiated from sheet-unaware R1C1")
        s = re.fullmatch(r"^R(?P<row>\d+)C(?P<col>\d+)$", r1c1)
        if not s:
            raise ValueError(f"Invalid address: {r1c1}")
        row = int(s.group("row"))
        col = int(s.group("col"))
        if row < 1 or col < 1:
            raise ValueError(f"Invalid address: {r1c1}")

        return cls(col - 1, row - 1, sheet=sheet)

    @classmethod
    def from_a1(cls, a1: str, sheet: int = 0) -> "Address":
        if "!" in a1:
            raise ValueError("Address can only be instantiated from sheet-unaware A1")

        try:
            x, y = parse_cell(a1)
        except ValueError:
            raise ValueError(f"Invalid address: {a1}")
        return cls(x, y, sheet=sheet)

    def to_a1(self) -> str:
        return format_cell(self.column, self.row)

    @classmethod
    def from_a1_or_str(cls, a1: str, sheet: int = 0) -> "Address":
        if a1.startswith("["):
            coords = json.loads(a1)
            assert not sheet
            return cls(*coords)

        return cls.from_a1(a1, sheet)

    def to_cell_id(self) -> str:
        return str([self.column, self.row, self.sheet])

    def to_coord(self) -> CoordAddr:
        return self.column, self.row, self.sheet

    @classmethod
    def from_coord(cls, coord: CoordAddr) -> "Address":
        return Address(*coord)

    @classmethod
    def from_list(cls, coords: list[float]) -> "Address":
        # This exists to satisfy Quicktype's insistence that all numbers are floats
        x, y, sheet = coords
        return cls(int(x), int(y), int(sheet))

    def to_float_coord(self) -> list[float]:
        # This exists to satisfy Quicktype's insistence that all numbers are floats
        return [*self.to_coord()]

    def __str__(self) -> str:
        return self.to_cell_id()

    @staticmethod
    def is_address(value: str) -> bool:
        try:
            parsed = json.loads(value)
            return (
                isinstance(parsed, list)
                and len(parsed) == 3
                and all(isinstance(v, int) for v in parsed)
            )
        except json.JSONDecodeError:
            return False

    def __lt__(self, other: "Address") -> bool:
        return (self.sheet, self.column, self.row) < (
            other.sheet,
            other.column,
            other.row,
        )

    def __post_init__(self) -> None:
        # quicktype/json round trips can make floats out of ints. Ensure this class always uses
        # ints.
        # we need object.__setattr__ because this is a frozen dataclass
        object.__setattr__(self, "column", int(self.column))
        object.__setattr__(self, "row", int(self.row))
        object.__setattr__(self, "sheet", int(self.sheet))


@dataclass(frozen=True)
class Range:
    """Represents an immutable excel style range as numbers. To mimic excel most accurately, max_* are inclusive\n\n
    For infinite ranges, -1 is used as the max_*\n

    """

    min_col: int
    """The minimum column in the range"""
    max_col: int
    """The maximum column in the range. -1 for infinite"""
    min_row: int
    """The minimum row in the range"""
    max_row: int
    """The maximum row in the range. -1 for infinite"""

    sheet: int
    """@private"""

    def __post_init__(self) -> None:
        # we need object.__setattr__ because this is a frozen dataclass
        object.__setattr__(self, "_shape", None)
        object.__setattr__(self, "_dim", None)

    def __iter__(self) -> Iterator[list[Address]]:
        for row in range(self.min_row, self.max_row + 1):
            yield [
                Address(col, row, self.sheet)
                for col in range(self.min_col, self.max_col + 1)
            ]

    def __contains__(self, other: Union[Address, "Range"]) -> bool:
        if other.sheet != self.sheet:
            return False
        if isinstance(other, Address):
            return (
                other.row >= self.min_row
                and (other.row <= self.max_row or self.max_row == -1)
                and other.column >= self.min_col
                and (other.column <= self.max_col or self.max_col == -1)
            )

        return (
            other.origin() in self
            and (
                self.max_col == -1
                or (other.max_col != -1 and self.max_col >= other.max_col)
            )
            and (
                self.max_row == -1
                or (other.max_row != -1 and self.max_row >= other.max_row)
            )
        )

    def is_fully_bounded(self) -> bool:
        """Returns True if the range is fully bounded, i.e. has a definite max_col and max_row."""
        return self.max_col != -1 and self.max_row != -1

    def is_bounded(self) -> bool:
        """Returns True if the range is bounded, i.e. has a definite max_col or max_row."""
        return self.max_col != -1 or self.max_row != -1

    def origin(self) -> Address:
        """Returns the top-left cell of the range."""
        return Address(self.min_col, self.min_row, self.sheet)

    def translated(self, dx: int, dy: int) -> "Range":
        """Returns a new range translated by dx columns and dy rows."""
        return Range(
            self.min_col + dx,
            self.max_col + dx,
            self.min_row + dy,
            self.max_row + dy,
            self.sheet,
        )

    def extended(self, dx: int, dy: int) -> "Range":
        """Returns a new range extended by dx columns and dy rows."""
        return Range(
            self.min_col,
            self.max_col + dx if self.max_col != -1 else -1,
            self.min_row,
            self.max_row + dy if self.max_row != -1 else -1,
            self.sheet,
        )

    def intersects(self, other: "Range") -> bool:
        """Returns True if the range intersects with another range."""
        if self.sheet != other.sheet:
            return False

        def overlaps_dimension(
            self_min: int, self_max: int, other_min: int, other_max: int
        ) -> bool:
            if self_max == -1 and other_max == -1:
                return True
            elif self_max == -1:
                return other_max >= self_min
            elif other_max == -1:
                return self_max >= other_min
            return self_min <= other_min <= self_max or (
                other_min <= self_min <= other_max
            )

        return overlaps_dimension(
            self.min_col, self.max_col, other.min_col, other.max_col
        ) and overlaps_dimension(
            self.min_row, self.max_row, other.min_row, other.max_row
        )

    def shape(self) -> tuple[int, int]:
        """Returns the shape of the range as a tuple of (n_rows, n_cols)."""
        if self._shape is None:  # type: ignore
            w = self.max_col
            if w != -1:
                w += 1 - self.min_col
            h = self.max_row
            if h != -1:
                h += 1 - self.min_row
            object.__setattr__(self, "_shape", (h, w))
        return self._shape  # type: ignore

    def dimensions(self) -> int:
        """Returns the number of dimensions of the range. 1 if either n_rows or n_cols is 1, 2 otherwise."""
        if self._dim is None:  # type: ignore
            object.__setattr__(
                self, "_dim", sum(d > 1 or d == -1 for d in self.shape())
            )
        return self._dim  # type: ignore

    @classmethod
    def from_address(cls, address: Address) -> "Range":
        """@private"""
        return cls(
            address.column,
            address.column,
            address.row,
            address.row,
            address.sheet,
        )

    @classmethod
    def from_addresses(cls, addr1: Address, addr2: Address) -> "Range":
        """@private"""
        assert addr1.sheet == addr2.sheet
        # Gsheets will let you do something like B1:A4 and correct it to A1:B4, so we support that.
        col_key = None if addr1.column >= 0 and addr2.column >= 0 else lambda x: -x
        row_key = None if addr1.row >= 0 and addr2.row >= 0 else lambda x: -x
        return cls(
            min((addr1.column, addr2.column), key=col_key),
            max((addr1.column, addr2.column), key=col_key),
            min((addr1.row, addr2.row), key=row_key),
            max((addr1.row, addr2.row), key=row_key),
            addr1.sheet,
        )

    @classmethod
    def from_a1(cls, a1: str, sheet: int = 0) -> "Range":
        """Creates a range from an A1:B4 notation string."""
        if "!" in a1:
            raise ValueError("Range can only be instantiated from sheet-unaware A1")

        if ":" not in a1:
            addr = Address.from_a1(a1, sheet)
            return cls.from_address(addr)

        a11, a12 = a1.split(":")

        addr1 = Address.from_a1(a11, sheet=sheet)
        addr2 = Address.from_a1(a12, sheet=sheet)

        return cls.from_addresses(addr1, addr2)

    def __str__(self) -> str:
        start = Address(self.min_col, self.min_row, self.sheet)
        end = Address(self.max_col, self.max_row, self.sheet)
        return start.to_a1() + ":" + end.to_a1()

    def to_a1(self) -> str:
        """Converts the range to A1:B4 notation."""
        if self.max_row == -1:
            # A1:A0 -> A1:A
            return self.__str__()[:-1]
        return self.__str__()

    def to_dict(self) -> dict[str, int]:
        """Converts the range to a dictionary."""
        return {
            "min_col": self.min_col,
            "max_col": self.max_col,
            "min_row": self.min_row,
            "max_row": self.max_row,
            "sheet": self.sheet,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Range":
        """Creates a range from a dictionary."""
        return cls(
            d["min_col"],
            d["max_col"],
            d["min_row"],
            d["max_row"],
            d["sheet"],
        )

    def to_coord(self) -> tuple[int, int, int, int, int]:
        """Converts the range to a tuple."""
        return self.min_col, self.max_col, self.min_row, self.max_row, self.sheet


# Convert -1 in max_row/max_col into the highest row/col.
def replace_negative_bounds_with_grid_size(
    src_range: Range, grid_size: tuple[int, int]
) -> Range:
    n_cols, n_rows = grid_size
    max_col = (
        (src_range.max_col + n_cols) if src_range.max_col < 0 else src_range.max_col
    )
    max_row = (
        (src_range.max_row + n_rows) if src_range.max_row < 0 else src_range.max_row
    )
    return replace(src_range, max_col=max_col, max_row=max_row)


def format_cell(*args: Any, **kwargs: Any) -> str:
    """Converts a pair of integers into an A1 notation cell reference."""
    if len(args) == 2:
        x, y = args
        result = [str(y + 1)]
        if kwargs.get("allow_infinite") and y < 0:
            result = []
        d1 = ""
    else:
        d1, x, d2, y = args
        result = [str(y + 1), d2]
    while x >= 0:
        mod = x % 26
        result.append(chr(mod + 65))
        x = x // 26 - 1
    return d1 + "".join(reversed(result))


def parse_cell(cell: str) -> tuple[int, int]:
    """Converts an A1 notation cell reference into a pair of integers."""
    x = 0
    for i, c in enumerate(cell):
        if "0" <= c <= "9":
            break
        x = x * 26 + 1
        x += ord(c) - 65
    return x - 1, int(cell[i:]) - 1


def is_sheet_cell(cell_id: str | list | tuple | Address) -> bool:
    if isinstance(cell_id, Address):
        return True
    if isinstance(cell_id, list | tuple):
        return len(cell_id) == 3
    return bool(cell_id) and (("A" <= cell_id[0] <= "Z") or cell_id.startswith("["))


def is_notebook_cell(cell_id: str) -> bool:
    return cell_id.startswith("0")
