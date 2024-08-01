from typing import Any

import numpy as np

from .cell_address import Address, format_cell
from .expression_compiler import (
    TOK_CELL,
    TOK_CELL_RANGE,
    TOK_COL,
    TOK_ROW,
    format_col,
    format_row,
    is_cell_formula,
    parse_cell_with_dollar,
    parse_col,
    parse_row,
    tokenize_with_ranges,
    untokenize_with_whitespace,
)


def cells_to_grid(cells: list[tuple[Address, Any]]) -> tuple[int, int, list[list[Any]]]:
    xy_to_value = {(address.column, address.row): value for address, value in cells}
    min_x = min(x for x, y in xy_to_value.keys())
    min_y = min(y for x, y in xy_to_value.keys())
    max_x = max(x for x, y in xy_to_value.keys())
    max_y = max(y for x, y in xy_to_value.keys())
    grid = [
        [xy_to_value[x, y] for x in range(min_x, max_x + 1)]
        for y in range(min_y, max_y + 1)
    ]
    return min_x, min_y, grid


def grid_to_cells(
    start_cell: Address, grid: list[list[str]]
) -> list[tuple[Address, str]]:
    x, y = start_cell.column, start_cell.row
    return [
        (Address(x + dx, y + dy, start_cell.sheet), grid[dy][dx])
        for dy in range(len(grid))
        for dx in range(len(grid[0]))
    ]


def copy_cell(formula: str, dx: int, dy: int) -> str:
    def adjust_cell(cell_id: str) -> str:
        d1, x, d2, y = parse_cell_with_dollar(cell_id)
        x1 = x + (0 if d1 else dx)
        if x1 < 0:
            raise ValueError("col < 0")
        y1 = y + (0 if d2 else dy)
        if y1 < 0:
            raise ValueError("row < 0")
        return format_cell(d1, x1, d2, y1)

    def adjust_row(row_id: str) -> str:
        dollar = row_id.startswith("$")
        val = parse_row(row_id)
        row = val + (0 if dollar else dy)
        if row < 0:
            raise ValueError("row < 0")
        return format_row(row, dollar)

    def adjust_col(col_id: str) -> str:
        dollar = col_id.startswith("$")
        val = parse_col(col_id)
        col = val + (0 if dollar else dx)
        if col < 0:
            raise ValueError("col < 0")
        return format_col(col, dollar)

    tokens = tokenize_with_ranges(formula)
    idx = 0
    while idx < len(tokens) - 1:
        toknum, tokval, start, end, line = tokens[idx]
        try:
            if toknum == TOK_CELL:
                tokval = adjust_cell(tokval)
            elif toknum == TOK_CELL_RANGE:
                split_range = tokval.split(":")
                cell_head = adjust_cell(split_range[0])
                cell_tail = adjust_cell(split_range[1])
                tokval = cell_head + ":" + cell_tail
            elif toknum == TOK_ROW:
                split_range = tokval.split(":")
                cell_head = adjust_cell(split_range[0])
                row_tail = adjust_row(split_range[1])
                tokval = cell_head + ":" + row_tail
            elif toknum == TOK_COL:
                split_range = tokval.split(":")
                cell_head = adjust_cell(split_range[0])
                col_tail = adjust_col(split_range[1])
                tokval = cell_head + ":" + col_tail
        except ValueError:
            return "=REF_ERROR"
        tokens[idx] = toknum, tokval, start, end, line

        idx += 1

    return untokenize_with_whitespace(tokens)


def almost_int(number: float) -> bool:
    remainder = number % 1
    if remainder > 0.5:
        remainder = -remainder + 1
    return remainder < 1e-7


def extend_row(
    row: list[str], how_many: int, transpose: bool, reverse: bool
) -> list[str]:
    """Return the next how_many items for row.

    Excel seems to just cycle through strings and apply copy_cell to formulas.
    For numbers it uses the least square method for all the numbers in the row
    ignoring anything that isn't a number. So [1 2 A] will continue as
    [3 4 A]. Numpy has of course the least square regression, see:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    """
    row_numbers: list[float] = []
    number_idxs: list[int] = []
    if reverse:
        row = row[::-1]
    for n in row:
        try:
            val = float(n)
            number_idxs.append(len(row_numbers))
            row_numbers.append(val)
        except ValueError:
            number_idxs.append(-1)
    if row_numbers:
        x = np.array([*range(len(row_numbers))])
        y = np.array(row_numbers)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        m = c = 0

    def next_cell(idx: int) -> str:
        mod_idx = idx % len(row)
        template = row[mod_idx]
        if number_idxs[mod_idx] != -1:
            idx = number_idxs[mod_idx] + (1 + idx // len(row)) * len(row_numbers)
            next_number = idx * m + c
            if almost_int(next_number):
                next_number = int(round(next_number))
            return str(next_number)
        elif is_cell_formula(template):
            dx, dy = len(row) * (1 + idx // len(row)), 0
            if transpose:
                dx, dy = dy, dx
            if reverse:
                dx = -dx
                dy = -dy
            return copy_cell(template, dx, dy)
        elif isinstance(template, str):
            return template

    res = [next_cell(idx) for idx in range(how_many)]
    if reverse:
        res = res[::-1]
    return res


def pre_copy_adjust(
    anchor: Address,
    to_copy: list[tuple[Address, str, dict[str, Any] | None]],
) -> list[tuple[Address, str, dict[str, Any] | None]]:
    def adjust_cell(formula: str, cell_id: Address) -> str:
        if not isinstance(formula, str) or not is_cell_formula(formula):
            return formula
        try:
            return copy_cell(
                formula, cell_id.column - anchor.column, cell_id.row - anchor.row
            )
        except ValueError:
            return formula

    return [
        (cell_id, adjust_cell(formula, cell_id), attributes or None)
        for cell_id, formula, attributes in to_copy
    ]


async def extend_cells(
    populate_from: list[tuple[Address, str]],
    populate_to_start: Address,
    populate_to_end: Address,
    context: list[str] | None = None,
) -> list[tuple[Address, str]]:
    if not populate_from:
        return []
    from_min_x, from_min_y, from_grid = cells_to_grid(populate_from)

    assert populate_to_start.column == from_min_x or populate_to_start.row == from_min_y
    transpose = populate_to_start.column == from_min_x
    if transpose:
        # Without loss of generality
        from_grid = [
            [from_grid[y][x] for y in range(len(from_grid))]
            for x in range(len(from_grid[0]))
        ]
        from_min_x, from_min_y = from_min_y, from_min_x
        to_min_x = populate_to_start.row
        to_max_x = populate_to_end.row
    else:
        to_min_x = populate_to_start.column
        to_max_x = populate_to_end.column
    coll_count_to_add = to_max_x - to_min_x + 1
    reverse = from_min_x > to_min_x
    result = [
        extend_row(row, coll_count_to_add, transpose, reverse) for row in from_grid
    ]
    if transpose:
        result = [
            [result[y][x] for y in range(len(result))] for x in range(len(result[0]))
        ]
    return grid_to_cells(populate_to_start, result)
