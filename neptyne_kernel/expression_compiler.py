import ast
import re
from dataclasses import dataclass, replace
from io import BytesIO
from token import (
    COMMENT,
    ERRORTOKEN,
    INDENT,
    NAME,
    NEWLINE,
    NUMBER,
    OP,
    STRING,
    TYPE_COMMENT,
    tok_name,
)
from tokenize import TokenError, tokenize, untokenize
from typing import Any, Callable, Iterable

import libcst as cst
from libcst import matchers as cst_matchers
from untokenize import untokenize as untokenize_with_whitespace

from .cell_address import (
    Address,
    Range,
    format_cell,
    parse_cell,
    replace_negative_bounds_with_grid_size,
)
from .formula_names import FORMULA_NAMES
from .neptyne_protocol import Dimension, WidgetRegistry
from .spreadsheet_error import SheetDoesNotExist
from .transformation import Transformation

N_RE = re.compile(r"N_[ \t]*\[(?P<addr>\-?\d+(?:[ \t]*,[ \t]*((\-?\d+)|\'\w+\'))+)\]")


DEFAULT_N_COLS = 26
DEFAULT_N_ROWS = 1000
DEFAULT_GRID_SIZE = (DEFAULT_N_COLS, DEFAULT_N_ROWS)

is_a1_str_pattern = re.compile(r"[A-Z]+[0-9]+")
f_string_start_pattern = re.compile(r"^(?:[fF][rR]?|[rR][fF])[\'\"]{1,3}")
f_string_expr_pattern = re.compile(r"(\{+([^{}]+)\}+)")

EXCEL_TO_PY_NAMES = {"T.DIST.2T": "T.DIST_2T", "T.INV.2T": "T.INV_2T"}
PY_TO_EXCEL_NAMES = {v: k for k, v in EXCEL_TO_PY_NAMES.items()}


def replace_n_with_a1_match(
    m: re.Match, positions: dict[tuple[int, int], str] | None = None
) -> str:
    def to_int_or_not(x: str) -> int | str:
        s = x.strip(" \t\n'\"")
        return int(s) if s.isdigit() or (s[1:].isdigit() and s.startswith("-")) else s

    from .dash import Dash

    addr_tuple = [to_int_or_not(x) for x in m.group("addr").split(",")]
    sheet = addr_tuple[-1]
    try:
        item = Dash.instance().from_coordinate(addr_tuple)  # type: ignore
    except SheetDoesNotExist:
        item = Dash.instance().from_coordinate([*addr_tuple[:-1], 0])  # type: ignore

    if isinstance(item, Range):
        # Range
        start_col_final, start_row_final = item.min_col, item.min_row
        end_col_final, end_row_final = Dash.instance().resolve_max_col_row(item)
        a1 = f"{format_cell(start_col_final,start_row_final)}:{format_cell(end_col_final,end_row_final, allow_infinite=True)}"
    else:
        # Address
        col, row = item.column, item.row
        a1 = format_cell(col, row)
    result = sheet + "!" + a1 if isinstance(sheet, str) else a1
    if positions is not None:
        positions[(m.start(), m.end())] = result
    return result


def replace_n_with_a1(
    line: str, func: Callable, positions: dict[tuple[int, int], str] | None = None
) -> str:
    return re.sub(
        N_RE, func if positions is None else lambda s: func(s, positions), line
    )


def is_f_string(s: str) -> bool:
    return bool(re.match(f_string_start_pattern, s))


def find_fstring_expressions(s: str) -> list[tuple[str, int, int]]:
    expressions = []
    for match in re.finditer(f_string_expr_pattern, s):
        expression = match.group(1)
        if is_f_string(expression):
            expressions += find_fstring_expressions(expression[1:-1])
        elif (expression.replace("{{", "").replace("}}", "")).startswith("{"):
            expressions.append(
                (s[match.start(2) : match.end(2)], match.start(2), match.end(2))
            )
    return expressions


def is_cell_formula(formula: str) -> bool:
    return formula.startswith("=")


def parse_row(row: str) -> int:
    row = row.replace("$", "")
    return int(row) - 1


def parse_col(col: str) -> int:
    col = col.replace("$", "")
    x = 0
    for i, c in enumerate(col):
        x = x * 26 + 1
        x += ord(c) - 65
    return x - 1


def format_row(row: int, leading_dollar: bool = False) -> str:
    dollar_str = "$" if leading_dollar else ""
    return dollar_str + str(row + 1)


def format_col(col: int, leading_dollar: bool = False) -> str:
    dollar_str = "$" if leading_dollar else ""
    result = []
    while col >= 0:
        mod = col % 26
        result.append(chr(mod + 65))
        col = col // 26 - 1
    return dollar_str + "".join(reversed(result))


def parse_cell_with_dollar(cell: str) -> tuple[str, int, str, int]:
    if cell.startswith("$"):
        first_dollar = "$"
        cell = cell[1:]
    else:
        first_dollar = ""
    second_dollar = ""
    x = 0
    for i, c in enumerate(cell):
        if c == "$":
            if second_dollar:
                raise ValueError("Too many $s in cell: " + cell)
            second_dollar = c
            continue
        if "0" <= c <= "9":
            break
        x = x * 26 + 1
        x += ord(c) - 65
    return first_dollar, x - 1, second_dollar, int(cell[i:]) - 1


def parse_partial_row_col(s: str) -> int:
    s = s.replace("$", "")
    if is_numeric_only(s):
        return int(s) - 1
    elif is_alpha_only(s):
        x = 0
        for _, c in enumerate(s):
            x = x * 26 + 1
            x += ord(c) - 65
        return x - 1
    raise ValueError("Invalid partial")


def parse_row_col_into_cell_range(s: str) -> str:
    s = s.replace("$", "")
    p = s.find(":")
    if p == -1:
        return ""
    s1 = s[:p]
    s2 = s[p + 1 :]

    # Resolve partial rows/columns into cells bounded by the max dimensions.
    if is_numeric_only(s1):
        s1 = format_cell(0, parse_partial_row_col(s1))
    elif is_alpha_only(s1):
        s1 = format_cell(parse_partial_row_col(s1), 0)
    if is_numeric_only(s2):
        s2 = format_cell(-1, parse_partial_row_col(s2))
    elif is_alpha_only(s2):
        s2 = format_cell(parse_partial_row_col(s2), -1)
    return s1 + ":" + s2


@dataclass
class CompileResult:
    compiled_code: str
    cells_mentioned: set[Address]
    raw_code: str | None = None


def compile_shell(expression: str) -> str:
    chunks = [f"'{chunk}'" for chunk in expression.split(" ")]
    return f"run_command([{', '.join(chunks)}])"


TOK_CELL = max(tok_name.keys()) + 1
TOK_CELL_RANGE = TOK_CELL + 1
TOK_ARRAY = TOK_CELL_RANGE + 1

# Intermediate tokens to hold "$22" or "$AB" respectively. Leading $ is optional.
TOK_PARTIAL_ROW = TOK_ARRAY + 1
TOK_PARTIAL_COL = TOK_PARTIAL_ROW + 1

# Holds rows/columns or segmented rows/columns. E.g. A:B or A$3:B.
TOK_ROW = TOK_PARTIAL_COL + 1
TOK_COL = TOK_ROW + 1

TOK_SHEET_NAME = TOK_COL + 1


def is_alpha_only(token: str) -> bool:
    return all("A" <= ch <= "Z" or ch == "$" for ch in token)


def is_numeric_only(token: str) -> bool:
    return all("0" <= ch <= "9" or ch == "$" for ch in token)


Tokens = list[tuple[int, str, tuple[int, int], tuple[int, int], str]]
SimplifiedTokens = list[tuple[int, str]]


def is_cell(token: str) -> bool:
    return is_a1_str_pattern.fullmatch(token) is not None


def parse_as_cell_or_partial(toknum: int, tokval: str) -> int:
    if toknum == NAME or toknum == NUMBER:
        if toknum == NAME:
            if is_alpha_only(tokval):
                return TOK_PARTIAL_COL
            elif is_cell(tokval):
                return TOK_CELL
        elif toknum == NUMBER:
            # Number should be a positive integer
            try:
                int_val = int(tokval)
                if int_val > 0:
                    return TOK_PARTIAL_ROW
            except ValueError:
                return toknum
    return toknum


def is_parsed_as_cell_or_partial(toknum: int) -> bool:
    return toknum in {TOK_PARTIAL_ROW, TOK_PARTIAL_COL, TOK_CELL}


def rename_variable_in_code(
    compiled_code: str, raw_code: str, old_var_name: str, new_var_name: str
) -> str | None:
    def extract_names(node: Any) -> Iterable:
        """Extracts names from 'ast.Name' nodes if node is not a call"""
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Call):
            for arg in node.args:
                yield from extract_names(arg)
            for keyword in node.keywords:
                yield from extract_names(keyword.value)
        else:
            for child_node in ast.iter_child_nodes(node):
                yield from extract_names(child_node)

    parsed = ast.parse(compiled_code)

    can_be_replaced = False

    for name in extract_names(parsed):
        if name == old_var_name:
            can_be_replaced = True
            break
    if not can_be_replaced:
        return None

    tokens = [
        token[:2] for token in tokenize(BytesIO(raw_code.encode("utf-8")).readline)
    ]
    for i, token in enumerate(tokens):
        if token[0] == NAME and token[1] == old_var_name:
            tokens[i] = (TOK_CELL, new_var_name, *token[2:])  # type: ignore

    return untokenize(tokens).decode("utf-8")


def try_parse_capitalized_range(
    raw_code: str,
    positions: dict[tuple[int, int], str],
    start: int | None,
    end: int | None,
) -> str | None:
    tokens = list(tokenize(BytesIO(raw_code.encode("utf-8")).readline))
    token_modified = False
    for i, token in enumerate(tokens):
        if token[1] == ":" and token[2][1] == start and token[3][1] == end:
            prev_token = tokens[i - 1]
            next_token = tokens[i + 1]
            prev_toknum_upper = parse_as_cell_or_partial(
                prev_token[0], prev_tokval_upper := prev_token[1].upper()
            )
            next_toknum_upper = parse_as_cell_or_partial(
                next_token[0], next_tokval_upper := next_token[1].upper()
            )
            if is_parsed_as_cell_or_partial(
                prev_toknum_upper
            ) and is_parsed_as_cell_or_partial(next_toknum_upper):
                tokens[i - 1] = (prev_token[0], prev_tokval_upper, *prev_token[2:])  # type: ignore
                tokens[i + 1] = (next_token[0], next_tokval_upper, *next_token[2:])  # type: ignore
                token_modified = True
            break
    return untokenize(tokens).decode("utf-8") if token_modified else None


def tokenize_with_cells(expression: str) -> Tokens:
    """Tokenize the expression converting cell references to TOK_CELL"""

    try:
        g = tokenize(
            BytesIO(expression.encode("utf-8")).readline
        )  # tokenize the string
        tokenized: Tokens = [*g]
    except (TokenError, IndentationError) as err:
        raise ValueError(str(err))

    # First pass attaches $ to following valid NAME and NUMBER converting them into:
    # TOK_CELL, TOK_PARTIAL_COL or TOK_PARTIAL_ROW.
    idx = 0
    while idx < len(tokenized) - 1:
        toknum, tokval, start, end, line = tokenized[idx]
        if toknum == ERRORTOKEN:
            if tokval == "$":
                toknum_next, tokval_next, start_next, end_next, line_next = tokenized[
                    idx + 1
                ]
                toknum_next = parse_as_cell_or_partial(
                    toknum_next, tokval_next_upper := tokval_next.upper()
                )
                if is_parsed_as_cell_or_partial(toknum_next):
                    del tokenized[idx]
                    tokenized[idx] = (
                        toknum_next,
                        "$" + tokval_next_upper,
                        start,
                        end_next,
                        line_next,
                    )
            elif tokval == "!" and idx > 0:
                previous = tokenized[idx - 1]
                toknum_prev, tokval_prev, start_prev, end_prev, line_prev = previous
                if (
                    toknum_prev == NAME
                    or toknum_prev == STRING
                    or toknum_prev == TOK_CELL
                ):
                    if toknum_prev == STRING:
                        tokval_prev = ast.literal_eval(tokval_prev)
                    tokenized[idx - 1] = (
                        TOK_SHEET_NAME,
                        tokval_prev,
                        start_prev,
                        end,
                        line,
                    )
                    idx -= 1
        elif toknum == NUMBER or toknum == NAME:
            if toknum == NAME and is_alpha_only(tokval):
                idx_start = idx
                func_name = tokenized[idx][1]
                while idx < len(tokenized) - 1 and (
                    tokenized[idx + 1][1] == "."
                    or tokenized[idx + 1][0] in [NAME, NUMBER]
                ):
                    func_name += tokenized[idx + 1][1]
                    idx += 1
                if idx > idx_start:
                    # Translate Excel function names to Python names (T.DIST.2T -> T.DIST_2T)
                    func_name = EXCEL_TO_PY_NAMES.get(func_name, func_name)
                    if func_name in FORMULA_NAMES:
                        _, _, _, end, line = tokenized[idx]
                        tokenized[idx_start : idx + 1] = [
                            (tokenized[idx_start][0], func_name, start, end, line)
                        ]
                        idx = idx_start + 1
                        continue
                    else:
                        idx = idx_start
            # A1:2. - parse 2 as TOK_PARTIAL_ROW and insert the dot as a separate token
            elif (
                toknum == NUMBER
                and tokval.endswith(".")
                and tokenized[idx - 1][1] == ":"
                and tokenized[idx - 2][0] in (TOK_CELL, TOK_PARTIAL_ROW)
            ):
                tokval = tokval[:-1]
                end_l, end_c = end
                split = (end_l, end_c - 1)
                tokenized[idx + 1 : idx + 1] = [(OP, ".", split, end, line)]
                end = split

            toknum = parse_as_cell_or_partial(toknum, tokval)
            if is_parsed_as_cell_or_partial(toknum):
                tokenized[idx] = toknum, tokval, start, end, line
        idx += 1

    # Second pass merges TOK_PARTIAL_COL followed by TOK_PARTIAL_ROW into TOK_CELL
    idx = 0
    while idx < len(tokenized) - 1:
        toknum, tokval, start, end, line = tokenized[idx]
        if toknum == TOK_PARTIAL_COL:
            toknum_next, tokval_next, start_next, end_next, line_next = tokenized[
                idx + 1
            ]
            if toknum_next == TOK_PARTIAL_ROW:
                del tokenized[idx]
                tokval += tokval_next
                tokenized[idx] = TOK_CELL, tokval, start, end_next, line_next
        idx += 1

    return tokenized


RANGE_TOKENS = {
    (TOK_CELL, TOK_CELL): TOK_CELL_RANGE,
    # Infinite ranges
    (TOK_CELL, TOK_PARTIAL_ROW): TOK_ROW,
    (TOK_CELL, TOK_PARTIAL_COL): TOK_COL,
    (TOK_PARTIAL_ROW, TOK_CELL): TOK_ROW,
    (TOK_PARTIAL_COL, TOK_CELL): TOK_COL,
    # A:A or A:B
    (TOK_PARTIAL_COL, TOK_PARTIAL_COL): TOK_COL,
    # 1:1 or 2:3
    (TOK_PARTIAL_ROW, TOK_PARTIAL_ROW): TOK_ROW,
}


def tokenize_with_ranges(expression: str) -> Tokens:
    tokens = tokenize_with_cells(expression)
    idx = 0

    fn_def: bool = False
    bracket_stack: list[str] = []

    def bracket_stack_top() -> str | None:
        return bracket_stack[-1] if bracket_stack else None

    while idx < len(tokens) - 2:
        (
            (tn1, tv1, start1, end1, line1),
            (tn2, tv2, start2, end2, line2),
            (tn3, tv3, start3, end3, line3),
        ) = tokens[idx : idx + 3]

        if tn3 == OP and tv3 == "(":
            if tn1 == NAME and tv1 == "def" and tn2 == NAME:
                fn_def = True
            else:
                bracket_stack.append(tv3)

        elif tn1 == OP:
            if tv1 == ")":
                if bracket_stack_top() == "(":
                    bracket_stack.pop()
                elif fn_def and tv2 == ":":
                    fn_def = False

            if tv1 == "[" or tv1 == "{":
                bracket_stack.append(tv1)
            elif bracket_stack:
                top_bracket = bracket_stack_top()
                if (tv1 == "]" and top_bracket == "[") or (
                    tv1 == "}" and top_bracket == "{"
                ):
                    bracket_stack.pop()

        elif tv2 == ":" and (tn1, tn3) in RANGE_TOKENS:
            if idx > 0:
                top_bracket = bracket_stack_top()
                is_openinig_br = top_bracket in ("[", "{")
                if (
                    tn1 == tn3 == TOK_PARTIAL_COL
                    and (
                        tokens[idx - 1][:2] == (NAME, "lambda")
                        or (fn_def and top_bracket is None)
                        or is_openinig_br
                    )
                ) or (tn1 == tn3 == TOK_PARTIAL_ROW and is_openinig_br):
                    idx += 1
                    continue

            tokens[idx] = (
                RANGE_TOKENS[(tn1, tn3)],
                tv1 + tv2 + tv3,
                start1,
                end3,
                line3,
            )

            del tokens[idx + 1]
            del tokens[idx + 1]
        idx += 1

    # Convert remaining TOK_PARTIAL_ROW and TOK_PARTIAL_COL back into NAME and NUMBER
    idx = 0
    while idx < len(tokens):
        toknum, tokval, start, end, line = tokens[idx]
        if toknum == TOK_PARTIAL_ROW or toknum == TOK_PARTIAL_COL:
            toknum = NUMBER if toknum == TOK_PARTIAL_ROW else NAME
            if tokval.startswith("$"):
                tokval = tokval.replace("$", "")
                boundary = (start[0], start[1] + 1)
                tokens[idx] = ERRORTOKEN, "$", start, boundary, line
                tokens.insert(idx + 1, (toknum, tokval, boundary, end, line))
                idx += 1
            else:
                tokens[idx] = (toknum, tokval, start, end, line)
        idx += 1
    return tokens


def transform_row_id(row_id: str, transformation: Transformation) -> object | str:
    if transformation.dimension != Dimension.ROW:
        return Transformation.NO_CHANGE
    leading_dollar = row_id.startswith("$")
    row_id = row_id.replace("$", "")
    parsed_id = parse_row(row_id)
    res = transformation.transform(0, parsed_id)
    if res is Transformation.NO_CHANGE or res is Transformation.REF_ERROR:
        return res
    else:
        assert isinstance(res, tuple)
        _, y = res
        return format_row(y, leading_dollar)


def transform_col_id(col_id: str, transformation: Transformation) -> object | str:
    if transformation.dimension != Dimension.COL:
        return Transformation.NO_CHANGE
    leading_dollar = col_id.startswith("$")
    row_id = col_id.replace("$", "")
    parsed_id = parse_col(row_id)
    res = transformation.transform(parsed_id, 0)
    if res is Transformation.NO_CHANGE or res is Transformation.REF_ERROR:
        return res
    else:
        assert isinstance(res, tuple)
        x, _ = res
        return format_col(x, leading_dollar)


def transform_cell_id(cell_id: str, transformation: Transformation) -> str | object:
    d1, x, d2, y = parse_cell_with_dollar(cell_id)
    res = transformation.transform(x, y)
    if res is Transformation.NO_CHANGE or res is Transformation.REF_ERROR:
        return res
    else:
        assert isinstance(res, tuple)
        x, y = res
        return format_cell(d1, x, d2, y)


def process_token_transformation(
    tok_num: int, token_val: str, transformation: Transformation
) -> tuple[int, object | str]:
    if tok_num == TOK_CELL:
        cell_id = transform_cell_id(token_val, transformation)
        if cell_id is Transformation.REF_ERROR:
            return NAME, "REF_ERROR"
        return tok_num, cell_id
    elif tok_num == TOK_CELL_RANGE or tok_num == TOK_ROW or tok_num == TOK_COL:
        # Apply transformation to individual segments
        ocell1, ocell2 = token_val.split(":")
        if tok_num == TOK_CELL_RANGE:
            cell1, cell2 = (
                transform_cell_id(cell_id, transformation)
                for cell_id in (ocell1, ocell2)
            )
        elif tok_num == TOK_ROW:
            if is_numeric_only(ocell1):
                cell1 = transform_row_id(ocell1, transformation)
            else:
                cell1 = transform_cell_id(ocell1, transformation)
            cell2 = transform_row_id(ocell2, transformation)
        else:
            if is_alpha_only(ocell1):
                cell1 = transform_col_id(ocell1, transformation)
            else:
                cell1 = transform_cell_id(ocell1, transformation)
            cell2 = transform_col_id(ocell2, transformation)
        # Entire range is deleted
        if cell1 is Transformation.REF_ERROR and cell2 is Transformation.REF_ERROR:
            return NAME, "REF_ERROR"
        # One end of the range is deleted.
        if cell1 is Transformation.REF_ERROR or cell2 is Transformation.REF_ERROR:
            if cell1 is Transformation.NO_CHANGE:
                cell1 = ocell1
                if tok_num == TOK_CELL_RANGE:
                    d1, x, d2, y = parse_cell_with_dollar(ocell2)
                    if transformation.dimension == Dimension.COL:
                        x -= transformation.amount
                    else:
                        y -= transformation.amount
                    cell2 = format_cell(d1, x, d2, y)
                elif tok_num == TOK_ROW:
                    leading_dollar = ocell2.startswith("$")
                    y = parse_row(ocell2.replace("$", ""))
                    y -= transformation.amount
                    cell2 = format_row(y, leading_dollar)
                elif tok_num == TOK_COL:
                    leading_dollar = ocell2.startswith("$")
                    x = parse_col(ocell2.replace("$", ""))
                    x -= transformation.amount
                    cell2 = format_col(x, leading_dollar)
            else:
                # We have a ref error on one side. The other side has already been transformed though:
                if cell1 is Transformation.REF_ERROR:
                    cell1 = ocell1
                    if cell2 is Transformation.NO_CHANGE:
                        cell2 = ocell2
                else:
                    cell2 = ocell2
            assert isinstance(cell1, str)
            assert isinstance(cell2, str)
            if tok_num == TOK_CELL_RANGE and cell1.replace("$", "") == cell2.replace(
                "$", ""
            ):
                return TOK_CELL, cell1
            return tok_num, cell1 + ":" + cell2
        # Range unmodified
        if cell1 is Transformation.NO_CHANGE and cell2 is Transformation.NO_CHANGE:
            return tok_num, Transformation.NO_CHANGE
        # Single side of range growth.
        if cell1 is Transformation.NO_CHANGE:
            cell1 = ocell1
        if cell2 is Transformation.NO_CHANGE:
            cell2 = ocell2
        assert isinstance(cell1, str)
        assert isinstance(cell2, str)
        return tok_num, cell1 + ":" + cell2
    else:
        return tok_num, Transformation.NO_CHANGE


def process_sheet_transformation(
    tokens: Tokens,
    transformation: Transformation,
    current_sheet_name: str,
    is_different_sheet: bool = False,
) -> str | None:
    token_modified = False
    specified_sheet = None
    for i, (tok_num, tok_val, start, end, line) in enumerate(tokens):
        if tok_num == TOK_SHEET_NAME:
            specified_sheet = tok_val
        elif tok_num == ERRORTOKEN and tok_val == "!" and specified_sheet:
            pass
        elif tok_num in (TOK_CELL, TOK_CELL_RANGE, TOK_ROW, TOK_COL) and (
            (specified_sheet is None and not is_different_sheet)
            or specified_sheet == current_sheet_name
        ):
            tok_num, tok_val_transformed = process_token_transformation(
                tok_num, tok_val, transformation
            )
            if tok_val_transformed != Transformation.NO_CHANGE:
                assert isinstance(tok_val_transformed, str)
                token_modified = True
                tokens[i] = (tok_num, tok_val_transformed, start, end, line)
        else:
            specified_sheet = None

    if token_modified:
        return untokenize_with_whitespace(tokens)
    else:
        return None


def process_sheet_rename(
    tokens: Tokens, old_sheet_name: str, new_sheet_name: str
) -> str | None:
    modified = False
    for i, (tok_num, tok_val, start, end, line) in enumerate(tokens):
        if tok_num == TOK_SHEET_NAME and tok_val == old_sheet_name:
            tokens[i] = tok_num, new_sheet_name, start, end, line
            modified = True

    if not modified:
        return None
    return untokenize_with_whitespace(tokens)


def transform_crosses_infinite_range(
    tokens: Tokens, transformation: Transformation
) -> bool:
    transformation_end = transformation.index + transformation.amount - 1
    for i, (tok_num, tok_val, start, end, line) in enumerate(tokens):
        if tok_num == TOK_ROW and transformation.dimension == Dimension.COL:
            range_start = tok_val.split(":")[0]
            if "A" <= range_start[0] <= "Z":
                x, _ = parse_cell(range_start)
            else:
                # The range start is purely a row, i.e. "1:1"
                x = parse_row(range_start)
            return transformation_end >= x
        elif tok_num == TOK_COL and transformation.dimension == Dimension.ROW:
            range_start = tok_val.split(":")[0]
            if "0" <= range_start[-1] <= "9":
                _, y = parse_cell(range_start.replace("$", ""))
            else:
                # The range start is purely a column, i.e. "B:B"
                y = parse_col(range_start)
            return transformation_end >= y

    return False


def next_non_whitespace(tokens: SimplifiedTokens, search_from: int) -> str | None:
    for toknum, tokval in tokens[search_from + 1 :]:
        if not tokval.isspace() and toknum not in (TYPE_COMMENT, COMMENT):
            return tokval
    return None


def client_expression_to_value(expression: str) -> Any:
    try:
        value = float(expression)
        if value.is_integer():
            value = int(value)
        return value
    except ValueError:
        return expression


def in_excel_formula(function_expression_stack: list[str | None]) -> bool:
    for func in function_expression_stack[::-1]:
        if func is not None:
            return func in FORMULA_NAMES
    return False


def replacements(
    code: str,
    sheet_cell: bool,
) -> Iterable[tuple[int, int, str, str | None]]:
    tokens = [*tokenize_with_ranges(code)]
    sheet_name = None
    line_lengths = [len(line) + 1 for line in code.splitlines()]
    function_expression_stack: list[str | None] = []
    ix = 0
    while ix < len(tokens):
        toknum, tokval, (start_line, start_col), (end_line, end_col), line = tokens[ix]
        pos = sum(line_lengths[: start_line - 1]) + start_col
        if toknum == TOK_SHEET_NAME:
            assert end_line == start_line
            length = end_col - start_col
            sheet_name = tokval
            yield pos, length + 1, "", sheet_name  # add 1 for the '!'
        elif toknum in (TOK_CELL, TOK_CELL_RANGE, TOK_ROW, TOK_COL):
            assert end_line == start_line
            length = end_col - start_col
            addr = tokval
            if toknum in (TOK_ROW, TOK_COL):
                addr = parse_row_col_into_cell_range(addr)
            cls = Range if ":" in addr else Address
            yield (
                pos,
                length,
                cls.from_a1(addr.replace("$", ""), 0),  # type: ignore
                sheet_name,
            )
            sheet_name = None
        elif toknum == STRING:
            if is_f_string(tokval):
                for expr, start, end in find_fstring_expressions(tokval):
                    for rpos, rlength, replacement, sheet_name in replacements(
                        expr, sheet_cell
                    ):
                        yield rpos + pos + start, rlength, replacement, sheet_name
        elif toknum == OP:
            if tokval == "(":
                if tokens[ix - 1][0] == NAME:
                    prev_name = tokens[ix - 1][1]
                    function_expression_stack.append(prev_name)
                else:
                    function_expression_stack.append(None)
            elif tokval == ")":
                function_expression_stack.pop()
            elif tokval == "=" and in_excel_formula(function_expression_stack):
                yield pos, 1, "==", sheet_name
            elif tokval == "{" and (
                in_excel_formula(function_expression_stack)
                or (len(function_expression_stack) == 0 and sheet_cell)
            ):
                a_end, ix, val = consume_excel_array(tokens, ix + 1)
                a_end_pos = sum(line_lengths[: a_end[0] - 1]) + a_end[1] + 1
                yield pos, a_end_pos - pos, f"CellRange([{val}])", sheet_name

        elif toknum == NAME and tokval in EXCEL_TO_PY_NAMES.values():
            yield pos, len(PY_TO_EXCEL_NAMES[tokval]), tokval, sheet_name
        ix += 1


def consume_excel_array(tokens: Tokens, start: int) -> tuple[tuple[int, int], int, str]:
    array: list[str] = []
    ar_row: str = ""
    ix = start
    while ix < len(tokens):
        toknum, tokval, a_start, _end, _ = tokens[ix]
        if tokval == "{":
            raise ValueError("Nested array")
        elif tokval == "}":
            if not array:
                val = f"[{ar_row}]"
            else:
                array.append(ar_row)
                val = ",".join(f"[{r}]" for r in array)
            return a_start, ix, val
        elif tokval == ";":
            array.append(ar_row)
            ar_row = ""
        elif tokval == "," or toknum in (NUMBER, STRING):
            ar_row += tokval
        elif toknum not in (NEWLINE, INDENT):
            raise ValueError("Array supports constant values only")
        ix += 1
    raise ValueError("unmatched }")


class LetExprTransformer(cst_matchers.MatcherDecoratableTransformer):
    @cst_matchers.leave(cst_matchers.Call(func=cst_matchers.Name("LET")))
    def transform_let(
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.Call:
        args = updated_node.args
        let_names: list[cst.Name] = [a.value for a in args[:-1:2]]  # type: ignore
        let_values = []

        def lambda_call(
            params: list[cst.Name], body: cst.BaseExpression, call_args: list[cst.Arg]
        ) -> cst.Call:
            return cst.Call(
                func=cst.Lambda(
                    params=cst.Parameters(params=[cst.Param(name=p) for p in params]),
                    body=body,
                    lpar=[cst.LeftParen()],
                    rpar=[cst.RightParen()],
                ),
                args=call_args,
            )

        for i, arg in enumerate(args[1::2]):
            if i == 0:
                let_values.append(cst.Arg(value=arg.value))
            else:
                let_values.append(
                    cst.Arg(
                        value=lambda_call(
                            params=let_names[:i],
                            body=arg.value,
                            call_args=let_values[:i],
                        )
                    )
                )

        new_call = lambda_call(
            params=let_names,
            body=args[-1],  # type: ignore
            call_args=let_values,
        )

        return new_call


class IfErrorExprTransformer(cst_matchers.MatcherDecoratableTransformer):
    @cst_matchers.leave(
        cst_matchers.Call(func=cst_matchers.Name("IFERROR") | cst_matchers.Name("IFNA"))
    )
    def transform_iferror(
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.Call:
        args = [*updated_node.args]
        if len(args) == 0:
            raise ValueError
        return updated_node.with_changes(
            args=[
                cst.Arg(value=cst.Lambda(body=args[0].value, params=cst.Parameters())),
                *args[1:],
            ]
        )


def run_cst_transforms(code: str) -> str:
    tree = cst.parse_module(code)
    return tree.visit(LetExprTransformer()).visit(IfErrorExprTransformer()).code


def compile_expression(
    expression: str,
    target_cell: Address | None = None,
    sheet_name_to_id: dict | None = None,
    grid_size: tuple[int, int] = DEFAULT_GRID_SIZE,
    *,
    compute_cells_mentioned: bool = True,
    reformat_compiled_code: bool = True,
) -> CompileResult:
    """Compile the expression for the target_cell.

    We want to replace all references to cells of the form [Letter][Number+] with a reference
    into the N_ global. We also want to handle:

    One line expressions like:
        = B1 + B2

    Things that end on just an expression:
        profit = A1 + B2
        loss = C3 + D4
        profit - loss
    """
    sheet_name_to_id = sheet_name_to_id or {}

    if target_cell:
        sheet_cell = True
        target_sheet = target_cell.sheet
        if is_cell_formula(expression):
            expression = expression[1:]
        else:
            if expression == "":
                return CompileResult(
                    compiled_code="",
                    cells_mentioned=set(),
                )
            return CompileResult(
                compiled_code=repr(client_expression_to_value(expression)),
                cells_mentioned=set(),
            )
    else:
        sheet_cell = False
        target_sheet = 0

    cells_mentioned: set[Address] = set()

    parts = []
    start = 0

    for pos, length, replacement, sheet_name in replacements(expression, sheet_cell):
        parts.append(expression[start:pos])
        if isinstance(replacement, (Address, Range)):
            cell_addr = replacement
            coord_tuple = replacement.to_coord()[:-1]
            replacement = (
                f"N_[{str(coord_tuple)[1:-1]}, {sheet_name or target_sheet!r}]"
            )
            if compute_cells_mentioned:
                if sheet_name is not None:
                    sheet_id = sheet_name_to_id.get(sheet_name)
                else:
                    sheet_id = target_sheet
                if sheet_id is not None and compute_cells_mentioned:
                    cell_addr = replace(cell_addr, sheet=sheet_id)
                    if isinstance(cell_addr, Address):
                        cells_mentioned.add(cell_addr)
                    else:
                        cells_mentioned.update(
                            c
                            for row in replace_negative_bounds_with_grid_size(
                                cell_addr, grid_size
                            )
                            for c in row
                        )

        parts.append(replacement)
        start = pos + length

    parts.append(expression[start:])
    compiled = "".join(parts)
    try:
        starts_with_equals = compiled.startswith("=")
        compiled = run_cst_transforms(compiled.removeprefix("="))
        if starts_with_equals:
            compiled = "=" + compiled
    except cst.ParserSyntaxError:
        pass

    if reformat_compiled_code:
        try:
            # Use a very long line length
            blacked_code = reformat_code(compiled, line_length=5000)[0]
            lines = blacked_code.splitlines()
        except ValueError:
            lines = compiled.splitlines()
            if not lines:
                lines.append("''")

            if sheet_cell:
                code = " ".join(lines)
            else:
                code = "\n".join(lines)
            return CompileResult(compiled_code=code, cells_mentioned=cells_mentioned)
    else:
        lines = compiled.splitlines()

    if sheet_cell:
        joiner = " "
    else:
        joiner = "\n"

    return CompileResult(
        compiled_code=joiner.join(lines),
        cells_mentioned=cells_mentioned,
        raw_code="=" + expression,
    )


def reformat_code(code: str, line_length: int | None = None) -> tuple[str, bool]:
    """Reformat the code. Might raise a ValueError.
    Returns:
        (code, is_modified) when 'black' succeeds (is_modified=True) or returns unchanged code (is_modified=False),
    Raises:
        ValueError when 'black' raises InvalidInput or TokenError
    """
    try:
        import black
        import black.parsing
    except ImportError:
        raise ImportError("black is required to reformat")
    line_length = line_length or black.DEFAULT_LINE_LENGTH
    try:
        code = black.format_cell(
            code,
            fast=True,
            mode=black.FileMode(string_normalization=False, line_length=line_length),
        )
        is_modified = True
    except black.NothingChanged:
        is_modified = False
    except (black.InvalidInput, black.parsing.TokenError, TokenError) as err:
        raise ValueError(str(err))
    return code.rstrip("\n") + "\n", is_modified


paren_params_open_to_close = {"(": ")", "{": "}", "[": "]"}
quote_params = {"'", '"', "'''", '"""', "`"}


def parse_widget_code(code: str, widget_registry: WidgetRegistry) -> dict[str, str]:
    tokens: list[tuple[int, str]] = [
        (toknum, tokval) for toknum, tokval, _, _, _ in tokenize_with_ranges(code)
    ]
    widget_info = None
    parameter_number = 0
    current_parameter_str = ""
    parameters = {}
    current_parens: list[str] = []
    current_quote_string = ""
    parameter_key = ""

    i = 0
    while i < len(tokens):
        token = tokens[i]
        toknum, tokval = token

        # Match first name to widget in registry or abort.
        if toknum == NAME and widget_info is None:
            widget_info = widget_registry.widgets.get(tokval)
            if not widget_info:
                return {}

        # Skip everything before we match the widget function.
        elif not widget_info:
            pass

        # Start of parameter
        elif (
            current_parameter_str == ""
            and parameter_key == ""
            and len(current_parens) > 0
        ):
            # Try to match 'x=EXPR'
            if (len(tokens) - i) > 2:
                next_tok_num, next_tok_val = tokens[i + 1]
                if toknum == NAME and next_tok_val == "=":
                    parameter_key = tokval
                    i += 1
            # Try to match 'EXPR'
            if parameter_key == "":
                parameter_key = widget_info.params[parameter_number].name
                i -= 1  # Reprocess this token now that parameter_key is set

        # End of parameter
        elif tokval == "," and len(current_parens) == 1 and current_quote_string == "":
            parameters[parameter_key] = current_parameter_str
            parameter_key = ""
            current_parameter_str = ""
            parameter_number += 1

        elif toknum == TOK_SHEET_NAME:
            if " " in tokval:
                current_parameter_str += repr(tokval)
            else:
                current_parameter_str += tokval

        # Start a quoted string
        elif tokval in quote_params:
            current_quote_string = tokval
            current_parameter_str += tokval

        # Pop out of current quoted string
        elif current_quote_string != "" and tokval == current_quote_string:
            current_quote_string = ""
            current_parameter_str += tokval

        # Passthrough all tokens while in quoted string.
        elif current_quote_string != "":
            current_parameter_str += tokval

        # Increase parenthesis nesting
        elif tokval in paren_params_open_to_close:
            current_parens.append(paren_params_open_to_close[tokval])

            # Don't add the expression opening paren to output str.
            if len(current_parens) > 1:
                current_parameter_str += tokval

        # Decrease parenthesis nesting
        elif len(current_parens) > 0 and current_parens[-1] == tokval:
            current_parens.pop()

            if len(current_parens) > 0:
                current_parameter_str += tokval
            else:  # End of expression
                parameters[parameter_key] = current_parameter_str
                return parameters

        # Passthrough intermediate tokvals into parameter string
        else:
            current_parameter_str += tokval

        i += 1

    return parameters


def compile_mime_typed_expression(raw_code: str, mime_type: str) -> CompileResult:
    code = f"N_.parse_mime_typed_expression({raw_code!r}, {mime_type!r})"
    return CompileResult(
        compiled_code=code,
        cells_mentioned=set(),
    )
