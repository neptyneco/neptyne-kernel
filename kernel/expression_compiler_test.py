from token import ENCODING, ENDMARKER, NAME, NEWLINE

import pytest

from .cell_address import Address, format_cell, parse_cell
from .expression_compiler import (
    TOK_CELL,
    TOK_CELL_RANGE,
    TOK_COL,
    TOK_PARTIAL_COL,
    TOK_PARTIAL_ROW,
    TOK_ROW,
    Dimension,
    compile_expression,
    parse_cell_with_dollar,
    parse_widget_code,
    process_sheet_transformation,
    process_token_transformation,
    tokenize_with_cells,
    tokenize_with_ranges,
    transform_crosses_infinite_range,
)
from .neptyne_protocol import SheetTransform
from .test_utils import a1
from .transformation import Transformation
from .widgets.register_widget import widget_registry


@pytest.mark.parametrize(
    ("expression", "result"),
    [
        (
            '=IF ( B1=1, "JA", dict(a=1))',
            'IF(N_[1, 0, 0] == 1, "JA", dict(a=1))',
        ),
        ("=i=5", "i = 5"),
        ("SUM(A1:A10)", "'SUM(A1:A10)'"),
        ("", ""),
        ("100", "100"),
        ("=AA1", "N_[26, 0, 0]"),
        (
            "=SUM(A1:B)",
            "SUM(N_[0, 1, 0, -1, 0])",
        ),
        (
            "=SUM(A4:5)",
            "SUM(N_[0, -1, 3, 4, 0])",
        ),
        ("=Sheet1!A1", "N_[0, 0, 'Sheet1']"),
        ("='Sheet1'!A1", "N_[0, 0, 'Sheet1']"),
        ("=Sheet1!A1 + A1", "N_[0, 0, 'Sheet1'] + N_[0, 0, 0]"),
        ("=Sheet1!A1 + Sheet0!A1", "N_[0, 0, 'Sheet1'] + N_[0, 0, 'Sheet0']"),
        ("=A1!A1", "N_[0, 0, 'A1']"),
    ],
)
def test_compile_sheet_expression(expression, result):
    sheet_name_to_id = {"Sheet0": 0, "Sheet1": 1, "A1": 2}
    assert (
        compile_expression(
            expression, Address(0, 0, 0), sheet_name_to_id=sheet_name_to_id
        ).compiled_code
        == result
    )


def test_compile_graph():
    compile_expression_result = compile_expression("=len(B1:C2)", Address(0, 0, 0))

    assert compile_expression_result.cells_mentioned == {
        a1("B1"),
        a1("B2"),
        a1("C1"),
        a1("C2"),
    }

    assert compile_expression_result.compiled_code == "len(N_[1, 2, 0, 1, 0])"

    assert compile_expression("=1 + A1 / 100", Address(0, 1, 0)).cells_mentioned == {
        a1("A1")
    }

    assert compile_expression("=AA1", Address(0, 0, 0)).compiled_code == "N_[26, 0, 0]"


@pytest.mark.parametrize(
    ("expression", "result"),
    [
        ("sqrt(10)", "sqrt(10)"),
        ("sqrt(A1)", "sqrt(N_[0, 0, 0])"),
        ("'hello'", "'hello'"),
        ("100", "100"),
        ("A1:A9=range(9)", "N_[0, 0, 0, 8, 0] = range(9)"),
    ],
)
def test_compile_notebook_expression(expression, result):
    assert compile_expression(expression, None).compiled_code == result


@pytest.mark.parametrize(
    ("expression", "target_cell", "result"),
    [
        ("=A$1", "B1", "N_[0, 0, 0]"),
        ("=A1", "B1", "N_[0, 0, 0]"),
        ("$A1", "007", "N_[0, 0, 0]"),
        ("$A$1", "007", "N_[0, 0, 0]"),
        ("$A$1:A5", "007", "N_[0, 0, 0, 4, 0]"),
        ("$AA$10:AB55", "007", "N_[26, 27, 9, 54, 0]"),
    ],
)
def test_compile_expression_eliminates_dollar(expression, target_cell, result):
    if target_cell.startswith("0"):
        target_cell = None
    else:
        target_cell = Address.from_a1_or_str(target_cell)
    assert compile_expression(expression, target_cell).compiled_code == result


@pytest.mark.parametrize(
    "xy,a1",
    [
        ((0, 0), "A1"),
        ((25, 9), "Z10"),
        ((26, 10), "AA11"),
        ((27, 0), "AB1"),
        ((51, 0), "AZ1"),
        ((52, 0), "BA1"),
        ((701, 99), "ZZ100"),
        ((702, 0), "AAA1"),
        ((703, 0), "AAB1"),
    ],
)
def test_a1_notation(xy, a1):
    assert format_cell(*xy) == a1
    assert parse_cell(a1) == xy


@pytest.mark.parametrize(
    "a1",
    [
        "A1",
        "Z10",
        "A$1",
        "$Z10",
        "$A$1",
    ],
)
def test_dollar_a1_notation(a1):
    xy = parse_cell_with_dollar(a1)
    out = format_cell(*xy)
    assert out == a1


@pytest.mark.parametrize(
    "expression,tokens",
    [
        (
            "A$3 + B4 + 155 + $A3:$A$4",
            ["A$3", "+", "B4", "+", "155", "+", "$A3", ":", "$A$4"],
        ),
        ("$", ["$"]),
        ("A$4", ["A$4"]),
        ("$A4", ["$A4"]),
    ],
)
def test_tokenize_with_dollars(expression, tokens):
    def simple_tokenize(expression):
        return [
            tokval
            for toknum, tokval, _, _, _ in tokenize_with_cells(expression)
            if tokval != " " and toknum not in (ENDMARKER, NEWLINE, ENCODING)
        ]

    assert simple_tokenize(expression) == tokens


def test_tokenize_with_cells():
    tokens = tokenize_with_cells("A1+A$2+i2")
    assert tokens[1][0] == TOK_CELL
    assert tokens[3][0] == TOK_CELL
    assert tokens[5][0] == NAME

    tokens = tokenize_with_cells("AA:$AB+$11:14+i2+A1+21")
    assert tokens[1][0] == TOK_PARTIAL_COL
    assert tokens[3][0] == TOK_PARTIAL_COL
    assert tokens[5][0] == TOK_PARTIAL_ROW
    assert tokens[7][0] == TOK_PARTIAL_ROW
    assert tokens[9][0] == NAME
    assert tokens[11][0] == TOK_CELL
    assert tokens[13][0] == TOK_PARTIAL_ROW


def test_tokenize_with_ranges():
    tokens = tokenize_with_ranges("A2:A3+A1+a[i:k]")
    assert tokens[1][0] == TOK_CELL_RANGE
    assert tokens[3][0] == TOK_CELL

    tokens = tokenize_with_ranges("AC21:AB53+AD122+a[i:k]")
    assert tokens[1][0] == TOK_CELL_RANGE
    assert tokens[3][0] == TOK_CELL

    tokens = tokenize_with_ranges("A1:$22+A$4:BB+a3")
    assert tokens[1][0] == TOK_ROW
    assert tokens[3][0] == TOK_COL
    assert tokens[5][0] == NAME


def test_transformation():
    t1 = Transformation(Dimension.COL, SheetTransform.DELETE, 5, 1, 0)
    assert t1.transform(3, 3) is Transformation.NO_CHANGE
    assert t1.transform(6, 3) == (5, 3)
    assert t1.transform(5, 3) is Transformation.REF_ERROR

    t2 = Transformation(Dimension.ROW, SheetTransform.DELETE, 5, 1, 0)
    assert t2.transform(3, 3) is Transformation.NO_CHANGE
    assert t2.transform(3, 6) == (3, 5)
    assert t2.transform(3, 5) is Transformation.REF_ERROR

    t3 = Transformation(Dimension.COL, SheetTransform.INSERT_BEFORE, 5, 1, 0)
    assert t3.transform(3, 3) is Transformation.NO_CHANGE
    assert t3.transform(6, 3) == (7, 3)
    assert t3.transform(5, 2) == (6, 2)


def test_process_token_transformation_delete():
    delete = Transformation(Dimension.COL, SheetTransform.DELETE, 5, 1, 0)
    assert (
        process_token_transformation(NAME, "Hello", delete)[1]
        == Transformation.NO_CHANGE
    )
    assert (
        process_token_transformation(TOK_CELL, "A1", delete)[1]
        == Transformation.NO_CHANGE
    )
    assert process_token_transformation(TOK_CELL, "G1", delete)[1] == "F1"

    assert process_token_transformation(TOK_CELL, "F1", delete)[1] == "REF_ERROR"
    assert process_token_transformation(TOK_CELL_RANGE, "G1:G$5", delete)[1] == "F1:F$5"
    assert process_token_transformation(TOK_CELL_RANGE, "$G1:H1", delete)[1] == "$F1:G1"
    assert process_token_transformation(TOK_CELL_RANGE, "G1:H5", delete)[1] == "F1:G5"

    assert (
        process_token_transformation(TOK_CELL_RANGE, "F1:F5", delete)[1] == "REF_ERROR"
    )
    assert process_token_transformation(TOK_CELL_RANGE, "F1:H1", delete)[1] == "F1:G1"
    assert process_token_transformation(TOK_CELL_RANGE, "F1:G1", delete) == (
        TOK_CELL,
        "F1",
    )

    assert process_token_transformation(TOK_CELL_RANGE, "D1:F1", delete)[1] == "D1:E1"
    assert process_token_transformation(TOK_CELL_RANGE, "E1:F1", delete) == (
        TOK_CELL,
        "E1",
    )
    assert (
        process_token_transformation(TOK_CELL_RANGE, "$D1:F$1", delete)[1] == "$D1:E$1"
    )
    assert process_token_transformation(TOK_CELL_RANGE, "$E$1:F1", delete) == (
        TOK_CELL,
        "$E$1",
    )

    assert process_token_transformation(TOK_COL, "$D$5:F", delete) == (
        TOK_COL,
        "$D$5:E",
    )

    assert process_token_transformation(TOK_COL, "E5:$F", delete) == (
        TOK_COL,
        "E5:$E",
    )

    assert process_token_transformation(TOK_COL, "F4:G", delete) == (
        TOK_COL,
        "F4:F",
    )


def test_process_token_transformation_insert():
    insert = Transformation(Dimension.ROW, SheetTransform.INSERT_BEFORE, 5, 1, 0)
    assert (
        process_token_transformation(NAME, "Hello", insert)[1]
        == Transformation.NO_CHANGE
    )
    assert (
        process_token_transformation(TOK_CELL, "A1", insert)[1]
        == Transformation.NO_CHANGE
    )
    assert process_token_transformation(TOK_CELL, "A8", insert)[1] == "A9"
    assert process_token_transformation(TOK_CELL, "A$6", insert)[1] == "A$7"

    assert (
        process_token_transformation(TOK_CELL_RANGE, "$A8:B$8", insert)[1] == "$A9:B$9"
    )

    assert process_token_transformation(TOK_ROW, "$1:$6", insert)[1] == "$1:$7"

    assert (
        process_token_transformation(TOK_ROW, "1:4", insert)[1]
        == Transformation.NO_CHANGE
    )

    assert process_token_transformation(TOK_ROW, "$A$7:8", insert)[1] == "$A$8:9"


def test_process_sheet_transformation():
    insert = Transformation(Dimension.ROW, SheetTransform.INSERT_BEFORE, 5, 1, 0)
    sheet_name = "Sheet0"

    tokens = tokenize_with_ranges("=SUM(A1:A4)")
    assert process_sheet_transformation(tokens, insert, sheet_name) is None

    tokens = tokenize_with_ranges("=SUM(A1:A8) +   SUM(B1:C99)")
    assert (
        process_sheet_transformation(tokens, insert, sheet_name)
        == "=SUM(A1:A9) +   SUM(B1:C100)"
    )

    tokens = tokenize_with_ranges("=SUM(A1:6)")
    assert process_sheet_transformation(tokens, insert, sheet_name) == "=SUM(A1:7)"

    insert = Transformation(Dimension.COL, SheetTransform.INSERT_BEFORE, 1, 1, 0)
    tokens = tokenize_with_ranges("=SUM(A1:D)")
    assert process_sheet_transformation(tokens, insert, sheet_name) == "=SUM(A1:E)"

    tokens = tokenize_with_ranges("=SUM(E1:F)")
    assert process_sheet_transformation(tokens, insert, sheet_name) == "=SUM(F1:G)"

    tokens = tokenize_with_ranges("=SUM(A1:2)")
    assert not process_sheet_transformation(tokens, insert, sheet_name)

    tokens = tokenize_with_ranges("=SUM(Sheet1!A1:A10)")
    assert not process_sheet_transformation(tokens, insert, sheet_name)

    delete = Transformation(Dimension.COL, SheetTransform.DELETE, 1, 1, 0)
    tokens = tokenize_with_ranges("=SUM(A1:B)")
    assert process_sheet_transformation(tokens, delete, sheet_name) == "=SUM(A1:A)"

    tokens = tokenize_with_ranges("=SUM(AA100:AB101 ) + SUM(A1:AA)")
    assert (
        process_sheet_transformation(tokens, delete, sheet_name)
        == "=SUM(Z100:AA101 ) + SUM(A1:Z)"
    )

    tokens = tokenize_with_ranges("=SUM(AA100:AB101 )\n+\nSUM(A1:AA)")
    assert (
        process_sheet_transformation(tokens, delete, sheet_name)
        == "=SUM(Z100:AA101 )\n+\nSUM(A1:Z)"
    )

    tokens = tokenize_with_ranges("=SUM(B1:B)")
    assert "REF_ERROR" in process_sheet_transformation(tokens, delete, sheet_name)

    tokens = tokenize_with_ranges("=SUM(A1:A)")
    assert (
        not process_sheet_transformation(tokens, delete, sheet_name)
        == Transformation.REF_ERROR
    )

    tokens = tokenize_with_ranges("=SUM(Sheet1!A1:A10)")
    assert not process_sheet_transformation(tokens, delete, sheet_name)

    tokens = tokenize_with_ranges("=SUM(A1:A10)")
    assert not process_sheet_transformation(
        tokens, delete, sheet_name, is_different_sheet=True
    )


@pytest.mark.parametrize(
    "code, result",
    [
        ("=T.DIST(60, 1, TRUE)", "T.DIST(60, 1, TRUE)"),
        ("=T.DIST.2T(1, 2)", "T.DIST_2T(1, 2)"),
        ("=T.INV.2T(0.546449, 60)", "T.INV_2T(0.546449, 60)"),
        ("=T.INV.2T(T.DIST.2T(1, 2), 60)", "T.INV_2T(T.DIST_2T(1, 2), 60)"),
    ],
)
def test_formulas_with_dots(code, result):
    assert compile_expression(code, Address(0, 0, 0)).compiled_code == result


@pytest.mark.parametrize(
    "code, result",
    [
        ("={1,2,3;4,5,6;7,8,9}", "CellRange([[1, 2, 3], [4, 5, 6], [7, 8, 9]])"),
        ("={10,9,8,7}", "CellRange([[10, 9, 8, 7]])"),  # single row
        ("={10;9;8;7}", "CellRange([[10], [9], [8], [7]])"),  # single column
        (
            '=MAX({1.5,2.5,3.5;4.5,5.5,6.5;7.5,8.5,9.5,"hello"})',
            'MAX(CellRange([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5, "hello"]]))',
        ),
        (
            "=T.INV.2T(T.DIST.2T(MIN({1,2,3;4,5,6;7,8,9}),2), 60)",
            "T.INV_2T(T.DIST_2T(MIN(CellRange([[1, 2, 3], [4, 5, 6], [7, 8, 9]])), 2), 60)",
        ),
        ('=   T.DIST.2T(2, 3,  "hello")', 'T.DIST_2T(2, 3, "hello")'),
    ],
)
def test_array(code, result):
    assert compile_expression(code, Address(0, 0, 0)).compiled_code == result


@pytest.mark.parametrize(
    "code, result",
    [
        (
            "x=MAX({1.5,2.5,3.5;4.5,5.5,6.5})",
            "x = MAX(CellRange([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]))",
        ),
        ("x={1,2,3}", "x = {1, 2, 3}"),
        ("x={}", "x = {}"),
    ],
)
def test_array_notebook(code, result):
    assert compile_expression(code, None).compiled_code == result


@pytest.mark.parametrize(
    "code",
    [
        "={1,2,{3,4}}",
        "=MAX({1,2," "MIN({1,2,3;4,5,6;7,8,9})})",
    ],
)
def test_array_errors(code):
    with pytest.raises(ValueError):
        compile_expression(code, Address(0, 0, 0))


@pytest.mark.parametrize(
    "code, result",
    [
        (
            """def update_portfolio_performance():
    today = int(TODAY())
    history = [(date, price) for date, price in A16:B116 if date and price]
    if history and history[0][0] == today:
        return
    A16 =[(today, B1)] + history""",
            """def update_portfolio_performance():\n    today = int(TODAY())\n    history = [(date, price) for date, price in N_[0, 1, 15, 115, 0] if date and price]\n    if history and history[0][0] == today:\n        return\n    N_[0, 15, 0] = [(today, N_[1, 0, 0])] + history""",
        ),
        (
            "today = int(TODAY())\nhistory = today + 1",
            "today = int(TODAY())\nhistory = today + 1",
        ),
        (
            "today = int(float(int(SQRT(TODAY())))) + POW(2, MAX(0, 2))\nhistory = today + 1",
            "today = int(float(int(SQRT(TODAY())))) + POW(2, MAX(0, 2))\nhistory = today + 1",
        ),
        ("('hello', 'tuple', NOW())", "('hello', 'tuple', NOW())"),
        (
            """result = MAX(int(IF(A1=4, "Ja", "Nee")), 10)""",
            """result = MAX(int(IF(N_[0, 0, 0] == 4, "Ja", "Nee")), 10)""",
        ),
        ("()", "()"),
    ],
)
def test_excel_vs_py_mode(code, result):
    assert compile_expression(code, None).compiled_code == result


@pytest.mark.parametrize(
    ("expression", "result"),
    [
        (
            "=SUM(A4:5)",
            "SUM(N_[0, -1, 3, 4, 0])",
        ),
        ("=SUM(A:A)", "SUM(N_[0, 0, 0, -1, 0])"),
        ("=SUM(B:A)", "SUM(N_[0, 1, 0, -1, 0])"),
        ("=SUM(A:D)", "SUM(N_[0, 3, 0, -1, 0])"),
        ("=SUM(3:3)", "SUM(N_[0, -1, 2, 2, 0])"),
        ("=SUM(4:10)", "SUM(N_[0, -1, 3, 9, 0])"),
        ("=lambda A:A*2", "lambda A: A * 2"),
        ("=SUM(10:4)", "SUM(N_[0, -1, 3, 9, 0])"),
        ('="hello"[1:3]', '"hello"[1:3]'),
        ('="hello, world"[1:3:2]', '"hello, world"[1:3:2]'),
        ('="hello, world"[1+0:3-2]', '"hello, world"[1 + 0 : 3 - 2]'),
        ('="hello, world"[1+0:3-2:-1+2]', '"hello, world"[1 + 0 : 3 - 2 : -1 + 2]'),
        ("=list({1:2, 3:4})", "list({1: 2, 3: 4})"),
        ("=max(SUM({1,3}),2)", "max(SUM(CellRange([[1, 3]])), 2)"),
        (
            "=max(SUM({1,3}),1:3[0])",
            "max(SUM(CellRange([[1, 3]])), N_[0, -1, 0, 2, 0][0])",
        ),
        (
            "=max(SUM({1,3}),A:A[0])",
            "max(SUM(CellRange([[1, 3]])), N_[0, 0, 0, -1, 0][0])",
        ),
        (
            "=list(sorted(1:2, key=lambda A:A*2))",
            "list(sorted(N_[0, -1, 0, 1, 0], key=lambda A: A * 2))",
        ),
        (
            "=list(sorted(A:A, key=lambda A:3*A))",
            "list(sorted(N_[0, 0, 0, -1, 0], key=lambda A: 3 * A))",
        ),
        (
            "=list(sorted(B4:6, key=lambda A:A*2))",
            "list(sorted(N_[1, -1, 3, 5, 0], key=lambda A: A * 2))",
        ),
        (
            "=list(sorted(B4:6[1 + 0 : 3 + 2 : 1 - 2]))",
            "list(sorted(N_[1, -1, 3, 5, 0][1 + 0 : 3 + 2 : 1 - 2]))",
        ),
        ("=MAX(1:1[1:1])", "MAX(N_[0, -1, 0, 0, 0][1:1])"),
        ("=list({A:A, B:B})", "list({A: A, B: B})"),
        ("={3,2,1}", "CellRange([[3, 2, 1]])"),
        ("=MAX(A1:2[10:20:2])", "MAX(N_[0, -1, 0, 1, 0][10:20:2])"),
    ],
)
def test_google_sheets_ranges(expression, result):
    assert compile_expression(expression, Address(0, 0, 0)).compiled_code == result


@pytest.mark.parametrize(
    ("expression", "result"),
    [
        ("def a(B:C): return B * 2", "def a(B: C):\n    return B * 2"),
        (
            "def a (B:C, F:E={A:D}): return B * 2",
            "def a(B: C, F: E = {A: D}):\n    return B * 2",
        ),
        (
            "def a (B:C=(2*3+(7-8)), F:E={A:D}): return B * 2",
            "def a(B: C = (2 * 3 + (7 - 8)), F: E = {A: D}):\n    return B * 2",
        ),
        (
            "def a (B:C=MAX(X:Y)): return B*2",
            "def a(B: C = MAX(N_[23, 24, 0, -1, 0])):\n    return B * 2",
        ),
        (
            "def a (x:int=(1+4+{1,2,3}.pop())*2, B:C=(2*3+(7-8)+MAX(X:Y)), F:E={A:D}): return B * 2",
            "def a(x: int = (1 + 4 + {1, 2, 3}.pop()) * 2, B: C = (2 * 3 + (7 - 8) + MAX(N_[23, 24, 0, -1, 0])), F: E = {A: D}):\n    return B * 2",
        ),
        (
            "def a (B:C=2*3+MAX(X:Y), F:E={A:D}): return B * 2",
            "def a(B: C = 2 * 3 + MAX(N_[23, 24, 0, -1, 0]), F: E = {A: D}):\n    return B * 2",
        ),
    ],
)
def test_type_annotations(expression, result):
    assert compile_expression(expression).compiled_code == result


@pytest.mark.parametrize(
    ("expression", "result"),
    [
        ("=Scatter(x=[1,2,3], y=[4,5,6])", {"x": "[1,2,3]", "y": "[4,5,6]"}),
        ("=Scatter(x=A1:A5, y=B1:B5)", {"x": "A1:A5", "y": "B1:B5"}),
        ("=Scatter(A1:A5, B1:B5)", {"x": "A1:A5", "y": "B1:B5"}),
        (
            "=Scatter(x=([1,2,3], [4,5], {}, A2:A4), y=B1:B5)",
            {"x": "([1,2,3],[4,5],{},A2:A4)", "y": "B1:B5"},
        ),
        ("=Scatter(A1:A5, y=B1:B5)", {"x": "A1:A5", "y": "B1:B5"}),
        (
            "=Scatter(([1,2,3], [4,5], {}, A2:A4), B1:B5)",
            {"x": "([1,2,3],[4,5],{},A2:A4)", "y": "B1:B5"},
        ),
        ('=Scatter({"abcdf": A1:A5}, B1:B5)', {"x": '{"abcdf":A1:A5}', "y": "B1:B5"}),
        (
            '=Scatter("cat{A1:A4}dog"dfdfdfd"", "")',
            {"x": '"cat{A1:A4}dog"dfdfdfd""', "y": '""'},
        ),
        ('=Button("A") + Button("B")', None),
        (
            "=Scatter(\"Sheet 2\"!A1:A3, 'Sheet 2'!B1:B3)",
            {"x": "'Sheet 2'!A1:A3", "y": "'Sheet 2'!B1:B3"},
        ),
        (
            "=Scatter(Sheet2!A1:A3, Sheet2!B1:B3)",
            {"x": "Sheet2!A1:A3", "y": "Sheet2!B1:B3"},
        ),
    ],
)
def test_parse_widget_expression(expression, result):
    if not result:
        parse_widget_code(expression, widget_registry)
    else:
        assert parse_widget_code(expression, widget_registry) == result


@pytest.mark.parametrize(
    ("expression", "result"),
    [
        ("f'1+{A1:C2}'", "f'1+{N_[0, 2, 0, 1, 0]}'"),
        ("F'{A1:C2}'", "F'{N_[0, 2, 0, 1, 0]}'"),
        (
            """f'''String \n\n\n{f'{f"{A1:C2}"}'} is an f-string'''""",
            """f'''String \n\n\n{f'{f"{N_[0, 2, 0, 1, 0]}"}'} is an f-string'''""",
        ),
        (
            '''f"A:B, 5:7 {A:B, 5:7} C2 {C2:8}"''',
            '''f"A:B, 5:7 {N_[0, 1, 0, -1, 0], N_[0, -1, 4, 6, 0]} C2 {N_[2, -1, 1, 7, 0]}"''',
        ),
        (
            """f'one, two: three {A1+B2:3d} four! five? six* {c+d=} seven% %eight%%% nine^$'""",
            """f'one, two: three {N_[0, 0, 0]+N_[1, -1, 1, 2, 0]d} four! five? six* {c+d=} seven% %eight%%% nine^$'""",
        ),
        (
            """f'one {A1+B2:.3f} two {c+d=} three'""",
            """f'one {N_[0, 0, 0]+N_[1, 1, 0]:.3f} two {c+d=} three'""",
        ),
        ("""f'{a !s:2}'""", """f'{a !s:2}'"""),
        ("""f'{a}'""", """f'{a}'"""),
        ("""f'{x[3:6]}'""", """f'{x[3:6]}'"""),
        (
            '''f"""{' '.join(A1:A5)}\n"""''',
            '''f"""{' '.join(N_[0, 0, 0, 4, 0])}\n"""''',
        ),
        ("""f'{7:9} {7:b}'""", """f'{N_[0, -1, 6, 8, 0]} {7:b}'"""),
        ("""f'{A1:.2f}'""", """f'{N_[0, 0, 0]:.2f}'"""),
        ("""f'{A1:2f}'""", """f'{N_[0, -1, 0, 1, 0]f}'"""),
        (
            """f'Percentage of true positive: {perc:.2%}'""",
            """f'Percentage of true positive: {perc:.2%}'""",
        ),
        ("""f'She says {A1:>10}'""", """f'She says {N_[0, 0, 0]:>10}'"""),
        ("""f'She says {A1:10}'""", """f'She says {N_[0, -1, 0, 9, 0]}'"""),
        ("""f'{{hello}} = {hello}'""", """f'{{hello}} = {hello}'"""),
        ("""f'{{A1}} = {A1}'""", """f'{{A1}} = {N_[0, 0, 0]}'"""),
        ("""f'{{A1}} = {{{A1}}}'""", """f'{{A1}} = {{{N_[0, 0, 0]}}}'"""),
        ("""f'{A1:^11}'""", """f'{N_[0, 0, 0]:^11}'"""),
        ("""f'{A1:B3:*^11}'""", """f'{N_[0, 1, 0, 2, 0]:*^11}'"""),
        ("""f'{A1:_}'""", """f'{N_[0, 0, 0]:_}'"""),
        ("""f'{A1:,}'""", """f'{N_[0, 0, 0]:,}'"""),
        ("""f'{A1:,.3f}'""", """f'{N_[0, 0, 0]:,.3f}'"""),
        ("""f'{A1:E}'""", """f'{N_[0, 4, 0, -1, 0]}'"""),
        ("""fr'{A1}'""", """fr'{N_[0, 0, 0]}'"""),
        ("""FR'{A1}'""", """FR'{N_[0, 0, 0]}'"""),
        ("""rf'{A1}'""", """rf'{N_[0, 0, 0]}'"""),
        ("""RF'{A1}'""", """RF'{N_[0, 0, 0]}'"""),
        ("""rF'{A1}'""", """rF'{N_[0, 0, 0]}'"""),
        ("""Rf'{A1}'""", """Rf'{N_[0, 0, 0]}'"""),
        ('''f"{'Sheet 2'!A1:A3}"''', '''f"{N_[0, 0, 0, 2, 'Sheet 2']}"'''),
        ('''f"{Sheet1!A1:A3}"''', '''f"{N_[0, 0, 0, 2, 'Sheet1']}"'''),
        (
            """print(f"Test string: {len(foo)}, with {bar:,} parameters")\ndef fn():\n    pass""",
            """print(f"Test string: {len(foo)}, with {bar:,} parameters")\n\n\ndef fn():\n    pass""",
        ),
    ],
)
def test_f_strings(expression, result):
    assert compile_expression(expression).compiled_code == result


INSERT_ROW_AT_INDEX_1 = Transformation(
    Dimension.ROW, SheetTransform.INSERT_BEFORE, 1, 1, 0
)
INSERT_COL_AT_INDEX_1 = Transformation(
    Dimension.COL, SheetTransform.INSERT_BEFORE, 1, 1, 0
)


@pytest.mark.parametrize(
    ("raw_code", "transform", "result"),
    [
        ("A:A", INSERT_ROW_AT_INDEX_1, True),
        ("A:A", INSERT_COL_AT_INDEX_1, False),
        ("1:1", INSERT_ROW_AT_INDEX_1, False),
        ("1:1", INSERT_COL_AT_INDEX_1, True),
        ("A3:A", INSERT_ROW_AT_INDEX_1, False),
        ("C1:1", INSERT_COL_AT_INDEX_1, False),
    ],
)
def test_crosses_infinite_range(raw_code, transform, result):
    tokens = tokenize_with_ranges(raw_code)
    assert transform_crosses_infinite_range(tokens, transform) == result


@pytest.mark.parametrize(
    ("raw_code", "result"),
    [
        ("LET(x, 1, x ** 2)", "(lambda x: x**2)(1)"),
        (
            "LET(x, 5, SUM(x, 1))",
            "(lambda x: SUM(x, 1))(5)",
        ),
        # Nested LET
        (
            "LET(x,1,y,LET(z,2,z*3),x+y)",
            "(lambda x, y: x + y)(1, (lambda x: (lambda z: z * 3)(2))(1))",
        ),
        (
            "LET(x,2,q, LET(y,3,w,LET(z,4,SUM(z,10)),MIN(y,w)),MAX(x,q))",
            "(lambda x, q: MAX(x, q))(2, (lambda x: (lambda y, w: MIN(y, w))(3, (lambda y: (lambda z: SUM(z, 10))(4))(3)))(2))",
        ),
        (
            "LET(x,SQRT(LET(q, 10000, q/100)),y,MIN(x,20,5,30),POWER(x,y))",
            "(lambda x, y: POWER(x, y))(SQRT((lambda q: q / 100)(10000)), (lambda x: MIN(x, 20, 5, 30))(SQRT((lambda q: q / 100)(10000))))",
        ),
        # Arguments depend on previous
        (
            "LET(x,2,y,POWER(3,x),MAX(x,y))",
            "(lambda x, y: MAX(x, y))(2, (lambda x: POWER(3, x))(2))",
        ),
        # Combo
        (
            "LET(x,1,y,x+5,z,LET(a,x+y,b,2,a*b),x+y+z)",
            "(lambda x, y, z: x + y + z)(1, (lambda x: x + 5)(1), (lambda x, y: (lambda a, b: a * b)(x + y, (lambda a: 2)(x + y)))(1, (lambda x: x + 5)(1)))",
        ),
    ],
)
def test_let(raw_code, result):
    assert compile_expression(raw_code).compiled_code == result


@pytest.mark.parametrize(
    ("raw_code", "result"),
    [
        ("=IFERROR(1/0, 2)", "=IFERROR(lambda: 1/0, 2)"),
    ],
)
def test_iferror(raw_code, result):
    assert compile_expression(raw_code).compiled_code == result
