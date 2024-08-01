import base64
import pickle
import types
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from .cell_address import Address, Range
from .cell_execution_graph import CellExecutionGraph
from .cell_range import CellRange
from .dash import hash_function, shape
from .formulas import AVERAGE
from .formulas.helpers import assert_equal
from .neptyne_protocol import Dimension, MessageTypes, SheetTransform
from .ops import ClearOp, ExecOp
from .test_utils import a1
from .transformation import Transformation
from .tyne_model.cell import SheetCell
from .tyne_model.jupyter_notebook import Output, OutputType
from .tyne_model.sheet import TyneSheets


def process_code_update(dash, cell_id, code, should_compile=True):
    """Simulate a code update from the client, but don't perform the execution"""
    if isinstance(cell_id, str):
        cell_id = a1(cell_id)

    dash.get_or_create_cell_meta(cell_id).raw_code = code
    if should_compile:
        dash.compile_and_update_cell_meta(cell_id)


def trace_cells_depends_on(dash, cell_id: Address, seen=None):
    if seen is None:
        seen = set()
    if cell_id in seen:
        # cycle:
        raise ValueError("Circular dependency between cells not allowed for functions")
    res = [cell_id]
    seen.add(cell_id)
    for other_cell in dash.graph.depends_on.get(cell_id, []):
        res.extend(trace_cells_depends_on(dash, other_cell, seen))
    seen.remove(cell_id)
    return res


def test_del_on_empty(dash):
    dash[Address(1, 0, 0)] = 1
    dash[Address(1, 0, 0)] = ""
    assert "B1" not in dash.cells[0]


def test_neptyne_load_values(dash):
    def a1c(s: str):
        return a1(s)

    def lst_to_content(lst: list):
        sheets = TyneSheets()
        sheet = sheets.sheets[0]
        for addr, value in lst:
            sheet.cells[addr] = SheetCell(
                cell_id=addr,
                output=Output(
                    data=value,
                    execution_count=-1,
                    metadata=None,
                    output_type=OutputType.EXECUTE_RESULT.value,
                    name=None,
                    text=None,
                    ename=None,
                    evalue=None,
                    traceback=None,
                ),
            )
        return sheets

    values = [
        (a1c("A1"), {"text/plain": "'hello'"}),
        (a1c("B1"), {"text/plain": "20"}),
    ]
    dash.load_values(lst_to_content(values))

    assert dash[Address(0, 0, 0)] == "hello"
    assert dash[Address(1, 0, 0)] == 20

    dash.load_values(
        lst_to_content(
            [
                (a1c("A3"), {"text/plain": "'Date'"}),
                (a1c("B3"), {"text/plain": "'Cases'"}),
                (a1c("C3"), {"text/plain": "'Deaths'"}),
                (a1c("E1"), {"text/plain": "'OK'"}),
                (a1c("A4"), {"text/plain": "'26 May 2020'"}),
                (a1c("B4"), {"text/plain": "106475"}),
                (a1c("C4"), {"text/plain": "5283"}),
                (a1c("A5"), {"text/plain": "'25 May 2020'"}),
                (a1c("B5"), {"text/plain": "92060"}),
                (a1c("C5"), {"text/plain": "4048"}),
                (a1c("A6"), {"text/plain": "'24 May 2020'"}),
                (a1c("B6"), {"text/plain": "90184"}),
                (a1c("C6"), {"text/plain": "3096"}),
                (a1c("A7"), {"text/plain": "'23 May 2020'"}),
                (a1c("B7"), {"text/plain": "96121"}),
                (a1c("C7"), {"text/plain": "3503"}),
                (a1c("A8"), {"text/plain": "''"}),
                (a1c("B8"), {"text/plain": "''"}),
                (a1c("C8"), {"text/plain": "''"}),
            ]
        )
    )

    assert dash[Address(2, 6, 0)] == 3503


def test_set_region(dash):
    dash[Address(0, 0, 0)] = [1, 2, 3]

    assert dash[Address(0, 0, 0)] == 1
    assert dash[Address(0, 2, 0)] == 3

    dash[Address(1, 0, 0)] = [[1, 2, 3], [4, 5, 6]]
    assert dash[Address(1, 0, 0)] == 1
    assert dash[Address(1, 1, 0)] == 4
    assert dash[Address(3, 1, 0)] == 6


def test_set_dict(dash):
    dash[a1("A1")] = {"a": 1, "b": 2, "c": 3}

    assert dash[a1("A1")] == "a"
    assert dash[a1("B1")] == 1
    assert dash[a1("B3")] == 3


def test_set_2d_array(dash):
    dash[a1("A1")] = [[1, 2, 3], [4, 5, 6]]
    assert dash[a1("A1")] == 1
    assert dash[a1("A2")] == 4
    assert dash[a1("C2")] == 6


@contextmanager
def message_capturer():
    import builtins

    captured = []
    kernel = types.SimpleNamespace(
        iopub_socket=None,
        send_response=lambda channel, msg_type, contents: captured.append(
            (msg_type, contents)
        ),
    )

    builtins.get_ipython = lambda: types.SimpleNamespace(kernel=kernel)
    yield captured
    del builtins.get_ipython


def test_available_functions(dash):
    dash.silent = False
    assert dash.is_callable_from_client("available_functions")
    with message_capturer() as captured:
        dash.available_functions(prefix="S")
    assert captured[0][0] == MessageTypes.RPC_RESULT.value


def test_available_functions_in_module(dash):
    from .neptyne_api import ai

    dash.silent = False
    dash.shell.user_ns = {"ai": ai, "other_fn": test_available_functions_in_module}
    assert dash.is_callable_from_client("available_functions")
    with message_capturer() as captured:
        dash.available_functions(prefix="ai.r")
    msg_type, contents = captured[0]
    assert msg_type == MessageTypes.RPC_RESULT.value
    method_name, doc_string, params = contents["result"][0]
    assert method_name == "research"


def test_generators(dash):
    dash[Address(0, 0, 0)] = range(10)
    dash[Address(1, 0, 0)] = map(lambda x: x + 1, dash[Range(0, 0, 0, 9, 0)])

    def add_one(r):
        for a in r:
            yield a + 1

    dash[Address(2, 0, 0)] = add_one(dash[Range(1, 1, 0, 9, 0)])

    assert dash[Address(2, 9, 0)] == 11


def test_clear_values(dash):
    dash[a1("A1")] = range(1, 11)
    assert dash[a1("A10")] == 10
    dash.clear_cells_internal(
        {Address.from_a1_or_str("A" + str(i)) for i in range(1, 11)}
    )

    assert dash[a1("A10")].is_empty()

    dash[a1("A1")] = "X"
    dash[a1("A1")] = []
    assert dash[a1("A1")].is_empty()


def test_cell_attributes(dash):
    dash[a1("A1")] = ","
    assert dash[a1("A1")].join(["a", "b"]) == "a,b"


def test_shape():
    assert shape("hello") == (1, 1)
    assert shape(25) == (1, 1)
    assert shape([1, 2, 3]) == (3, 1)
    assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)
    assert shape(("a", "b")) == (2, 1)
    assert shape([[1, 2, 3]]) == (1, 3)
    assert shape([{"foo": "bar"}]) == (1, 1)
    assert shape([[1, 2], [1, 2, 3, 4]]) == (2, 4)
    assert shape([[1, 2], "world"]) == (2, 2)
    assert shape([None, [1, 2]]) == (2, 2)
    assert shape(np.ones((4, 2))) == (4, 2)
    df = pd.DataFrame(
        {
            "num_legs": [2, 4, 8, 0],
            "num_wings": [2, 0, 0, 0],
            "num_specimen_seen": [10, 2, 1, 8],
        },
        index=["falcon", "dog", "spider", "fish"],
    )
    assert shape(df) == (5, 4)


def test_dash_set_slice(dash):
    dash[a1("A1:C2")] = [[1, 2, 3], [4, 5, 6]]
    assert_equal(dash[a1("B2")], 5)
    assert_equal(dash[a1("A1:C1")], (1, 2, 3))
    assert_equal(dash[a1("A1:A2")], (1, 4))

    with pytest.raises(ValueError):
        dash[a1("A1:B3")] = [[1, 2, 3], [4, 5, 6]]

    with pytest.raises(ValueError):
        dash[a1("A1:C3")] = range(3)

    dash[a1("A1:C1")] = range(1, 4)
    assert_equal(dash[a1("B1")], 2)

    dash[a1("N10:Q10")] = range(4)
    assert_equal(dash[a1("Q10")], 3)

    dash[a1("C10:C13")] = range(4)
    assert dash[a1("C13")] == 3


def test_dash_cell_range_set_item(dash):
    dash[a1("A1:F6")] = [[ch1 + ch2 for ch1 in "ABCDEF"] for ch2 in "123456"]

    r = dash[a1("B2:D4")]
    r[(0, 0)] = "2B"

    assert dash[a1("B2")] == "2B"
    r[0][0] = "B2"
    assert dash[a1("B2")] == "B2"

    r[1:3, 1:3] = [["1", "2"], ["2", "3"]]
    assert dash[a1("D4")] == "3"


def test_slicing(dash):
    dash[a1("A1:F6")] = [[ch1 + ch2 for ch1 in "ABCDEF"] for ch2 in "123456"]

    assert_equal(dash[a1("A1:F6")][0, 2], "C1")

    assert_equal(dash[Range(0, -1, 0, 0, 0)], ["A1", "B1", "C1", "D1", "E1", "F1"])
    assert_equal(dash[Range(0, -1, 0, 0, 0)][-1], "F1")
    assert_equal(dash[a1("A1:F1")][-1], "F1")

    assert_equal(dash[a1("A1:F1")], ["A1", "B1", "C1", "D1", "E1", "F1"])
    assert_equal(dash[a1("A1:F1")][1:], ["B1", "C1", "D1", "E1", "F1"])
    assert_equal(dash[a1("A1:F1")][:-1], ["A1", "B1", "C1", "D1", "E1"])

    t = dash[a1("A1:F6")] + "!"
    assert_equal(t[0:2, 1], ["B1!", "B2!"])
    assert_equal(t[1, 0:2], ["A2!", "B2!"])

    assert_equal(dash[a1("A1:F6")][0:2, 1], ["B1", "B2"])

    assert_equal(dash[a1("A1:F6")][0][2], "C1")


def test_average(dash):
    dash[a1("A1")] = [1, 2, 3]
    assert AVERAGE(dash[a1("A1:A3")]) == 2


@pytest.mark.parametrize(
    "cell_range_or_value, filename, expected_payload, expected_mime_type",
    [
        ("A1", None, "A1", "text/plain"),
        ("A1:A3", None, "A1\nA2\nA3", "text/plain"),
        ("A1:B3", None, "A1,B1\nA2,B2\nA3,B3", "text/csv"),
        (
            "A1:B3",
            "cells.json",
            '[["A1", "B1"], ["A2", "B2"], ["A3", "B3"]]',
            "application/json",
        ),
        (["data1", "data2"], "data.csv", "data1\ndata2", "text/csv"),
        ({"key": "value"}, "data.json", '{"key": "value"}', "application/json"),
        (["data1", "data2"], None, '["data1", "data2"]', "application/json"),
        (["data1", "data2"], "data.unknown", '["data1", "data2"]', "application/json"),
        ("A1:A1", None, "A1", "text/plain"),
        ({1, 2, 3}, None, "1\n2\n3", "text/plain"),
        ({1, 2, 3}, "set.json", "[1, 2, 3]", "application/json"),
    ],
)
def test_download(
    dash, cell_range_or_value, filename, expected_payload, expected_mime_type
):
    dash[a1("A1:F6")] = [[ch1 + ch2 for ch1 in "ABCDEF"] for ch2 in "123456"]

    replied = None
    mime_type = None

    def get_reply(_msg_type, d):
        nonlocal replied, mime_type
        replied = base64.b64decode(d["payload"].encode("utf8")).decode("utf8").strip()
        mime_type = d["mimetype"]

    dash.reply_to_client = get_reply

    if isinstance(cell_range_or_value, str):
        value = dash[a1(cell_range_or_value)]
    else:
        value = cell_range_or_value

    dash.initiate_download(value, filename)
    assert replied == expected_payload
    assert mime_type == expected_mime_type


def test_statements():
    statements = [
        ClearOp(["a1", "b1"]),
        ExecOp("c1", "=a1"),
        ClearOp(["f1", "f4"]),
        ExecOp("g1", "=b1"),
    ]

    params = repr(statements)

    statements_out = eval(params)

    assert statements == statements_out


def test_infinite_ranges(dash):
    dash[a1("B5")] = 1
    assert dash[Range(min_col=1, max_col=1, min_row=4, max_row=-1, sheet=0)][
        100
    ].is_empty()
    dash[a1("C10")] = 1
    assert len(dash[Range(min_col=2, max_col=3, min_row=0, max_row=-1, sheet=0)]) == 10
    assert len(dash[Range(min_col=2, max_col=2, min_row=0, max_row=-1, sheet=0)]) == 10
    assert len(dash[Range(min_col=0, max_col=-1, min_row=9, max_row=9, sheet=0)]) == 3
    assert len(dash[Range(min_col=1, max_col=1, min_row=0, max_row=-1, sheet=0)]) == 5
    assert len(dash[Range(min_col=0, max_col=-1, min_row=4, max_row=4, sheet=0)]) == 2

    assert dash[Range(min_col=1, max_col=1, min_row=0, max_row=-1, sheet=0)][4] == 1


def test_pickle(dash):
    dash[a1("A1")] = 1
    dash[a1("B1")] = 1.0
    dash[a1("C1")] = "1"
    assert pickle.loads(pickle.dumps(dash[a1("A1")])) == 1
    assert pickle.loads(pickle.dumps(dash[a1("B1")])) == 1.0
    assert pickle.loads(pickle.dumps(dash[a1("C1")])) == "1"
    assert pickle.loads(pickle.dumps(dash[a1("D1")])) is None


def test_max_column(dash):
    with pytest.raises(IndexError):
        dash[a1("ZY1")]
    with pytest.raises(IndexError):
        dash[a1("AAA1")] = 1
    with pytest.raises(IndexError):
        dash[a1("XXX1:XXX2")]
    with pytest.raises(IndexError):
        dash[a1("PLAYER1:PLAYER2")] = CellRange([None, None])


def test_get_code(dash):
    process_code_update(dash, "A1", "10 + 20")
    assert dash.cell_meta.get(a1("A1")).raw_code == "10 + 20"


def run_full_graph(exec_graph: CellExecutionGraph):
    while statements := exec_graph.ready_statements():
        for statement in statements:
            yield statement
            if isinstance(statement, ExecOp):
                exec_graph.done(statement.address)
            elif isinstance(statement, ClearOp):
                exec_graph.done(*statement.to_clear)
            else:
                raise ValueError(f"Unknown statement type {statement}")


def test_dependency_graph(
    dash,
):
    process_code_update(dash, "A1", "500")

    assert len(dash.graph.feeds_into.get(a1("A1"), [])) == 0

    process_code_update(dash, "B1", "=A1")
    process_code_update(dash, "C1", "=A1")
    process_code_update(dash, "D1", "=B1 + C1")
    process_code_update(dash, "E1", "100")

    exec_graph = dash.get_execution_graph({a1("A1")})
    ops = [*run_full_graph(exec_graph)]
    assert len(ops) == 4

    exec_graph = dash.get_execution_graph({a1("B1"), a1("C1")})
    keys = [op.address for op in run_full_graph(exec_graph) if isinstance(op, ExecOp)]
    assert len(keys) == len(set(keys))

    addrs = trace_cells_depends_on(dash, a1("D1"))
    assert set(addrs) == {
        a1("A1"),
        a1("B1"),
        a1("C1"),
        a1("D1"),
    }

    addrs = trace_cells_depends_on(dash, a1("A1"))
    assert set(addrs) == {a1("A1")}

    addrs = trace_cells_depends_on(dash, a1("C1"))
    assert set(addrs) == {a1("C1"), a1("A1")}

    process_code_update(dash, "A1", "=D1")
    res = dash.get_execution_graph({a1("A1")})
    for op in run_full_graph(res):
        assert op.expression.find("REF_ERROR") == 0


def test_exec_graph(dash):
    process_code_update(dash, "A1", "=B1 + C1 + B2")
    process_code_update(dash, "B1", "=C1 + D1")
    process_code_update(dash, "B2", "=C1 + D1")
    process_code_update(dash, "C1", "=E1")
    process_code_update(dash, "D1", "=E1")

    def cells_to_run_on_update(dash, addresses):
        cells = []
        for op in run_full_graph(dash.get_execution_graph(addresses)):
            if isinstance(op, ExecOp):
                cells.append(op.address)
            elif isinstance(op, ClearOp):
                cells.extend(op.to_clear)
            else:
                raise ValueError(f"Unknown statement type {op}")
        return cells

    assert cells_to_run_on_update(dash, {a1("B1")}) == [
        a1("B1"),
        a1("A1"),
    ]

    order = {
        cell: idx for idx, cell in enumerate(cells_to_run_on_update(dash, {a1("E1")}))
    }

    assert order[a1("A1")] == max(order.values()), [
        cell for cell, idx in order.items() if idx == max(order.values())
    ]
    assert order[a1("E1")] == min(order.values())

    assert order[a1("C1")] < order[a1("B1")]


def test_compute_inverse_insert_delete_transformation(dash):
    process_code_update(dash, "A1", "NON_EMPTY")
    process_code_update(dash, "A2", "NON_EMPTY")

    transform = Transformation(Dimension.ROW, SheetTransform.DELETE, 1, 1, 0)

    (
        inverse_transform,
        cells_to_populate,
    ) = dash.compute_inverse_insert_delete_transformation(transform)

    (
        double_inverse_transform,
        double_inverse_transform_cells_to_populate,
    ) = dash.compute_inverse_insert_delete_transformation(inverse_transform)

    assert transform == double_inverse_transform
    assert inverse_transform.operation == SheetTransform.INSERT_BEFORE
    assert inverse_transform.index == 1
    assert len(cells_to_populate) == 1
    assert cells_to_populate[0]["cell_id"] == a1("A2").to_coord()
    assert len(double_inverse_transform_cells_to_populate) == 0


def test_recompile_everything(dash):
    process_code_update(dash, "A1", "500", False)

    process_code_update(dash, "B1", "=A1", False)
    process_code_update(dash, "C1", "=A1", False)
    process_code_update(dash, "D1", "=B1 + C1", False)
    process_code_update(dash, "E1", "100", False)

    assert len(dash.sheet_cell_for_address(a1("A1")).feeds_into) == 0

    dash.recompile_everything()

    assert len(dash.sheet_cell_for_address(a1("A1")).feeds_into) == 2

    graph = trace_cells_depends_on(dash, a1("D1"))
    assert set(graph) == {
        a1("A1"),
        a1("B1"),
        a1("C1"),
        a1("D1"),
    }

    graph = trace_cells_depends_on(dash, a1("A1"))
    assert set(graph) == {a1("A1")}

    graph = trace_cells_depends_on(dash, a1("C1"))
    assert set(graph) == {a1("C1"), a1("A1")}


def test_get_autofill_context(dash):
    dash[a1("B1")] = "Hello"
    ctx = dash.get_autofill_context(1, 0, 0, 0, transpose=False)
    assert not ctx
    ctx = dash.get_autofill_context(2, 0, 0, 0, transpose=False)
    assert ctx == ["Hello"]

    dash[a1("A10")] = "Hours"
    dash[a1("B10")] = "Rate"
    dash[a1("C10")] = "Billing"

    ctx = dash.get_autofill_context(10, 0, 2, 0, transpose=True)
    assert ctx == ["Hours", "Rate", "Billing"]


def test_hash_function():
    def simple(x, y):
        return x + y

    hash_add = hash_function(simple)

    def simple(x, y):
        return x * y

    assert hash_add != hash_function(simple)

    def simple(x, y):
        return x + y

    assert hash_add == hash_function(simple)


def test_hash_function_with_list_comprehension():
    def list_comprehension(x):
        return [x + i for i in range(10)]

    hash_before = hash_function(list_comprehension)

    def list_comprehension(x):
        return [x + i for i in range(10)]

    assert hash_before == hash_function(list_comprehension)


def test_hasher_with_int_collision():
    # hash(-1) == hash(-2) in Python:
    def func():
        return -1

    hash_minus_one = hash_function(func)

    def func():
        return -2

    assert hash_minus_one != hash_function(func)


def test_hash_function_renamed_vars():
    def func():
        x = 1
        return x

    hash_var1 = hash_function(func)

    def func():
        y = 1
        return y

    assert hash_var1 == hash_function(func)


def test_hash_function_inner_outer():
    def outer():
        def inner():
            return 1

        return inner()

    hash_outer = hash_function(outer)

    def outer():
        def inner():
            return 2

        return inner()

    assert hash_outer != hash_function(outer)


def test_upload_csv(dash):
    df = pd.DataFrame(
        {
            "country": ["Netherlands", "Germany", "France", "Japan", "United States"],
            "capital": ["Amsterdam", "Berlin", "Paris", "Tokyo", "Washington D.C."],
            "population": [
                17_280_000,
                83_020_000,
                67_120_000,
                126_800_000,
                328_200_000,
            ],
        },
    )

    csv = df.to_csv(index=False)

    def fake_initiate_upload(prompt, accept):
        return csv.encode("utf8"), "countries.csv"

    dash.initiate_upload = fake_initiate_upload

    sheet_id = dash.upload_csv_to_new_sheet("Upload a csv")

    assert dash.sheets[sheet_id].name == "countries"

    assert (dash[Range(0, 2, 0, 0, 1)] == ["country", "capital", "population"]).all()
