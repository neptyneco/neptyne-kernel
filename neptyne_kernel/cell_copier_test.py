import asyncio

import pytest

from .cell_address import Address
from .cell_copier import (
    almost_int,
    cells_to_grid,
    extend_cells,
    pre_copy_adjust,
)
from .test_utils import a1


def extend_cells_sync(
    populate_from: list[tuple[Address, str]],
    populate_to_start: Address,
    populate_to_end: Address,
    context: list[str] | None = None,
) -> list[tuple[Address, str]]:
    loop = asyncio.get_event_loop()
    res = loop.run_until_complete(
        extend_cells(
            populate_from,
            populate_to_start,
            populate_to_end,
            context,
        )
    )
    return res


def test_dict_to_grid():
    assert cells_to_grid([(a1("A1"), "0")]) == (0, 0, [["0"]])


@pytest.mark.parametrize(
    ("formula", "offset", "result"),
    [
        ("=A1 + A$2", (0, 1), "=A2 + A$2"),
        ("=A1 +$A$2", (0, 1), "=A2 +$A$2"),
        ("=A1  +  $A$2", (1, 1), "=B2  +  $A$2"),
        ("=Z1 + $A$2", (1, 1), "=AA2 + $A$2"),
        ("=A1:A + A$2:A10", (0, 1), "=A2:A + A$2:A11"),
        ("=A1:2 + $A$2", (0, 1), "=A2:3 + $A$2"),
        ("=A$1:$2 + A2", (0, 1), "=A$1:$2 + A3"),
        ("=A1:B + $A$2", (1, 1), "=B2:C + $A$2"),
        ("=A1:3 + A2:C9", (1, 1), "=B2:4 + B3:D10"),
        ("=Z99:3 + A2:C9", (1, 1), "=AA100:4 + B3:D10"),
        ("=AA100:3 + B2:C9", (-1, -1), "=Z99:2 + A1:B8"),
    ],
)
def test_pre_copy_adjust(formula, offset, result):
    target_cell = Address(*offset, 0)
    adjusted = pre_copy_adjust(a1("A1"), [(target_cell, formula, {})])
    assert adjusted[0][1] == result


def test_pre_copy_adjust_bulk():
    def pre_copy_to_dict(pca):
        return dict([(k, v) for k, v, _ in pca])

    c1 = pre_copy_to_dict(
        pre_copy_adjust(
            a1("A1"),
            [(a1("B2"), "=C1", {}), (a1("C2"), 10, {}), (a1("D2"), "Hello", {})],
        )
    )
    assert c1[a1("B2")] == "=D2"
    assert c1[a1("C2")] == 10
    assert c1[a1("D2")] == "Hello"
    c2 = pre_copy_to_dict(
        pre_copy_adjust(
            a1("A1"),
            [(a1("B2"), "=C1", {}), (a1("C2"), "=D2", {}), (a1("D2"), "=E2", {})],
        )
    )
    assert c2[a1("B2")] == "=D2"
    assert c2[a1("C2")] == "=F3"
    assert c2[a1("D2")] == "=H3"

    c3 = pre_copy_to_dict(
        pre_copy_adjust(
            a1("B3"),
            [
                (a1("B4"), "=fn(A1)", {}),
                (a1("B5"), "=fn(A1)", {}),
                (a1("B6"), "=fn(A1)", {}),
            ],
        )
    )
    assert c3[a1("B4")] == "=fn(A2)"
    assert c3[a1("B5")] == "=fn(A3)"
    assert c3[a1("B6")] == "=fn(A4)"

    c4 = pre_copy_to_dict(
        pre_copy_adjust(
            a1("A1"),
            [
                (a1("A2"), "=fn(C10)", {}),
                (a1("A3"), "=fn(C10)", {}),
                (a1("A4"), "=fn(C10)", {}),
            ],
        )
    )
    assert c4[a1("A2")] == "=fn(C11)"
    assert c4[a1("A3")] == "=fn(C12)"
    assert c4[a1("A4")] == "=fn(C13)"


def test_pre_copy_adjust_attributes():
    def pre_copy_to_dict(pca):
        return dict([(k, v) for k, _, v in pca])

    c1 = pre_copy_to_dict(
        pre_copy_adjust(
            a1("A1"),
            [
                (a1("B2"), "=C1", {"attr": "value"}),
                (a1("C2"), 10, {}),
                (a1("D2"), "Hello", None),
            ],
        )
    )

    assert c1[a1("B2")] == {"attr": "value"}
    assert c1[a1("C2")] is None
    assert c1[a1("D2")] is None


def test_almost_int():
    assert not almost_int(0.8)
    assert almost_int(0.999999999)
    assert almost_int(10.00000001)
    assert almost_int(-8.00000001)
    assert almost_int(-10000.999999999)
    assert not almost_int(4.9999)


def test_extend_cells_sync():
    assert extend_cells_sync([], a1("A1"), a1("A2")) == []
    assert dict(extend_cells_sync([(a1("C3"), "1")], a1("D3"), a1("E3"))) == {
        a1("D3"): "1",
        a1("E3"): "1",
    }
    assert dict(
        extend_cells_sync(
            [(a1("A1"), "1"), (a1("A2"), "2"), (a1("A3"), "3")],
            a1("A4"),
            a1("A4"),
        )
    ) == {a1("A4"): "4"}
    e = dict(
        extend_cells_sync(
            [(a1("A1"), "1"), (a1("B1"), "2"), (a1("C1"), "2")],
            a1("D1"),
            a1("E1"),
        )
    )
    assert float(e[a1("D1")]) == pytest.approx(8 / 3)
    assert float(e[a1("E1")]) == pytest.approx(9.5 / 3)
    e = dict(
        extend_cells_sync(
            [
                (a1("A1"), "=A2"),
                (a1("B1"), "2"),
                (a1("C1"), "C1"),
                (a1("D1"), "3"),
            ],
            a1("E1"),
            a1("G1"),
        )
    )
    assert e[a1("E1")] == "=E2"
    assert e[a1("F1")] == "4"
    assert e[a1("G1")] == "C1"


def test_extend_cells_sync_up():
    assert dict(
        extend_cells_sync(
            [
                (a1("A3"), "3"),
                (a1("A4"), "4"),
                (a1("A5"), "5"),
            ],
            a1("A1"),
            a1("A2"),
        )
    ) == {a1("A1"): "1", a1("A2"): "2"}

    assert dict(
        extend_cells_sync(
            [
                (a1("C1"), "3"),
                (a1("D1"), "4"),
                (a1("E1"), "5"),
            ],
            a1("A1"),
            a1("B1"),
        )
    ) == {a1("A1"): "1", a1("B1"): "2"}

    assert dict(
        extend_cells_sync(
            [
                (a1("A3"), "=B3"),
            ],
            a1("A1"),
            a1("A2"),
        )
    ) == {a1("A1"): "=B1", a1("A2"): "=B2"}

    assert dict(
        extend_cells_sync(
            [(a1("A3"), "=B1")],
            a1("A1"),
            a1("A2"),
        )
    ) == {a1("A1"): "=REF_ERROR", a1("A2"): "=REF_ERROR"}
