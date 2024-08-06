from .neptyne_protocol import Dimension, SheetTransform
from .test_utils import a1
from .transformation import (
    Transformation,
    insert_delete_content_to_sheet_transform,
    sheet_transform_to_insert_delete_content,
)


def test_compute_add_delete_row_col_kv_update_dict():
    cells = [a1(x) for x in ("A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3")]

    # Insert row before B2
    insert = Transformation(Dimension.ROW, SheetTransform.INSERT_BEFORE, 1, 1, 0)
    (
        to_delete_set,
        to_unlink_set,
        to_update_dict,
    ) = insert.compute_add_delete_row_col_kv_update_dict(cells)
    assert to_delete_set == {a1("A2"), a1("B2"), a1("C2")}
    assert to_unlink_set == set()
    assert to_update_dict == {
        a1("A2"): a1("A3"),
        a1("B2"): a1("B3"),
        a1("C2"): a1("C3"),
        a1("A3"): a1("A4"),
        a1("B3"): a1("B4"),
        a1("C3"): a1("C4"),
    }

    # Insert col after B2
    insert = Transformation(Dimension.COL, SheetTransform.INSERT_BEFORE, 2, 1, 0)
    (
        to_delete_set,
        to_unlink_set,
        to_update_dict,
    ) = insert.compute_add_delete_row_col_kv_update_dict(cells)
    assert to_delete_set == {a1("C1"), a1("C2"), a1("C3")}
    assert to_unlink_set == set()
    assert to_update_dict == {
        a1("C1"): a1("D1"),
        a1("C2"): a1("D2"),
        a1("C3"): a1("D3"),
    }

    # Insert col after C2
    insert = Transformation(Dimension.COL, SheetTransform.INSERT_BEFORE, 3, 1, 0)
    (
        to_delete_set,
        to_unlink_set,
        to_update_dict,
    ) = insert.compute_add_delete_row_col_kv_update_dict(cells)
    assert to_delete_set == set()
    assert to_unlink_set == set()
    assert to_update_dict == dict()

    # Delete row at B2
    delete = Transformation(Dimension.ROW, SheetTransform.DELETE, 1, 1, 0)
    (
        to_delete_set,
        to_unlink_set,
        to_update_dict,
    ) = delete.compute_add_delete_row_col_kv_update_dict(cells)
    assert to_delete_set == {a1("A3"), a1("B3"), a1("C3")}
    assert to_unlink_set == {a1("A2"), a1("B2"), a1("C2")}
    assert to_update_dict == {
        a1("A3"): a1("A2"),
        a1("B3"): a1("B2"),
        a1("C3"): a1("C2"),
    }


def test_sheet_transform_to_insert_delete_content():
    cells_to_populate = [{"cell_id": a1("A2"), "value": "1"}]

    # Go back and forth to confirm reversibility
    transform = Transformation(Dimension.ROW, SheetTransform.DELETE, 1, 1, 0)
    insert_delete_content = sheet_transform_to_insert_delete_content(
        transform,
        cells_to_populate,
    )

    (
        new_transform,
        new_cells_to_populate,
    ) = insert_delete_content_to_sheet_transform(insert_delete_content)

    new_insert_delete_content = sheet_transform_to_insert_delete_content(
        new_transform,
        new_cells_to_populate,
    )

    assert transform == new_transform
    assert len(new_cells_to_populate) == 1
    assert new_cells_to_populate[0]["cell_id"] == a1("A2")
    assert (
        insert_delete_content.selected_index == new_insert_delete_content.selected_index
    )
    assert (
        insert_delete_content.sheet_transform
        == new_insert_delete_content.sheet_transform
    )
