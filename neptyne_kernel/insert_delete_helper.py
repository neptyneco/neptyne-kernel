from typing import TYPE_CHECKING, Any, Iterable

from .cell_address import Address, Range
from .expression_compiler import (
    process_sheet_transformation,
    tokenize_with_ranges,
    transform_crosses_infinite_range,
)
from .mime_handling import outputs_to_value
from .neptyne_protocol import (
    CellAttribute,
    CellChange,
    Dimension,
    MessageTypes,
    SheetAttribute,
    SheetTransform,
)
from .transformation import (
    Transformation,
    sheet_transform_to_insert_delete_content,
    transformation_is_unbounded,
)
from .tyne_model.cell import CellMetadata

if TYPE_CHECKING:
    from .dash import Dash


def _compute_attribute_transfer_dict(
    dash: "Dash", transformation: Transformation
) -> dict[int, dict[str, Any]]:
    seed_attributes = {}
    for addr, cell in dash.cell_meta.items():
        if addr.sheet != transformation.sheet_id or not cell.attributes:
            continue
        if transformation.boundary and addr not in transformation.boundary:
            continue
        if transformation.dimension == Dimension.ROW:
            if addr.row == transformation.index:
                seed_attributes[addr.column] = cell.attributes
        else:
            if addr.column == transformation.index:
                seed_attributes[addr.row] = cell.attributes

    return seed_attributes


def _transfer_cell_attributes_to_inserted_cells(
    dash: "Dash",
    transformation: Transformation,
    seed_attributes: dict[int, dict[str, Any]],
    changed_cells: set,
) -> None:
    for i in range(transformation.amount):
        index = transformation.index + i
        for j, attributes in seed_attributes.items():
            if transformation.dimension == Dimension.COL:
                address = Address(index, j, transformation.sheet_id)
            else:
                address = Address(j, index, transformation.sheet_id)
            cell_meta = dash.get_or_create_cell_meta(address)
            cell_meta.attributes.update(attributes)
            changed_cells.add(address)


def _compute_undo_message(
    dash: "Dash", transformation: Transformation, send_undo: bool = False
) -> tuple[MessageTypes, dict] | None:
    if send_undo:
        (
            inverse_transform,
            inverse_cells_to_populate,
        ) = dash.compute_inverse_insert_delete_transformation(transformation)
        return (
            MessageTypes.INSERT_DELETE_CELLS,
            sheet_transform_to_insert_delete_content(
                inverse_transform, inverse_cells_to_populate
            ).to_dict(),
        )
    return None


def _maybe_update_grid_size(dash: "Dash", transformation: Transformation) -> None:
    if transformation_is_unbounded(transformation):
        sheet = dash.sheets[transformation.sheet_id]
        transformation_amount = (
            transformation.amount
            if transformation.operation == SheetTransform.INSERT_BEFORE
            else -transformation.amount
        )
        if transformation.dimension == Dimension.COL:
            sheet.n_cols += transformation_amount
        else:
            sheet.n_rows += transformation_amount


def _compute_sheet_attribute_updates(
    dash: "Dash", transformation: Transformation
) -> list[tuple[int, str, Any]]:
    sheet_attribute_updates: list[tuple[int, str, Any]] = []
    if transformation.dimension == Dimension.COL:
        hidden_headers_attribute = "colsHiddenHeaders"
        frozen_count_attribute = "colsFrozenCount"
        header_sizes_attribute = SheetAttribute.COLS_SIZES.value
    else:
        hidden_headers_attribute = "rowsHiddenHeaders"
        frozen_count_attribute = "rowsFrozenCount"
        header_sizes_attribute = SheetAttribute.ROWS_SIZES.value
    dimension_index = 0 if transformation.dimension == Dimension.COL else 1

    if hidden_headers := dash.sheets[transformation.sheet_id].attributes.get(
        hidden_headers_attribute
    ):
        new_hidden_headers = []
        for header in hidden_headers:
            transform_args = (
                (header, 0)
                if transformation.dimension == Dimension.COL
                else (0, header)
            )
            result = transformation.transform(*transform_args)
            if result == Transformation.NO_CHANGE:
                new_hidden_headers.append(header)
            elif result == Transformation.REF_ERROR:
                continue
            else:
                assert isinstance(result, tuple)
                new_hidden_headers.append(result[dimension_index])
        sheet_attribute_updates.append(
            (transformation.sheet_id, hidden_headers_attribute, new_hidden_headers)
        )

    if frozen_count := dash.sheets[transformation.sheet_id].attributes.get(
        frozen_count_attribute
    ):
        if transformation.index < frozen_count:
            frozen_change = (
                transformation.amount
                if transformation.operation == SheetTransform.INSERT_BEFORE
                else -transformation.amount
            )
            frozen_count = max(0, frozen_change + frozen_count)
            sheet_attribute_updates.append(
                (transformation.sheet_id, frozen_count_attribute, frozen_count)
            )

    if header_sizes := dash.sheets[transformation.sheet_id].attributes.get(
        header_sizes_attribute
    ):
        new_header_sizes = {}
        if transformation.operation == SheetTransform.INSERT_BEFORE:
            source_header_size = header_sizes.get(str(transformation.index))
            for index_str, size in header_sizes.items():
                index = int(index_str)
                if index >= transformation.index:
                    new_header_sizes[str(index + transformation.amount)] = size
                else:
                    new_header_sizes[str(index)] = size
            if source_header_size is not None:
                for i in range(transformation.amount):
                    new_header_sizes[str(transformation.index + i)] = source_header_size
        elif transformation.operation == SheetTransform.DELETE:
            transformation_end = transformation.index + transformation.amount
            for index_str, size in header_sizes.items():
                index = int(index_str)
                if index >= transformation_end:
                    new_header_sizes[str(index - transformation.amount)] = size
                elif index < transformation.index:
                    new_header_sizes[str(index)] = size
        sheet_attribute_updates.append(
            (transformation.sheet_id, header_sizes_attribute, new_header_sizes)
        )
    return sheet_attribute_updates


def _update_keys_combined(
    dash: "Dash",
    key_old_to_new: dict[Address, Address],
    to_clear: Iterable[Address],
    to_clear_values: Iterable[Address],
) -> list[Address]:
    # Store values at changing ids, and delete these keys.
    key_new_to_value: dict[Address, Any] = {}

    changes: list[Address] = []
    for old_id, new_id in key_old_to_new.items():
        key_new_to_value[new_id] = (
            dash.cells[old_id.sheet].pop(old_id, None),
            dash.cell_meta.pop(old_id, None),
            dash.graph.calculated_by.pop(old_id, None),
            dash.graph.feeds_into.pop(old_id, None),
            dash.graph.depends_on.pop(old_id, None),
        )
    for cell_id, (
        value,
        cell_meta,
        calculated_by,
        feeds_in,
        depends_on,
    ) in key_new_to_value.items():
        changes.append(dash.set_item(cell_id, [[value]]))
        if cell_meta is not None:
            dash.cell_meta[cell_id] = cell_meta
        elif cell_id in dash.cell_meta:
            del dash.cell_meta[cell_id]
        if calculated_by is not None:
            dash.graph.calculated_by[cell_id] = calculated_by
        elif cell_id in dash.graph.calculated_by:
            del dash.graph.calculated_by[cell_id]
        if feeds_in is not None:
            dash.graph.feeds_into[cell_id] = feeds_in
        elif cell_id in dash.graph.feeds_into:
            del dash.graph.feeds_into[cell_id]
        if depends_on is not None:
            dash.graph.depends_on[cell_id] = depends_on
        elif cell_id in dash.graph.depends_on:
            del dash.graph.depends_on[cell_id]

    # Clear only value, not metadata
    if to_clear_values:
        dash.clear_cells_internal(to_clear_values)
        changes.extend(to_clear_values)

    # Full clear, value and metadata
    if to_clear:
        dash.clear_cells_internal(to_clear)
        changes.extend(to_clear)
        for cell_id in to_clear:
            dash.cell_meta.pop(cell_id, None)
            dash.graph.calculated_by.pop(cell_id, None)
            dash.graph.feeds_into.pop(cell_id, None)
            dash.graph.depends_on.pop(cell_id, None)

    return changes


def assert_transform_no_merged_cell_overlap(
    dash: "Dash", transformation: Transformation
) -> None:
    if not transformation.boundary:
        return

    for address, cell_meta in dash.cell_meta.items():
        # No way to determine if a cell is part of a merged range, so need a full scan for all merge origins.
        if address.sheet != transformation.sheet_id:
            continue

        row_span = cell_meta.attributes.get(CellAttribute.ROW_SPAN.value)
        col_span = cell_meta.attributes.get(CellAttribute.COL_SPAN.value)
        if row_span and col_span:
            merge_span = Range(
                address.column,
                address.column + col_span - 1,
                address.row,
                address.row + row_span - 1,
                address.sheet,
            )
            if (
                merge_span.intersects(transformation.boundary)
                and merge_span not in transformation.boundary
            ):
                raise ValueError(f"Transformation crosses a merged cell: {merge_span}")


def recompile_cells(
    dash: "Dash",
    transformation: Transformation,
    cells_to_recompile: list[tuple[Address, CellMetadata]],
    cells_to_execute: set[Address],
    changed_cells: set[Address],
) -> None:
    """Recompile cells that are affected by a transformation. Append to cells_to_execute and changed_cells"""
    for cell_id, cell in cells_to_recompile:
        tokens = tokenize_with_ranges(dash.get_raw_code(cell_id))
        transformed_raw = process_sheet_transformation(
            tokens,
            transformation,
            dash.sheets[transformation.sheet_id].name,
            transformation.sheet_id != cell_id.sheet,
        )

        if transformed_raw or transform_crosses_infinite_range(tokens, transformation):
            if transformed_raw and (
                not transformation.boundary or cell_id in transformation.boundary
            ):
                dash.set_raw_code(cell_id, transformed_raw)
            dash.compile_and_update_cell_meta(cell_id)

            if cell.is_input_widget():
                changed_cells.add(cell_id)
            else:
                cells_to_execute.add(cell_id)


def add_delete_cells_helper(
    dash: "Dash",
    transformation: Transformation,
    cells_to_populate: list[dict] | None = None,
    send_undo: bool = False,
) -> None:
    assert_transform_no_merged_cell_overlap(dash, transformation)

    changed_cells: set[Address] = set()

    sheet_id = transformation.sheet_id

    # Compute and send inverse transform before any modifications
    undo_msg = _compute_undo_message(dash, transformation, send_undo=send_undo)

    # Key updates for cell_meta + cells (shift things like cells with attributes but no values)
    (
        to_delete_combined,
        to_unlink_combined,
        key_old_to_new_combined,
    ) = transformation.compute_add_delete_row_col_kv_update_dict(
        set(dash.cells[sheet_id].keys()).union(
            set([key for key in dash.cell_meta.keys() if key.sheet == sheet_id])
        ),
        transformation.boundary,
    )

    # Key updates for only cells with values.
    (
        to_clear_values,
        to_unlink,
        key_old_to_new,
    ) = transformation.compute_add_delete_row_col_kv_update_dict(
        list(dash.cells[sheet_id].keys()),
        transformation.boundary,
    )

    seed_attributes = {}
    if transformation.operation == SheetTransform.INSERT_BEFORE:
        seed_attributes = _compute_attribute_transfer_dict(dash, transformation)

    # Re-key dependency graph prior to changing cell keys
    spilled_to_clear = set()
    for cell_id, calculated_by_id in [*dash.graph.calculated_by.items()]:
        if cell_id in to_unlink:
            continue

        if calculated_by_id in to_unlink:
            dash.unlink(cell_id)
            spilled_to_clear.add(cell_id)
        elif calculated_by_id in key_old_to_new:
            dash.graph.calculated_by[cell_id] = key_old_to_new_combined[
                calculated_by_id
            ]

    for cell_id, feeds_into_ids in dash.graph.feeds_into.items():
        if cell_id in to_unlink:
            continue

        dash.graph.feeds_into[cell_id] = {
            key_old_to_new_combined.get(feeds_into_id, feeds_into_id)
            for feeds_into_id in feeds_into_ids
        }

    for cell_id, depends_on_ids in dash.graph.depends_on.items():
        if cell_id in to_unlink:
            continue

        dash.graph.depends_on[cell_id] = {
            key_old_to_new_combined.get(depends_on_id, depends_on_id)
            for depends_on_id in depends_on_ids
        }

    spilled_to_clear = {
        key_old_to_new_combined[cell_id] for cell_id in spilled_to_clear
    }

    # Move the values for all cells that have shifted
    combined_changes = _update_keys_combined(
        dash,
        key_old_to_new_combined,
        to_delete_combined.union(spilled_to_clear),
        to_clear_values,
    )
    changed_cells.update(combined_changes)

    # Cells to populate is only set when this is from an undo
    cells_to_execute = set()
    if cells_to_populate:
        for to_populate in cells_to_populate:
            cell_id = Address(*to_populate["cell_id"])
            changed_cells.add(cell_id)
            cell_meta = CellMetadata.from_dict(to_populate)
            dash.cell_meta[cell_id] = cell_meta

            if calculated_by := to_populate.get("calculated_by"):
                source_cell_id = Address.from_coord(calculated_by)
                dash.graph.calculated_by[cell_id] = source_cell_id
                dash.graph.feeds_into.setdefault(source_cell_id, set()).add(cell_id)
            else:
                cells_to_execute.add(cell_id)
                continue
            outputs = to_populate["outputs"]
            if outputs:
                output_data = [output["data"] for output in outputs]
                dash.set_item(cell_id, outputs_to_value(output_data))

    # Any cell that was calculated by a cell that now gets deleted needs to be cleared in the
    # kernel - after adjusting the coordinates though:
    calculated_by_to_clear: list[Address] = []
    for cell_id in to_unlink:
        for inner_cell_id in dash.graph.feeds_into.get(cell_id, ()):
            if dash.graph.calculated_by.get(inner_cell_id):
                transform_result = transformation.transform(
                    inner_cell_id.column, inner_cell_id.row
                )
                if isinstance(transform_result, tuple):
                    calculated_by_to_clear.append(Address(*transform_result, sheet_id))

    _maybe_update_grid_size(dash, transformation)

    # Recompile and execute cells whose formula changed.
    cells_to_recompile = [
        (cell_id, cell)
        for cell_id, cell in dash.cell_meta.items()
        if cell_id not in to_clear_values
        and cell_id not in cells_to_execute
        and dash.has_formula(cell_id)
    ]
    recompile_cells(
        dash, transformation, cells_to_recompile, cells_to_execute, changed_cells
    )

    # Execute code formula changes.
    run_cell_changes = []
    for cell_id in cells_to_execute:
        cell_meta = dash.get_or_create_cell_meta(cell_id)
        run_cell_changes.append(
            CellChange(
                attributes=cell_meta.attributes,
                cell_id=cell_id.to_float_coord(),
                content=dash.get_raw_code(cell_id),
                mime_type=cell_meta.mime_type,
            ).to_dict()  # TODO: Optimize for the fact that run_cells will just unpack this dict.
        )
    dash.run_cells_with_cascade(
        cell_changes=run_cell_changes,
        pre_clear=calculated_by_to_clear,
    )

    if transformation.operation == SheetTransform.INSERT_BEFORE:
        _transfer_cell_attributes_to_inserted_cells(
            dash, transformation, seed_attributes, changed_cells
        )

    sheet_attribute_updates = _compute_sheet_attribute_updates(dash, transformation)
    dash.update_sheet_attributes_internal(sheet_attribute_updates)
    client_sheet_attribute_updates = {
        attribute: value for (sheet_id, attribute, value) in sheet_attribute_updates
    }

    def cell_is_in_bounds(addr: Address) -> bool:
        sheet = dash.sheets[addr.sheet]
        return addr.row < sheet.n_rows and addr.column < sheet.n_cols

    changed_cells = {addr for addr in changed_cells if cell_is_in_bounds(addr)}
    dash.notify_client_cells_have_changed(
        changed_cells,
        undo=undo_msg,
    )

    dash.send_sheet_attribute_and_grid_size_update(
        sheet_id, client_sheet_attribute_updates, transformation
    )
