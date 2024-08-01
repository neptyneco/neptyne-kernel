from dataclasses import dataclass
from itertools import chain
from typing import ClassVar, Iterable, Literal

from .cell_address import Address, Range
from .neptyne_protocol import (
    Dimension,
    InsertDeleteContent,
    SelectionRect,
    SheetTransform,
)

Result = Literal["REF_ERROR", "NO_CHANGE"] | tuple[int, int]


@dataclass
class Transformation:
    dimension: Dimension
    operation: SheetTransform
    index: int
    amount: int
    sheet_id: int
    boundary: Range | None = None

    REF_ERROR: ClassVar[Result] = "REF_ERROR"
    NO_CHANGE: ClassVar[Result] = "NO_CHANGE"

    @classmethod
    def from_dict(cls, d: dict) -> "Transformation":
        return Transformation(
            dimension=Dimension(d["dimension"]),
            operation=SheetTransform(d["operation"]),
            index=d["index"],
            amount=d["amount"],
            sheet_id=d["sheet_id"],
            boundary=Range.from_dict(d["boundary"]) if d["boundary"] else None,
        )

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension.value,
            "operation": self.operation.value,
            "index": self.index,
            "amount": self.amount,
            "sheet_id": self.sheet_id,
            "boundary": self.boundary.to_dict() if self.boundary else None,
        }

    def transform(self, x: int, y: int) -> Result:
        if self.dimension == Dimension.ROW:
            res = self.do_transform(y, x)
            if isinstance(res, tuple):
                return res[1], res[0]
            return res
        return self.do_transform(x, y)

    def do_transform(self, x: int, y: int) -> Result:
        stop = self.index + self.amount - 1
        if self.operation == SheetTransform.DELETE:
            if self.index <= x <= stop:
                return Transformation.REF_ERROR
            elif x > stop:
                return x - self.amount, y
            else:
                return Transformation.NO_CHANGE
        else:
            if x >= self.index:
                return x + self.amount, y
            return Transformation.NO_CHANGE

    def compute_add_delete_row_col_kv_update_dict(
        self,
        addresses: Iterable[Address],
        boundary: Range | None = None,
    ) -> tuple[set[Address], set[Address], dict[Address, Address]]:
        """Returns 3 values:
        1. A set of cells to delete. These are from the edge of the transform after shifting and are not being replaced
        2. A set of cells that are actually deleted by the transform. DELETE transform or a bounded shift can cause this
        3. A dict of cell_ids to update. These are cells that are not being deleted, but are being shifted by the transform."""
        transform_deleted_keys = set()
        key_update_dict = {}
        values = set()
        for address in addresses:
            transform_result = self.transform(address.column, address.row)

            if transform_result is Transformation.REF_ERROR:
                transform_deleted_keys.add(address)
            elif isinstance(transform_result, tuple):
                new_address = Address(*transform_result, address.sheet)
                key_update_dict[address] = new_address
                values.add(new_address)

        # to_delete_keys represents cells cleared after all index shifting is complete.
        to_delete_keys = set(
            x
            for x in chain(key_update_dict.keys(), transform_deleted_keys)
            if x not in values
        )

        if boundary:
            to_delete_keys = set(x for x in to_delete_keys if x in boundary)
            transform_deleted_keys = set(
                x for x in transform_deleted_keys if x in boundary
            )

            key_update_dict_new = {}
            for key, value in key_update_dict.items():
                if key not in boundary:
                    continue
                if value not in boundary:
                    transform_deleted_keys.add(key)
                    continue
                key_update_dict_new[key] = value
            key_update_dict = key_update_dict_new

        return to_delete_keys, transform_deleted_keys, key_update_dict


def insert_delete_content_to_sheet_transform(
    insert_delete_content: InsertDeleteContent,
) -> tuple[Transformation, list[dict]]:
    sheet_id = int(insert_delete_content.sheet_id or 0)

    transformation = Transformation(
        insert_delete_content.dimension,
        insert_delete_content.sheet_transform,
        int(insert_delete_content.selected_index),
        int(insert_delete_content.amount) if insert_delete_content.amount else 1,
        sheet_id,
    )

    if insert_delete_content.boundary:
        r = insert_delete_content.boundary
        transformation.boundary = Range(
            min_col=int(r.min_col),
            min_row=int(r.min_row),
            max_col=int(r.max_col),
            max_row=int(r.max_row),
            sheet=sheet_id,
        )

    non_empty_cells_to_populate = (
        insert_delete_content.cells_to_populate
        if insert_delete_content.cells_to_populate
        else []
    )
    return transformation, non_empty_cells_to_populate


def sheet_transform_to_insert_delete_content(
    transformation: Transformation,
    cells_to_populate: list[dict],
) -> InsertDeleteContent:
    return InsertDeleteContent(
        cells_to_populate=cells_to_populate,
        dimension=transformation.dimension,
        selected_index=transformation.index,
        sheet_transform=transformation.operation,
        sheet_id=transformation.sheet_id,
        amount=transformation.amount,
        boundary=SelectionRect.from_dict(transformation.boundary.to_dict())
        if transformation.boundary
        else None,
    )


def is_insert_delete_unbounded(dimension: Dimension, boundary: Range | None) -> bool:
    return (
        not boundary
        or (dimension == Dimension.COL and boundary.max_row == -1)
        or (dimension == Dimension.ROW and boundary.max_col == -1)
    )


def transformation_is_unbounded(transformation: Transformation) -> bool:
    return is_insert_delete_unbounded(transformation.dimension, transformation.boundary)
