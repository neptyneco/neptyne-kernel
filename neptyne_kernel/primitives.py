from dataclasses import fields
from datetime import datetime
from functools import total_ordering as _total_ordering
from typing import Any, Iterator

from .cell_api import ApiRef, CellApiMixin
from .widgets.base_widget import BaseWidget


class NeptynePrimitive(CellApiMixin):
    @property
    def value(self) -> "NeptynePrimitive":
        return self

    @value.setter
    def value(self, new_value: "NeptynePrimitive") -> None:
        if self.ref is None:
            raise ValueError("Cannot set value of detached item")
        self.ref.setitem((0, 0), new_value)

    def __copy__(self) -> "NeptynePrimitive":
        return self

    def __deepcopy__(self, memo: Any) -> "NeptynePrimitive":
        return self

    def __setattr__(self, key: str, value: Any) -> None:
        if key not in ("ref", "value"):
            raise AttributeError(f"Cannot set attribute {key}")
        super().__setattr__(key, value)

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        plain_type = self.__class__.__bases__[0]
        v = plain_type(self)
        return plain_type, (v,)


class NeptyneInt(int, NeptynePrimitive):
    def __new__(cls, value: int, ref: ApiRef | None = None) -> "NeptyneInt":
        x = int.__new__(cls, value)
        x.ref = ref
        return x

    def __and__(self, other: Any) -> str:  # type: ignore
        return str(self) + str(other)

    def __rand__(self, other: Any) -> str:  # type: ignore
        return str(other) + str(self)

    def to_datetime(self) -> datetime:
        from .spreadsheet_datetime import excel2datetime

        return excel2datetime(self)


class NeptyneStr(str, NeptynePrimitive):
    def __new__(cls, value: str, ref: ApiRef | None = None) -> "NeptyneStr":
        x = str.__new__(cls, value)
        x.ref = ref
        return x

    def __and__(self, other: Any) -> str:
        return str(self) + str(other)

    def __rand__(self, other: Any) -> str:
        return str(other) + str(self)


class NeptyneFloat(float, NeptynePrimitive):
    def __new__(cls, value: float, ref: ApiRef | None = None) -> "NeptyneFloat":
        x = float.__new__(cls, value)
        x.ref = ref
        return x

    def __and__(self, other: Any) -> str:
        return str(self) + str(other)

    def __rand__(self, other: Any) -> str:
        return str(other) + str(self)

    def to_datetime(self) -> datetime:
        from .spreadsheet_datetime import excel2datetime

        return excel2datetime(self)


def _empty_div(other: Any) -> Any:
    if other == 0:
        raise ZeroDivisionError("division by zero")
    return 0


@_total_ordering
class Empty(NeptynePrimitive):
    ref: ApiRef | None

    # Shared object to make something empty
    MakeItSo = object()

    def __init__(self, ref: ApiRef | None = None):
        self.ref = ref

    def is_empty(self) -> bool:
        return True

    def __add__(self, other: Any) -> Any:
        return other

    def __radd__(self, other: Any) -> Any:
        return other

    def __sub__(self, other: Any) -> Any:
        return -other

    def __rsub__(self, other: Any) -> Any:
        return other

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, str):
            return ""
        return 0

    def __rmul__(self, other: Any) -> Any:
        return self.__mul__(other)

    def __floordiv__(self, other: Any) -> Any:
        return _empty_div(other)

    def __truediv__(self, other: Any) -> Any:
        return _empty_div(other)

    def __rtruediv__(self, other: Any) -> Any:
        raise ZeroDivisionError("Cannot divide by empty")

    def __rfloordiv__(self, other: Any) -> Any:
        raise ZeroDivisionError("Cannot divide by empty")

    def __gt__(self, other: Any) -> bool:
        return other > 0

    def __eq__(self, other: Any) -> bool:
        return other == 0

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "None"

    def __and__(self, other: Any) -> str:
        return str(other)

    def __rand__(self, other: Any) -> str:
        return str(other)

    def __iter__(self) -> Iterator:
        return iter([])

    def __len__(self) -> int:
        return 0

    def __reduce__(self) -> tuple[type, tuple[()]]:
        return type(None), ()

    def __getitem__(self, item: Any) -> Any:
        if ref := self.ref:
            msg = f"{ref.range.origin().to_a1()} is empty and can't be indexed"
        else:
            msg = "Empty object can't be indexed"
        raise ValueError(msg)


def proxy_methods(cls: type) -> type:
    def proxy_method(method_name: str) -> None:
        def method(self: Any, *args: Any, **kwargs: Any) -> Any:
            return getattr(self.obj, method_name)(*args, **kwargs)

        setattr(cls, method_name, method)

    for method_name in [
        "__add__",
        "__sub__",
        "__mul__",
        "__matmul__",
        "__truediv__",
        "__floordiv__",
        "__mod__",
        "__divmod__",
        "__pow__",
        "__lshift__",
        "__rshift__",
        "__and__",
        "__xor__",
        "__or__",
        "__radd__",
        "__rsub__",
        "__rmul__",
        "__rmatmul__",
        "__rtruediv__",
        "__rfloordiv__",
        "__rmod__",
        "__rdivmod__",
        "__rpow__",
        "__rlshift__",
        "__rrshift__",
        "__rand__",
        "__rxor__",
        "__ror__",
        "__iadd__",
        "__isub__",
        "__imul__",
        "__imatmul__",
        "__itruediv__",
        "__ifloordiv__",
        "__imod__",
        "__ipow__",
        "__ilshift__",
        "__irshift__",
        "__iand__",
        "__ixor__",
        "__ior__",
        "__neg__",
        "__pos__",
        "__abs__",
        "__invert__",
        "__complex__",
        "__int__",
        "__float__",
        "__index__",
        "__round__",
        "__trunc__",
        "__floor__",
        "__ceil__",
        "__len__",
        "__length_hint__",
        "__iter__",
        "__next__",
        "__lt__",
        "__le__",
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
        "__repr__",
        "__str__",
        "__getitem__",
        "__setitem__",
        "__delitem__",
        "__missing__",
        "__contains__",
        "__call__",
    ]:
        proxy_method(method_name)

    return cls


@proxy_methods
class ProxiedObject(CellApiMixin):
    obj: Any
    ref: ApiRef | None
    recalc_fields: set[str]

    def __init__(
        self, obj: Any, ref: ApiRef | None, recalc_fields: set[str] | None = None
    ):
        # Avoid infinite recursion due to set_attr:
        if recalc_fields is None:
            recalc_fields = set()
        self.__dict__.update(dict(obj=obj, ref=ref, recalc_fields=recalc_fields))

    @property  # type: ignore
    def __class__(self) -> type:
        return self.obj.__class__

    def __getattr__(self, item: str) -> Any:
        return getattr(self.obj, item)

    def __setattr__(self, key: str, value: Any) -> None:
        if not hasattr(self.obj, key):
            raise AttributeError(f"{key} is not a valid attribute")

        setattr(self.obj, key, value)
        if key in self.recalc_fields:
            from .dash_ref import DashRef

            assert isinstance(self.ref, DashRef)
            self.ref.dash.update_cell_meta_on_value_change(
                self.ref.range.origin(), self.obj
            )
            self.ref.dash.notify_client_cells_have_changed(self.ref.range.origin())


def proxy_val(val: Any, ref: ApiRef) -> CellApiMixin:
    if val is None:
        return Empty(ref)
    if isinstance(val, int):
        return NeptyneInt(val, ref)
    if isinstance(val, float):
        return NeptyneFloat(val, ref)
    if isinstance(val, str):
        return NeptyneStr(val, ref)
    recalc_fields = None
    if isinstance(val, BaseWidget):
        # Widgets are currently kinda doubly proxied since they are CellApi objects. We should be
        # able to do away with that at some point:
        recalc_fields = {"value", *[f.name for f in fields(val)]}
        setattr(val, "ref", ref)
    return ProxiedObject(val, ref, recalc_fields)


def unproxy_val(val: Any) -> Any:
    if isinstance(val, Empty):
        return None
    if isinstance(val, NeptyneInt):
        return int(val)
    if isinstance(val, NeptyneFloat):
        return float(val)
    if isinstance(val, NeptyneStr):
        return str(val)
    if isinstance(val, ProxiedObject):
        return val.obj
    return val


def check_none(arg: Any) -> bool:
    return arg is None or isinstance(arg, Empty)
