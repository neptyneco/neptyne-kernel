from typing import TYPE_CHECKING, Union

from ..cell_api import CellApiMixin
from ..dash_ref import DashRef
from ..kernel_runtime import get_kernel
from ..neptyne_protocol import MessageTypes


def alert(msg: str) -> None:
    k = get_kernel()
    k.send_response(
        k.iopub_socket,
        MessageTypes.SHOW_ALERT.value,
        {"msg": msg},
    )


def confetti(duration: float = 5) -> None:
    k = get_kernel()
    k.send_response(
        k.iopub_socket,
        MessageTypes.CONFETTI.value,
        {"duration": duration},
    )


def navigate_to(ref: Union[CellApiMixin, "DashRef"]) -> None:
    k = get_kernel()

    if isinstance(ref, CellApiMixin):
        assert ref.ref
        ref = ref.ref  # type: ignore

    if TYPE_CHECKING:
        assert isinstance(ref, DashRef)

    ref_range = ref.range
    k.send_response(
        k.iopub_socket,
        MessageTypes.NAVIGATE_TO.value,
        {
            "sheet": ref_range.sheet,
            "col": ref_range.min_col,
            "row": ref_range.min_row,
        },
    )
