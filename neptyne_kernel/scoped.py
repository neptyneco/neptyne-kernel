from typing import Any

from .get_ipython_mockable import get_ipython_mockable

_SCOPED_OBJECTS: list[Any] = []


def close_scoped_objects() -> None:
    for scoped_obj in _SCOPED_OBJECTS:
        scoped_obj.close()


def close_post_run_cell(obj: Any) -> None:
    ip = get_ipython_mockable()
    if close_scoped_objects not in ip.events.callbacks:
        ip.events.register("post_run_cell", close_scoped_objects)
    _SCOPED_OBJECTS.append(obj)
