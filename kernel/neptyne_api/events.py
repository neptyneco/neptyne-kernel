import threading

from ..dash import Dash


async def do_events(time_to_sleep: float = 0) -> None:
    await Dash.instance().do_events(time_to_sleep)


async def get_mutex(key: str = "") -> threading.Lock:
    """Get a mutex by key. If the mutex does not exist, it will be created.

    Example usage:

    ```python
    import neptyne as nt
    with nt.get_mutex("my_key"):
        A1 += 1
    ```
    """
    return Dash.instance()._mutex_manager.get_mutex(key)
