import os
from functools import wraps
from typing import Any, Callable

from ..dash import Dash
from ..neptyne_protocol import StreamlitAppConfig


def streamlit(
    fn: Callable | None = None,
    *,
    width: int = 641,
    height: int = 480,
    window_caption: str = "",
    sidebar: bool = False,
    auto_open: bool = False,
    public: bool = False,
) -> Callable:
    """Register a function to be called as a Streamlit app

    width, height: the size of the popup window
    window_caption: the title of the popup window (not streamlit title)
    sidebar: whether to show the streamlit sidebar - if so, width, heigth and window_caption are ignored
    auto_open: whether to open automatically when the sheet loads
    public: whether the app is accessible outside of the sheet

    Use as a decorator, for example:
    ```python
    @streamlit(width=800, height=600)
    def my_function():
        st.write("Hello World!")
    ```

    For more information on Neptyne's streamlit integration, see the [Streamlit API](/neptyne_kernel/neptyne_api#streamlit) documentation.
    """
    if fn is None:
        return lambda f: streamlit(
            f,
            window_caption=window_caption,
            width=width,
            height=height,
            sidebar=sidebar,
            auto_open=auto_open,
            public=public,
        )

    if bool(os.getenv("NEPTYNE_LOCAL_NKS_KERNEL")):
        raise ValueError("Streamlit apps are not yet supported in local kernels")

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    Dash.instance().register_streamlit(
        fn,
        StreamlitAppConfig(
            window_caption=window_caption,
            width=width,
            height=height,
            sidebar=sidebar,
            auto_open=auto_open,
            public=public,
        ),
    )

    return wrapper
