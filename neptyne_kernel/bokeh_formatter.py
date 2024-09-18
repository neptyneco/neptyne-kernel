from typing import Any


def maybe_format_bokeh(value: Any) -> str | None:
    try:
        from bokeh.embed import file_html
        from bokeh.plotting import figure
        from bokeh.resources import CDN
    except ImportError:
        return None

    if not isinstance(value, figure):
        return None
    try:
        # Don't show the toolbar - revisit when we can support it in the iframe
        toolbar_location = value.toolbar_location
        value.toolbar_location = None  # type: ignore
        return file_html(value, CDN, "Neptyne")
    finally:
        value.toolbar_location = toolbar_location
