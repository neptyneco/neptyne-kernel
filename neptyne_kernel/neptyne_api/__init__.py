# ruff: noqa: F401, F403
"""
.. include:: ./README.md
"""

import sys

from ..dash import Dash
from ..gsheets_api import Formula
from . import ai, cell_range, email, formatting, google, sheet
from .caching import cache
from .cron import daily, weekly
from .deprecated import send_owner_email
from .events import do_events, get_mutex
from .hooks import api_function, on_range_change, on_value_change
from .images import qr_for_url, url_for_image
from .local_kernels import connect_kernel
from .secrets import get_secret, get_secrets
from .streamlit_decorator import streamlit
from .ui import alert, confetti, navigate_to
from .upload_download import download, upload, upload_csv_to_new_sheet
from .user import get_user
from .vectorize import vectorize_cells

sheets = Dash.instance().sheets
"""The global sheets collection. Access with `nt.sheets`

See [Sheet API](/neptyne_kernel/neptyne_api#sheets) for more details."""
named_ranges = Dash.instance().named_ranges


COMMON = [
    "sheets",
    "ai",
    "cell",
    "cell_range",
    "data",
    "email",
    "google",
    "daily",
    "weekly",
    "send_owner_email",
    "qr_for_url",
    "sheet",
    "get_secret",
    "get_secrets",
    "upload",
    "download",
    "vectorize_cells",
    "url_for_image",
    "api_function",
    "connect_kernel",
]

try:
    from . import data

    COMMON += ["data"]
except ImportError:
    pass

try:
    from . import geo

    COMMON += ["geo"]
except ImportError:
    pass

CORE = [
    "datetime_to_serial",
    "do_events",
    "on_value_change",
    "on_range_change",
    "alert",
    "confetti",
    "navigate_to",
    "get_user",
    "upload_csv_to_new_sheet",
]

GSHEETS = [
    "Formula",
    "cache",
    "formatting",
    "streamlit",
    "named_ranges",
]


if Dash.instance().in_gs_mode:
    __all__ = [*COMMON, *CORE]
else:
    __all__ = [*COMMON, *GSHEETS]

sys.modules["neptyne"] = sys.modules[__name__]
