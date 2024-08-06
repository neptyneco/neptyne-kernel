from unittest import mock

import pytest

from .dash import Dash


@pytest.fixture
def dash():
    with (
        mock.patch("neptyne_kernel.dash.get_ipython_mockable"),
        mock.patch("neptyne_kernel.kernel_runtime.get_ipython_mockable"),
    ):
        Dash._instance = None
        dash = Dash(silent=True)
        dash.sheets._register_sheet(0, "Sheet0")
        return dash
