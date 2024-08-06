from unittest import mock

from ..cell_address import Address
from ..dash_ref import DashRef
from .input_widgets import Button, Dropdown


@mock.patch("neptyne_kernel.kernel_runtime.get_ipython_mockable", mock.MagicMock())
@mock.patch("neptyne_kernel.dash.get_ipython_mockable", mock.MagicMock())
def test_button(dash):
    def trigger_action_using_button(action):
        button = Button(action.__name__, action=action)
        button.ref = DashRef(dash, Address.from_a1("A1"))
        button.trigger(None, button._get_event(), True)

    def my_action():
        my_action.clicked = True

    trigger_action_using_button(my_action)
    assert my_action.clicked

    def action_with_event(event):
        action_with_event.event = event

    trigger_action_using_button(action_with_event)
    assert action_with_event.event.cell.xy == (0, 0)


@mock.patch("neptyne_kernel.kernel_runtime.get_ipython_mockable", mock.MagicMock())
@mock.patch("neptyne_kernel.dash.get_ipython_mockable", mock.MagicMock())
def test_dropdown(dash):
    def action():
        action.clicked = True

    dropdown = Dropdown(choices=["a", "b", "c"], action=action)
    address = Address.from_a1("A1")
    dropdown.ref = DashRef(dash, address)
    dash[address] = dropdown
    dropdown.value = "b"

    assert action.clicked
