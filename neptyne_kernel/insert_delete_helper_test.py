from .cell_address import Address
from .dash_test import a1
from .insert_delete_helper import _update_keys_combined


def test_update_keys(dash):
    dash[Address(0, 0, 0)] = 1
    dash[Address(0, 1, 0)] = 2
    dash[Address(0, 2, 0)] = 3
    dash[Address(1, 6, 0)] = 4

    key_update_map = {a1("A1"): a1("A2"), a1("A2"): a1("A3"), a1("A3"): a1("A4")}

    _update_keys_combined(dash, key_update_map, [], [])

    assert dash[Address(0, 0, 0)].is_empty()
    assert dash[Address(0, 1, 0)] == 1
    assert dash[Address(0, 2, 0)] == 2
    assert dash[Address(0, 3, 0)] == 3
    assert dash[Address(1, 6, 0)] == 4
