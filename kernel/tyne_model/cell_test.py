import pytest

from .cell import CellMetadata, output_from_dict, represents_simple_value


@pytest.mark.parametrize(
    "as_dict, expected",
    [
        (
            {
                "data": None,
                "ename": "#DIV/0!",
                "evalue": None,
                "execution_count": -1,
                "metadata": None,
                "name": None,
                "output_type": "error",
                "text": None,
                "traceback": ["Function DIVIDE parameter 2 cannot be zero."],
            },
            False,
        ),
        ({"data": {"application/json": None}, "output_type": "execute_result"}, True),
    ],
)
def test_represents_simple_value_for_outputs(as_dict: dict, expected):
    output = output_from_dict(as_dict)
    assert represents_simple_value(output) == expected


@pytest.mark.parametrize(
    "meta, expected",
    [
        (
            CellMetadata(
                attributes={"numberFormat": None},
                raw_code="=1/0",
                compiled_code="1 / 0",
                mime_type=None,
                execution_policy=-1,
                next_execution_time=1684774811.9209,
                output=output_from_dict(
                    {
                        "data": None,
                        "ename": "#DIV/0!",
                        "evalue": None,
                        "execution_count": -1,
                        "metadata": None,
                        "name": None,
                        "output_type": "error",
                        "text": None,
                        "traceback": ["Function DIVIDE parameter 2 cannot be zero."],
                    }
                ),
            ),
            False,
        ),
        (
            CellMetadata(
                attributes={},
                raw_code="2",
                compiled_code="2",
                mime_type=None,
                execution_policy=-1,
                next_execution_time=1684774811.920949,
                output=None,
            ),
            True,
        ),
    ],
)
def test_represents_simple_value_for_cell_metadata(meta: CellMetadata, expected):
    assert represents_simple_value(meta) == expected
