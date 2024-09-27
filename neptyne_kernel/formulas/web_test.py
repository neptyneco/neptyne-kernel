import pytest

from ..cell_range import CellRange
from ..spreadsheet_error import VALUE_ERROR
from .helpers import assert_equal
from .web import ENCODEURL, FILTERXML, WEBSERVICE


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://api.mathjs.org/v4/?expr=2*(7-3)", "8"),
        ("111", VALUE_ERROR),
        ("https://google.com/123", VALUE_ERROR),
    ],
)
def test_WEBSERVICE(url, expected):
    fetched = WEBSERVICE(url)
    if isinstance(fetched, str):
        assert expected in expected
    else:
        assert expected == fetched


@pytest.mark.parametrize(
    "text, result",
    [
        (
            "автомобиль",
            "%D0%B0%D0%B2%D1%82%D0%BE%D0%BC%D0%BE%D0%B1%D0%B8%D0%BB%D1%8C",
        ),
        (
            "http://contoso.sharepoint.com/Finance/Profit and Loss Statement.xlsx",
            "http%3A%2F%2Fcontoso.sharepoint.com%2FFinance%2FProfit%20and%20Loss%20Statement.xlsx",
        ),
    ],
)
def test_ENCODEURL(text, result):
    assert ENCODEURL(text) == result


FILTERXML_SOURCE = """<members>
<member>
<name>John Doe</name>
<email>sbriacc2@nasigoreng.buzz</email>
</member>
<member>
<name>Jane Doe</name>
<email>7jobelso.57c@bacharg.com</email>
</member>
<member>
<name>Ann Other</name>
<email>2kingmuslehh@elhida.com</email>
</member>
</members>"""


@pytest.mark.parametrize(
    "xml, xpath, result",
    [
        (
            FILTERXML_SOURCE,
            "/members/member/name",
            CellRange(["John Doe", "Jane Doe", "Ann Other"]),
        ),
        (FILTERXML_SOURCE, "//member[2]/name", CellRange(["Jane Doe"])),
        (
            FILTERXML_SOURCE,
            "/members/member[last()-1]/email",
            CellRange(["7jobelso.57c@bacharg.com"]),
        ),
        (
            FILTERXML_SOURCE,
            "//member[position()>2]",
            CellRange(["""\nAnn Other\n2kingmuslehh@elhida.com\n"""]),
        ),
        (FILTERXML_SOURCE + "xx", "/members/member/name", VALUE_ERROR),
        (FILTERXML_SOURCE, "/hello/world", None),
        (FILTERXML_SOURCE, "++/members/member/name", VALUE_ERROR),
    ],
)
def test_FILTERXML(xml, xpath, result):
    assert_equal(FILTERXML(xml, xpath), result)
