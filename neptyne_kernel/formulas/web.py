from urllib.parse import quote, urlparse

from lxml import etree as ET
from requests import get
from requests.exceptions import RequestException

from ..cell_range import CellRange
from ..spreadsheet_error import VALUE_ERROR

__all__ = ["ENCODEURL", "FILTERXML", "WEBSERVICE"]


def ENCODEURL(text: str) -> str:
    """Returns a URL-encoded string"""
    return quote(text, safe="")


def FILTERXML(xml_source: str, xpath: str) -> CellRange | None:
    """Returns specific data from the XML content by using the specified XPath"""
    try:
        elements = ET.fromstring(xml_source).xpath(xpath)
    except (ET.ParserError, ET.XMLSyntaxError, ET.XPathEvalError):
        return VALUE_ERROR
    if not elements:
        return
    return CellRange(["".join(el.itertext()) for el in elements])


def WEBSERVICE(url: str) -> str:
    """Returns data from a web service"""
    if len(url) > 2048:
        return VALUE_ERROR
    p = urlparse(url)
    if p.scheme not in ["http", "https"]:
        return VALUE_ERROR
    try:
        result = get(url)
    except RequestException:
        return VALUE_ERROR
    return result.text if result.ok and len(result.text) < 32767 else VALUE_ERROR
