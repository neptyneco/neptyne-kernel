from .color import Color


def test_repr():
    c = Color(255, 0, 0)
    assert repr(c) == "Color(r=255, g=0, b=0)"


def test_webcolor():
    c = Color(255, 0, 0)
    assert c.webcolor == "#FF0000"


def test_to_dict():
    c = Color(255, 0, 0)
    assert c.to_dict() == {"r": 255, "g": 0, "b": 0}


def test_from_dict():
    c = Color.from_dict({"r": 0, "g": 255, "b": 0})
    assert c == Color(0, 255, 0)
    assert c.webcolor == "#00FF00"


def test_from_webcolor():
    c = Color.from_webcolor("#0000ff")
    assert c == Color(0, 0, 255)
    assert c.webcolor == "#0000FF"


def test_constructor_with_tuple():
    c = Color((128, 128, 128))
    assert c == Color(128, 128, 128)
    assert c.webcolor == "#808080"
