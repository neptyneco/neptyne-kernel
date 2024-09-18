from collections import namedtuple

try:
    import matplotlib.colors as mcolors
except ImportError:
    mcolors = None


class Color(namedtuple("ColorBase", ["r", "g", "b"])):
    __slots__ = ()

    def __new__(cls, *args: list | tuple | str | int) -> "Color":
        if len(args) == 3 and all(isinstance(val, int) for val in args):
            red, green, blue = args
        elif len(args) == 1 and isinstance(args[0], list | tuple) and len(args[0]) == 3:
            red, green, blue = args[0]
        elif (
            len(args) == 1
            and isinstance(args[0], str)
            and mcolors is not None
            and (rgb := mcolors.CSS4_COLORS.get(args[0], args[0])).startswith("#")
        ):
            red, green, blue = Color.from_webcolor(rgb)
        else:
            raise ValueError(
                "Invalid color input. Please provide either three integers, a list or tuple of "
                "RGB values, a hex color string or a named color from matplotlib. You provided: "
                + str(args)
            )

        return super().__new__(cls, red, green, blue)

    def __repr__(self) -> str:
        return f"Color(r={self.r}, g={self.g}, b={self.b})"

    @property
    def webcolor(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}".upper()

    def to_dict(self) -> dict[str, int]:
        return {"r": self.r, "g": self.g, "b": self.b}

    @classmethod
    def from_dict(cls, color_dict: dict[str, int]) -> "Color":
        return cls(color_dict["r"], color_dict["g"], color_dict["b"])

    @classmethod
    def from_webcolor(cls, webcolor: str) -> "Color":
        r = int(webcolor[1:3], 16)
        g = int(webcolor[3:5], 16)
        b = int(webcolor[5:7], 16)
        return cls(r, g, b)
