from dataclasses import dataclass


@dataclass
class Download:
    name: str
    value: bytes | bytearray | str
    mimetype: str

    def __repr__(self) -> str:
        return f"Download({self.name!r}, {self.mimetype!r})"
