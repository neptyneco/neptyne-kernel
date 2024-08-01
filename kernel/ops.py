from dataclasses import dataclass

from .cell_address import Address


@dataclass
class ExecOp:
    address: Address
    expression: str


@dataclass
class ClearOp:
    to_clear: list[Address]
