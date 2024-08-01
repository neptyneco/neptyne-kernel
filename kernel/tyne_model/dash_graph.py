from typing import Any

from ..cell_address import Address


class DashGraph:
    def __init__(self) -> None:
        self.feeds_into: dict[Address, set[Address]] = {}
        self.depends_on: dict[Address, set[Address]] = {}
        self.calculated_by: dict[Address, Address] = {}

    def check_integrity(self) -> None:
        depends_on = flatten_edges(self.depends_on)
        feeds_into = flatten_edges(self.feeds_into)
        calculated_by = set(self.calculated_by.items())

        for left, right in depends_on:
            if (right, left) not in feeds_into:
                raise ValueError(
                    f"{left.to_a1()} depends on {right.to_a1()} but does not feed into it"
                )

        for left, right in calculated_by:
            if (right, left) not in feeds_into:
                raise ValueError(
                    f"{left.to_a1()} is calculated by {right.to_a1()} but does not feed into it"
                )

        for left, right in feeds_into:
            if (right, left) not in depends_on and (right, left) not in calculated_by:
                raise ValueError(
                    f"{left.to_a1()} feeds into {right.to_a1()} but is not "
                    f"depended on or calculated by it"
                )

    def to_dict(self) -> dict:
        return {
            "depends_on": [
                (key.to_coord(), [value.to_coord() for value in values])
                for key, values in self.depends_on.items()
            ],
            "calculated_by": [
                (key.to_coord(), value.to_coord())
                for key, value in self.calculated_by.items()
            ],
            # "feeds_into" is omitted and reconstructed in from_dict
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DashGraph":
        res = cls()
        for key_s, values in data["depends_on"]:
            key = Address.from_coord(key_s)
            depends_on = res.depends_on[key] = set()
            for value_s in values:
                value = Address.from_coord(value_s)
                depends_on.add(value)
                res.feeds_into.setdefault(value, set()).add(key)
        for key_s, value_s in data["calculated_by"]:
            key = Address.from_coord(key_s)
            value = Address.from_coord(value_s)
            res.calculated_by[key] = value
            res.feeds_into.setdefault(value, set()).add(key)

        return res


def flatten_edges(edges: dict[Address, set[Address]]) -> set[tuple[Address, Address]]:
    return {(left, right) for left, rights in edges.items() for right in rights}
