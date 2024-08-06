from dataclasses import dataclass


@dataclass
class Condition:
    op: str
    value: float

    def evaluate(self, lhs: float) -> bool:
        match self.op:
            case "<":
                return lhs < self.value
            case "<=":
                return lhs <= self.value
            case ">":
                return lhs > self.value
            case ">=":
                return lhs >= self.value
            case "<>":
                return lhs != self.value
            case "=":
                return lhs == self.value
        return False
