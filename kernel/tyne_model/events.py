from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..neptyne_protocol import Severity
from ..neptyne_protocol import TyneEvent as EventProtocol


@dataclass
class Event:
    message: str
    severity: Severity
    extra: dict[str, Any] = field(default_factory=dict)

    created: datetime = field(default_factory=datetime.utcnow)

    def export(self) -> dict:
        return EventProtocol(
            message=self.message,
            severity=self.severity,
            extra=self.extra,
            date=self.created.isoformat(),
        ).to_dict()
