from dataclasses import dataclass

from .condition import Condition
from .decimal_section import DecimalSection
from .exponential_section import ExponentialSection
from .fraction_section import FractionSection
from .section_type import SectionType


@dataclass
class Section:
    index: int
    section_type: SectionType
    color: str
    condition: Condition
    exponential: ExponentialSection
    fraction: FractionSection
    number: DecimalSection
    general_text_date_duration_parts: list[str]
