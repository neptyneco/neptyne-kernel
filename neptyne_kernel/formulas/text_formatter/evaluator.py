from datetime import datetime
from typing import Any

from .section import Section, SectionType


def get_section(sections: list[Section], value: Any) -> Section | None:
    # Standard format has up to 4 sections:
    # Positive;Negative;Zero;Text
    if isinstance(value, str):
        if len(sections) >= 4:
            return sections[3]
        return get_first_section(sections, SectionType.Text)

    elif isinstance(value, datetime):
        return get_first_section(sections, SectionType.Date)

    elif isinstance(value, int | float):
        return get_numeric_section(sections, value)


def get_first_section(sections: list[Section], _type: SectionType) -> Section | None:
    for section in sections:
        if section.section_type == _type:
            return section


def get_numeric_section(sections: list[Section], value: float) -> Section | None:
    # First section applies if
    # - Has a condition:
    # - There is 1 section, or
    # - There are 2 sections, and the value is 0 or positive, or
    # - There are >2 sections, and the value is positive
    if not sections:
        return

    section0 = sections[0]
    sections_len = len(sections)

    if section0.condition:
        if section0.condition.evaluate(value):
            return section0
    elif (
        sections_len == 1
        or (sections_len == 2 and value >= 0)
        or (sections_len >= 2 and value > 0)
    ):
        return section0

    if sections_len < 2:
        return

    section1 = sections[1]

    # First condition didn't match, or was a negative number. Second condition applies if:
    # - Has a condition, or
    # - Value is negative, or
    # - There are two sections, and the first section had a non-matching condition
    if section1.condition:
        if section1.condition.evaluate(value):
            return section1

    elif value < 0 or (sections_len == 2 and section0.condition):
        return section1

    # Second condition didn't match, or was positive. The following
    # sections cannot have conditions, always fall back to the third
    # section (for zero formatting) if specified.
    if sections_len < 3:
        return

    return sections[2]
