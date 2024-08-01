from .evaluator import get_first_section
from .parser import parse_sections
from .section import Section, SectionType


class NumberFormat:
    def __init__(self, format_str: str):
        sections, syntax_error = parse_sections(format_str)
        self.is_valid = not syntax_error
        self.format_str = format_str
        if self.is_valid:
            self.sections = sections
            self.is_date_time_fmt = (
                get_first_section(self.sections, SectionType.Date) is not None
            )
            self.is_time_span_fmt = (
                get_first_section(self.sections, SectionType.Duration) is not None
            )
        else:
            self.sections: list[Section] = []
