from dataclasses import dataclass
from typing import Any

from IPython.core.compilerop import CachingCompiler
from pyflakes import checker
from pyflakes.messages import (
    AssertTuple,
    CommentAnnotationSyntaxError,
    DoctestSyntaxError,
    ForwardAnnotationSyntaxError,
    FStringMissingPlaceholders,
    IfTuple,
    ImportShadowedByLoopVar,
    ImportStarUsed,
    IsLiteral,
    LateFutureImport,
    Message,
    MultiValueRepeatedKeyLiteral,
    MultiValueRepeatedKeyVariable,
    PercentFormatExtraNamedArguments,
    PercentFormatMixedPositionalAndNamed,
    PercentFormatPositionalCountMismatch,
    RedefinedWhileUnused,
    StringDotFormatExtraNamedArguments,
    StringDotFormatExtraPositionalArguments,
    UndefinedExport,
    UndefinedLocal,
    UndefinedName,
    UnusedVariable,
)

from .formula_names import FORMULA_NAMES
from .neptyne_protocol import MessageTypes, TracebackFrame

# Other errors are caught by InteractiveShell.run_cell_async
VALID_WARNING_TYPES = {
    RedefinedWhileUnused,
    ImportStarUsed,
    DoctestSyntaxError,
    UndefinedExport,
    UnusedVariable,
    UndefinedLocal,
    MultiValueRepeatedKeyLiteral,
    MultiValueRepeatedKeyVariable,
    LateFutureImport,
    IfTuple,
    AssertTuple,
    ForwardAnnotationSyntaxError,
    CommentAnnotationSyntaxError,
    IsLiteral,
    FStringMissingPlaceholders,
    StringDotFormatExtraPositionalArguments,
    StringDotFormatExtraNamedArguments,
    PercentFormatMixedPositionalAndNamed,
    PercentFormatPositionalCountMismatch,
    PercentFormatExtraNamedArguments,
    ImportShadowedByLoopVar,
}

EXCLUDE_WARNINGS = {
    UndefinedName: lambda x: x in FORMULA_NAMES or x == "N_",
    ImportStarUsed: lambda x: x == "neptyne_kernel.kernel_init",
}


class TyneCachingCompiler(CachingCompiler):
    def ast_parse(
        self, source: str, filename: str = "<unknown>", symbol: str = "exec"
    ) -> Any:
        from .dash import Dash

        tree = super().ast_parse(source, filename, symbol)
        warnings = check_warnings(source, filename, tree)
        if warnings:
            Dash.instance().reply_to_client(MessageTypes.LINTER, {"linter": warnings})
        return tree


def is_valid_warning(warning: Message) -> bool:
    """Skip import error for Excel formulas and N_"""
    warning_type = type(warning)
    if warning_type not in VALID_WARNING_TYPES:
        return False
    if warning_type in EXCLUDE_WARNINGS:
        fn = EXCLUDE_WARNINGS[warning_type]
        args = warning.message_args
        return not fn(*args)
    return True


def check_warnings(code: str, filename: str, tree: Any) -> list[dict]:
    reporter = TynePyFlakesReporter()
    file_tokens = checker.make_tokens(code)
    w = checker.Checker(tree, file_tokens=file_tokens, filename=filename)
    for warning in w.messages:
        reporter.flake(warning)

    return [
        TracebackFrame(True, None, w.line, w.lineno).to_dict()
        for w in reporter.warnings
    ]


@dataclass
class WarningItem:
    line: str
    lineno: int

    def __hash__(self) -> int:
        return hash((self.line, self.lineno))


class TynePyFlakesReporter:
    def __init__(self) -> None:
        # set() removes duplicate warnings (like 'dictionary key repeated' in the same line)
        self.warnings: set[WarningItem] = set()

    def flake(self, warning: Message) -> None:
        if is_valid_warning(warning):
            self.warnings.add(
                WarningItem(warning.message % warning.message_args, warning.lineno)
            )
