import json
from typing import IO, Iterable, Type

from pygments import highlight
from pygments.formatter import Formatter
from pygments.formatters import HtmlFormatter
from pygments.lexers.python import PythonLexer
from pygments.token import Token

TokenSource = Iterable[tuple[Token, str]]


class MailchimpFormatter(HtmlFormatter):
    def _format_lines(self, tokensource: TokenSource) -> Iterable[tuple[int, str]]:
        yield from super()._format_lines(_mailchimp_hack(tokensource))


class GSheetsTextRunFormatter(Formatter):
    def __init__(self, **options: dict):
        super().__init__(**options)
        self.pos = 0

    def format(self, tokensource: TokenSource, outfile: IO) -> None:
        runs = []
        trailing_whitespace = 0
        for ix, (ttype, value) in enumerate(tokensource):
            if ttype is Token.Text.Whitespace:
                trailing_whitespace += 1
            else:
                trailing_whitespace = 0
            style = self.style.style_for_token(ttype)
            run = {
                "fontFamily": "Roboto Mono",
                "bold": style["bold"],
                "italic": style["italic"],
                "underline": style["underline"],
            }
            if color := style["color"]:
                run["foregroundColorStyle"] = {
                    "rgbColor": {
                        "red": int(color[:2], 16) / 255,
                        "green": int(color[2:4], 16) / 255,
                        "blue": int(color[4:], 16) / 255,
                    }
                }

            runs.append(
                {
                    "startIndex": self.pos,
                    "format": run,
                }
            )
            self.pos += len(value)
        if trailing_whitespace:
            runs = runs[:-trailing_whitespace]
        json.dump(runs, outfile, indent=2)


def _mailchimp_hack(tokensource: TokenSource) -> TokenSource:
    last_token = None
    for token in tokensource:
        if token == (Token.Punctuation, "]") and last_token == (
            Token.Punctuation,
            "[",
        ):
            yield (Token.NeptyneMagic, "]")
        else:
            yield token
        last_token = token


def highlight_code(text: str | bytes, formatter_cls: Type = HtmlFormatter) -> str:
    formatter = formatter_cls(
        cssstyles="margin: 20px; padding: 5px 10px; white-space: pre; background-color: #E9E9E9",
        nobackground=True,
        wrapcode=True,
        linespans=True,
    )

    output = []
    if formatter_cls is MailchimpFormatter:
        output.append("<style>")
        output.append(formatter.get_style_defs(".highlight"))
        output.append("</style>")

    output.append(
        highlight(
            text,
            PythonLexer(),
            formatter,
        )
    )
    return "\n".join(output)
