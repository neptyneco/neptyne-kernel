# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = jupyter_notebook_from_dict(json.loads(json_string))

from enum import Enum
from typing import Any, Callable, TypeVar, cast

T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):  # type: ignore
    for f in fs:
        try:
            return f(x)
        except:  # noqa: E722
            pass
    assert False


def from_dict(f: Callable[[Any], T], x: Any) -> dict[str, T]:
    assert isinstance(x, dict)
    return {k: f(v) for (k, v) in x.items()}


def from_list(f: Callable[[Any], T], x: Any) -> list[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def to_class(c: type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def to_enum(c: type[EnumT], x: Any) -> EnumT:
    assert isinstance(x, c)
    return x.value


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


class CellType(Enum):
    """String identifying the type of cell."""

    CODE = "code"
    MARKDOWN = "markdown"
    RAW = "raw"


class Execution:
    """Execution time for the code in the cell. This tracks time at which messages are received
    from iopub or shell channels
    """

    """header.date (in ISO 8601 format) of iopub channel's execute_input message. It indicates
    the time at which the kernel broadcasts an execute_input message to connected frontends
    """
    iopub_execute_input: str | None
    """header.date (in ISO 8601 format) of iopub channel's kernel status message when the status
    is 'busy'
    """
    iopub_status_busy: str | None
    """header.date (in ISO 8601 format) of iopub channel's kernel status message when the status
    is 'idle'. It indicates the time at which kernel finished processing the associated
    request
    """
    iopub_status_idle: str | None
    """header.date (in ISO 8601 format) of the shell channel's execute_reply message. It
    indicates the time at which the execute_reply message was created
    """
    shell_execute_reply: str | None

    def __init__(
        self,
        iopub_execute_input: str | None,
        iopub_status_busy: str | None,
        iopub_status_idle: str | None,
        shell_execute_reply: str | None,
    ) -> None:
        self.iopub_execute_input = iopub_execute_input
        self.iopub_status_busy = iopub_status_busy
        self.iopub_status_idle = iopub_status_idle
        self.shell_execute_reply = shell_execute_reply

    @staticmethod
    def from_dict(obj: Any) -> "Execution":
        assert isinstance(obj, dict)
        iopub_execute_input = from_union(
            [from_str, from_none], obj.get("iopub.execute_input")
        )
        iopub_status_busy = from_union(
            [from_str, from_none], obj.get("iopub.status.busy")
        )
        iopub_status_idle = from_union(
            [from_str, from_none], obj.get("iopub.status.idle")
        )
        shell_execute_reply = from_union(
            [from_str, from_none], obj.get("shell.execute_reply")
        )
        return Execution(
            iopub_execute_input,
            iopub_status_busy,
            iopub_status_idle,
            shell_execute_reply,
        )

    def to_dict(self) -> dict:
        result: dict = {}
        result["iopub.execute_input"] = from_union(
            [from_str, from_none], self.iopub_execute_input
        )
        result["iopub.status.busy"] = from_union(
            [from_str, from_none], self.iopub_status_busy
        )
        result["iopub.status.idle"] = from_union(
            [from_str, from_none], self.iopub_status_idle
        )
        result["shell.execute_reply"] = from_union(
            [from_str, from_none], self.shell_execute_reply
        )
        return result


class ScrolledEnum(Enum):
    AUTO = "auto"


class CellMetadata:
    """Cell-level metadata."""

    """Raw cell metadata format for nbconvert."""
    format: str | None
    """Official Jupyter Metadata for Raw Cells
    
    Official Jupyter Metadata for Markdown Cells
    
    Official Jupyter Metadata for Code Cells
    """
    jupyter: dict[str, Any] | None
    name: str | None
    tags: list[str] | None
    """Whether the cell's output is collapsed/expanded."""
    collapsed: bool | None
    """Execution time for the code in the cell. This tracks time at which messages are received
    from iopub or shell channels
    """
    execution: Execution | None
    """Whether the cell's output is scrolled, unscrolled, or autoscrolled."""
    scrolled: bool | ScrolledEnum | None

    def __init__(
        self,
        format: str | None,
        jupyter: dict[str, Any] | None,
        name: str | None,
        tags: list[str] | None,
        collapsed: bool | None,
        execution: Execution | None,
        scrolled: bool | ScrolledEnum | None,
    ) -> None:
        self.format = format
        self.jupyter = jupyter
        self.name = name
        self.tags = tags
        self.collapsed = collapsed
        self.execution = execution
        self.scrolled = scrolled

    @staticmethod
    def from_dict(obj: Any) -> "CellMetadata":
        assert isinstance(obj, dict)
        format = from_union([from_str, from_none], obj.get("format"))
        jupyter = from_union(
            [lambda x: from_dict(lambda x: x, x), from_none], obj.get("jupyter")
        )
        name = from_union([from_str, from_none], obj.get("name"))
        tags = from_union(
            [lambda x: from_list(from_str, x), from_none], obj.get("tags")
        )
        collapsed = from_union([from_bool, from_none], obj.get("collapsed"))
        execution = from_union([Execution.from_dict, from_none], obj.get("execution"))
        scrolled = from_union([from_bool, ScrolledEnum, from_none], obj.get("scrolled"))
        return CellMetadata(format, jupyter, name, tags, collapsed, execution, scrolled)

    def to_dict(self) -> dict:
        result: dict = {}
        result["format"] = from_union([from_str, from_none], self.format)
        result["jupyter"] = from_union(
            [lambda x: from_dict(lambda x: x, x), from_none], self.jupyter
        )
        result["name"] = from_union([from_str, from_none], self.name)
        result["tags"] = from_union(
            [lambda x: from_list(from_str, x), from_none], self.tags
        )
        result["collapsed"] = from_union([from_bool, from_none], self.collapsed)
        result["execution"] = from_union(
            [lambda x: to_class(Execution, x), from_none], self.execution
        )
        result["scrolled"] = from_union(
            [from_bool, lambda x: to_enum(ScrolledEnum, x), from_none], self.scrolled
        )
        return result


class OutputType(Enum):
    """Type of cell output."""

    DISPLAY_DATA = "display_data"
    ERROR = "error"
    EXECUTE_RESULT = "execute_result"
    STREAM = "stream"


class Output:
    """Result of executing a code cell.

    Data displayed as a result of code cell execution.

    Stream output from a code cell.

    Output of an error that occurred during code cell execution.
    """

    data: dict[str, list[str] | str] | None
    """A result's prompt number."""
    execution_count: int | None
    metadata: dict[str, Any] | None
    """Type of cell output."""
    output_type: OutputType
    """The name of the stream (stdout, stderr)."""
    name: str | None
    """The stream's text output, represented as an array of strings."""
    text: list[str] | None | str
    """The name of the error."""
    ename: str | None
    """The value, or message, of the error."""
    evalue: str | None
    """The error's traceback, represented as an array of strings."""
    traceback: list[str] | None

    def __init__(
        self,
        data: dict[str, list[str] | str] | None,
        execution_count: int | None,
        metadata: dict[str, Any] | None,
        output_type: OutputType,
        name: str | None,
        text: list[str] | None | str,
        ename: str | None,
        evalue: str | None,
        traceback: list[str] | None,
    ) -> None:
        self.data = data
        self.execution_count = execution_count
        self.metadata = metadata
        self.output_type = output_type
        self.name = name
        self.text = text
        self.ename = ename
        self.evalue = evalue
        self.traceback = traceback

    @staticmethod
    def from_dict(obj: Any) -> "Output":
        assert isinstance(obj, dict)
        data = from_union(
            [
                lambda x: from_dict(
                    lambda x: from_union(
                        [lambda x: from_list(from_str, x), from_str], x
                    ),
                    x,
                ),
                from_none,
            ],
            obj.get("data"),
        )
        execution_count = from_union([from_none, from_int], obj.get("execution_count"))
        metadata = from_union(
            [lambda x: from_dict(lambda x: x, x), from_none], obj.get("metadata")
        )
        output_type = OutputType(obj.get("output_type"))
        name = from_union([from_str, from_none], obj.get("name"))
        text = from_union(
            [lambda x: from_list(from_str, x), from_str, from_none], obj.get("text")
        )
        ename = from_union([from_str, from_none], obj.get("ename"))
        evalue = from_union([from_str, from_none], obj.get("evalue"))
        traceback = from_union(
            [lambda x: from_list(from_str, x), from_none], obj.get("traceback")
        )
        return Output(
            data,
            execution_count,
            metadata,
            output_type,
            name,
            text,
            ename,
            evalue,
            traceback,
        )

    def to_dict(self) -> dict:
        result: dict = {}
        result["data"] = from_union(
            [
                lambda x: from_dict(
                    lambda x: from_union(
                        [lambda x: from_list(from_str, x), from_str], x
                    ),
                    x,
                ),
                from_none,
            ],
            self.data,
        )
        result["execution_count"] = from_union(
            [from_none, from_int], self.execution_count
        )
        result["metadata"] = from_union(
            [lambda x: from_dict(lambda x: x, x), from_none], self.metadata
        )
        result["output_type"] = to_enum(OutputType, self.output_type)
        result["name"] = from_union([from_str, from_none], self.name)
        result["text"] = from_union(
            [lambda x: from_list(from_str, x), from_str, from_none], self.text
        )
        result["ename"] = from_union([from_str, from_none], self.ename)
        result["evalue"] = from_union([from_str, from_none], self.evalue)
        result["traceback"] = from_union(
            [lambda x: from_list(from_str, x), from_none], self.traceback
        )
        return result


class Cell:
    """Notebook raw nbconvert cell.

    Notebook markdown cell.

    Notebook code cell.
    """

    attachments: dict[str, dict[str, list[str] | str]] | None
    """String identifying the type of cell."""
    cell_type: CellType
    """Cell-level metadata."""
    metadata: CellMetadata
    source: list[str] | str
    """The code cell's prompt number. Will be null if the cell has not been run."""
    execution_count: int | None
    """Execution, display, or stream outputs."""
    outputs: list[Output] | None

    def __init__(
        self,
        attachments: dict[str, dict[str, list[str] | str]] | None,
        cell_type: CellType,
        metadata: CellMetadata,
        source: list[str] | str,
        execution_count: int | None,
        outputs: list[Output] | None,
    ) -> None:
        self.attachments = attachments
        self.cell_type = cell_type
        self.metadata = metadata
        self.source = source
        self.execution_count = execution_count
        self.outputs = outputs

    @staticmethod
    def from_dict(obj: Any) -> "Cell":
        assert isinstance(obj, dict)
        attachments = from_union(
            [
                lambda x: from_dict(
                    lambda x: from_dict(
                        lambda x: from_union(
                            [lambda x: from_list(from_str, x), from_str], x
                        ),
                        x,
                    ),
                    x,
                ),
                from_none,
            ],
            obj.get("attachments"),
        )
        cell_type = CellType(obj.get("cell_type"))
        metadata = CellMetadata.from_dict(obj.get("metadata"))
        source = from_union(
            [lambda x: from_list(from_str, x), from_str], obj.get("source")
        )
        execution_count = from_union([from_none, from_int], obj.get("execution_count"))
        outputs = from_union(
            [lambda x: from_list(Output.from_dict, x), from_none], obj.get("outputs")
        )
        return Cell(attachments, cell_type, metadata, source, execution_count, outputs)

    def to_dict(self) -> dict:
        result: dict = {}
        result["attachments"] = from_union(
            [
                lambda x: from_dict(
                    lambda x: from_dict(
                        lambda x: from_union(
                            [lambda x: from_list(from_str, x), from_str], x
                        ),
                        x,
                    ),
                    x,
                ),
                from_none,
            ],
            self.attachments,
        )
        result["cell_type"] = to_enum(CellType, self.cell_type)
        result["metadata"] = to_class(CellMetadata, self.metadata)
        result["source"] = from_union(
            [lambda x: from_list(from_str, x), from_str], self.source
        )
        result["execution_count"] = from_union(
            [from_none, from_int], self.execution_count
        )
        result["outputs"] = from_union(
            [lambda x: from_list(lambda x: to_class(Output, x), x), from_none],
            self.outputs,
        )
        return result


class Kernelspec:
    """Kernel information."""

    """Name to display in UI."""
    display_name: str
    """Name of the kernel specification."""
    name: str

    def __init__(self, display_name: str, name: str) -> None:
        self.display_name = display_name
        self.name = name

    @staticmethod
    def from_dict(obj: Any) -> "Kernelspec":
        assert isinstance(obj, dict)
        display_name = from_str(obj.get("display_name"))
        name = from_str(obj.get("name"))
        return Kernelspec(display_name, name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["display_name"] = from_str(self.display_name)
        result["name"] = from_str(self.name)
        return result


class LanguageInfo:
    """Kernel information."""

    """The codemirror mode to use for code in this language."""
    codemirror_mode: dict[str, Any] | None | str
    """The file extension for files in this language."""
    file_extension: str | None
    """The mimetype corresponding to files in this language."""
    mimetype: str | None
    """The programming language which this kernel runs."""
    name: str
    """The pygments lexer to use for code in this language."""
    pygments_lexer: str | None

    def __init__(
        self,
        codemirror_mode: dict[str, Any] | None | str,
        file_extension: str | None,
        mimetype: str | None,
        name: str,
        pygments_lexer: str | None,
    ) -> None:
        self.codemirror_mode = codemirror_mode
        self.file_extension = file_extension
        self.mimetype = mimetype
        self.name = name
        self.pygments_lexer = pygments_lexer

    @staticmethod
    def from_dict(obj: Any) -> "LanguageInfo":
        assert isinstance(obj, dict)
        codemirror_mode = from_union(
            [lambda x: from_dict(lambda x: x, x), from_str, from_none],
            obj.get("codemirror_mode"),
        )
        file_extension = from_union([from_str, from_none], obj.get("file_extension"))
        mimetype = from_union([from_str, from_none], obj.get("mimetype"))
        name = from_str(obj.get("name"))
        pygments_lexer = from_union([from_str, from_none], obj.get("pygments_lexer"))
        return LanguageInfo(
            codemirror_mode, file_extension, mimetype, name, pygments_lexer
        )

    def to_dict(self) -> dict:
        result: dict = {}
        result["codemirror_mode"] = from_union(
            [lambda x: from_dict(lambda x: x, x), from_str, from_none],
            self.codemirror_mode,
        )
        result["file_extension"] = from_union(
            [from_str, from_none], self.file_extension
        )
        result["mimetype"] = from_union([from_str, from_none], self.mimetype)
        result["name"] = from_str(self.name)
        result["pygments_lexer"] = from_union(
            [from_str, from_none], self.pygments_lexer
        )
        return result


class JupyterNotebookMetadata:
    """Notebook root-level metadata."""

    """The author(s) of the notebook document"""
    authors: list[Any] | None
    """Kernel information."""
    kernelspec: Kernelspec | None
    """Kernel information."""
    language_info: LanguageInfo | None
    """Original notebook format (major number) before converting the notebook between versions.
    This should never be written to a file.
    """
    orig_nbformat: int | None
    """The title of the notebook document"""
    title: str | None

    def __init__(
        self,
        authors: list[Any] | None,
        kernelspec: Kernelspec | None,
        language_info: LanguageInfo | None,
        orig_nbformat: int | None,
        title: str | None,
    ) -> None:
        self.authors = authors
        self.kernelspec = kernelspec
        self.language_info = language_info
        self.orig_nbformat = orig_nbformat
        self.title = title

    @staticmethod
    def from_dict(obj: Any) -> "JupyterNotebookMetadata":
        assert isinstance(obj, dict)
        authors = from_union(
            [lambda x: from_list(lambda x: x, x), from_none], obj.get("authors")
        )
        kernelspec = from_union(
            [Kernelspec.from_dict, from_none], obj.get("kernelspec")
        )
        language_info = from_union(
            [LanguageInfo.from_dict, from_none], obj.get("language_info")
        )
        orig_nbformat = from_union([from_int, from_none], obj.get("orig_nbformat"))
        title = from_union([from_str, from_none], obj.get("title"))
        return JupyterNotebookMetadata(
            authors, kernelspec, language_info, orig_nbformat, title
        )

    def to_dict(self) -> dict:
        result: dict = {}
        result["authors"] = from_union(
            [lambda x: from_list(lambda x: x, x), from_none], self.authors
        )
        result["kernelspec"] = from_union(
            [lambda x: to_class(Kernelspec, x), from_none], self.kernelspec
        )
        result["language_info"] = from_union(
            [lambda x: to_class(LanguageInfo, x), from_none], self.language_info
        )
        result["orig_nbformat"] = from_union([from_int, from_none], self.orig_nbformat)
        result["title"] = from_union([from_str, from_none], self.title)
        return result


class JupyterNotebook:
    """Jupyter Notebook v4.4 JSON schema."""

    """Array of cells of the current notebook."""
    cells: list[Cell]
    """Notebook root-level metadata."""
    metadata: JupyterNotebookMetadata
    """Notebook format (major number). Incremented between backwards incompatible changes to the
    notebook format.
    """
    nbformat: int
    """Notebook format (minor number). Incremented for backward compatible changes to the
    notebook format.
    """
    nbformat_minor: int

    def __init__(
        self,
        cells: list[Cell],
        metadata: JupyterNotebookMetadata,
        nbformat: int,
        nbformat_minor: int,
    ) -> None:
        self.cells = cells
        self.metadata = metadata
        self.nbformat = nbformat
        self.nbformat_minor = nbformat_minor

    @staticmethod
    def from_dict(obj: Any) -> "JupyterNotebook":
        assert isinstance(obj, dict)
        cells = from_list(Cell.from_dict, obj.get("cells"))
        metadata = JupyterNotebookMetadata.from_dict(obj.get("metadata"))
        nbformat = from_int(obj.get("nbformat"))
        nbformat_minor = from_int(obj.get("nbformat_minor"))
        return JupyterNotebook(cells, metadata, nbformat, nbformat_minor)

    def to_dict(self) -> dict:
        result: dict = {}
        result["cells"] = from_list(lambda x: to_class(Cell, x), self.cells)
        result["metadata"] = to_class(JupyterNotebookMetadata, self.metadata)
        result["nbformat"] = from_int(self.nbformat)
        result["nbformat_minor"] = from_int(self.nbformat_minor)
        return result


def jupyter_notebook_from_dict(s: Any) -> JupyterNotebook:
    return JupyterNotebook.from_dict(s)


def jupyter_notebook_to_dict(x: JupyterNotebook) -> Any:
    return to_class(JupyterNotebook, x)
