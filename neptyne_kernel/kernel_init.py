from IPython.core import interactiveshell as _interactiveshell
from jedi.api import Interpreter as _jedi_Interpreter
from stack_data.core import Source as _Source

import neptyne_kernel.neptyne_api as _neptyne_api  # noqa: F401

from .dash import Dash as _Dash

N_ = _Dash.instance()


def _prime_source_cache():
    # Prime the executing.Source cache so the first kernel traceback isn't slow
    _ = _Source.for_filename(_interactiveshell.__file__)


def _fix_jedi_interpreter_getattr():
    try:
        _jedi_Interpreter._allow_descriptor_getattr_default = False
    except AttributeError:
        pass


_prime_source_cache()
_fix_jedi_interpreter_getattr()
