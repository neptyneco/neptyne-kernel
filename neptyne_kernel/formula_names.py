from inspect import getmembers, isclass, isfunction
from types import ModuleType


def _get_names_from_module(obj: ModuleType, prefix: str = "") -> set[str]:
    result = {
        prefix + fn_name
        for (fn_name, _fn) in getmembers(obj, isfunction)
        if fn_name.upper() == fn_name
    }
    # We may have Excel function inside class or other function
    for fn_name, _fn in getmembers(obj, lambda m: isclass(m) or isfunction(m)):
        if fn_name.upper() == fn_name and _fn != obj:
            result.update(_get_names_from_module(_fn, prefix + fn_name + "."))
    return result


FORMULA_NAMES: set[str]
try:
    from . import formulas

    FORMULA_NAMES = _get_names_from_module(formulas)
except ImportError:
    FORMULA_NAMES = set()
