import ast
import inspect
from collections import defaultdict
from typing import Any


def get_code_dependencies(
    code: str, globals: dict[str, Any]
) -> tuple[dict[str, set], dict[str, str]]:
    """
    :param code: module source code
    :param globals: module globals
    :return: dict {name: set of names dependent on this name }, dict {name: source code for name}
    """
    dependencies = defaultdict(set)
    source_for_name: dict[str, str] = {}

    def maybe_add_dependent_node(child: str, parent: str) -> None:
        if child != parent and child in globals:
            inner_module = inspect.getmodule(globals[child])
            if inner_module is None or inner_module.__name__ == "__main__":
                dependencies[parent].add(child)

    def visit_name(node: ast.Name) -> str:
        return node.id

    def visit_call(node: ast.Call) -> str:
        func = node.func
        if isinstance(func, ast.Name):
            return visit_name(func)
        else:
            # func is an instance of ast.Attribute
            value = func.value  # type: ignore
            if isinstance(value, ast.Call):
                return visit_call(value)
            elif isinstance(value, ast.Name):
                return visit_name(value)
        return ""

    def visit_function_def(child_node: ast.AST, parent_node_name: str) -> None:
        arg_names = {arg.arg for arg in child_node.args.args}  # type: ignore
        for n in ast.walk(child_node):
            if isinstance(n, ast.Name) and not (
                n.id in arg_names or isinstance(n.ctx, ast.Param)
            ):
                maybe_add_dependent_node(n.id, parent_node_name)

    def process_call_or_name(node: ast.AST) -> str:
        if isinstance(node, ast.Call):
            return visit_call(node)
        if isinstance(node, ast.Name):
            return visit_name(node)
        return ""

    def process_assign(target: ast.Name, value: ast.AST) -> None:
        for n in ast.walk(value):
            if name := process_call_or_name(n):
                maybe_add_dependent_node(name, target.id)

    def process_list_or_tuple_assign(
        targets: list[ast.expr], values: list[ast.expr]
    ) -> None:
        for target, value in zip(targets, values):
            if isinstance(target, ast.Name):
                source_for_name[target.id] = ast.unparse(value)
                process_assign(target, value)
            elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                # Recursive process list/tuple assignment
                targets = target.elts
                values = value.elts  # type: ignore
                process_list_or_tuple_assign(targets, values)

    def iter_module_or_class(
        node: ast.AST | list[ast.stmt], parent_node_name: str | None = None
    ) -> None:
        nodes = ast.iter_child_nodes(node) if parent_node_name is None else node  # type: ignore
        for child_node in nodes:  # type: ignore
            if isinstance(child_node, ast.ClassDef):
                if parent_node_name is None:
                    parent = child_node.name
                    source_for_name[child_node.name] = ast.unparse(child_node)
                else:
                    parent = parent_node_name
                iter_module_or_class(child_node.body, parent)

            elif isinstance(child_node, ast.FunctionDef | ast.AsyncFunctionDef):
                visit_function_def(child_node, parent_node_name or child_node.name)
                source_for_name[child_node.name] = ast.unparse(child_node)

            elif isinstance(child_node, ast.Assign):
                if isinstance(child_node.targets[0], ast.Tuple) or isinstance(
                    child_node.targets[0], ast.List
                ):
                    targets = child_node.targets[0].elts
                    values = child_node.value.elts  # type: ignore
                    process_list_or_tuple_assign(targets, values)
                else:
                    target = child_node.targets[0]
                    if isinstance(target, ast.Name):
                        source_for_name[target.id] = ast.unparse(child_node.value)
                        process_assign(target, child_node.value)

    def flatten(parent: str, children: set[str]) -> None:
        """Flatten multilevel dependencies"""
        for child in list(children):
            if child != parent and child not in (depends_on := dependencies[parent]):
                depends_on.add(child)
                flatten(parent, dependencies.get(child, set()))

    def reverse(depends_on: dict[str, set[str]]) -> dict[str, set[str]]:
        """Reverse the dependency tree dict {name: names it depends on} into {name: names that depend on this name}"""
        reversed = defaultdict(set)
        for parent, names in depends_on.items():
            for name in names:
                reversed[name].add(parent)
        return reversed

    root = ast.parse(code)
    iter_module_or_class(root)

    for parent, children in dependencies.items():
        flatten(parent, children)

    return reverse(dependencies), source_for_name
