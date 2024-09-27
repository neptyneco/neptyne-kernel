from collections import defaultdict
from typing import TYPE_CHECKING, Any, Sequence

from .cell_address import Address
from .ops import ClearOp, ExecOp

if TYPE_CHECKING:
    from .dash import Dash


class CellExecutionGraph:
    def __init__(
        self,
        dash: "Dash",
        cell_ids: set[Address],
        pre_clear: Sequence[Address] | None,
    ) -> None:
        import networkx as nx
        from graphlib import CycleError, TopologicalSorter

        self.dash = dash
        self.cell_ids = cell_ids
        self.pre_clear = pre_clear
        self.cycle_error_ops: list[ExecOp] | None = []
        self.pre_clear_op = None

        if pre_clear:
            self.pre_clear_op = ClearOp([*pre_clear])

        graph: dict[Address, list[Address]] = defaultdict(list)
        to_process = set(cell_ids)
        while to_process:
            next_cell = to_process.pop()
            if next_cell not in graph:
                graph[next_cell] = []
            # Resolve the graph, but skip anything that is calculated by the to_process we are getting
            # the graph for since we will override those connections. This allows local self reference,
            # i.e. you can put a formula in A1 that reads and writes from a range starting in A1

            feeds_into = [
                cell
                for cell in dash.graph.feeds_into.get(next_cell, ())
                if dash.graph.calculated_by.get(next_cell) != cell
            ]
            to_process.update(cell for cell in feeds_into if cell not in graph)
            for c in feeds_into:
                graph[c].append(next_cell)
        self.ts = TopologicalSorter(graph)
        try:
            self.ts.prepare()
        except CycleError:
            cell_to_cycle = {}
            for cycle in nx.simple_cycles(nx.DiGraph(graph)):
                for cell in cycle:
                    cell_to_cycle[cell] = cycle
            for cycle_id, cycle_cells in cell_to_cycle.items():
                formatted = ", ".join(c.to_a1() for c in sorted(cycle_cells))
                self.cycle_error_ops.append(
                    ExecOp(
                        cycle_id,
                        f"REF_ERROR.with_message('{formatted} refer to each other')",
                    )
                )

    def ready_statements(self) -> Sequence[ExecOp | ClearOp] | None:
        if self.cycle_error_ops:
            ops = self.cycle_error_ops
            self.cycle_error_ops = None
            return ops
        elif self.cycle_error_ops is None:
            return None

        if not self.ts.is_active():
            return None

        statements: list[ExecOp | ClearOp] = []
        if self.pre_clear_op:
            statements.append(self.pre_clear_op)
            self.pre_clear_op = None
        for cell_run in self.ts.get_ready():
            compiled_code = self.dash.get_or_create_cell_meta(cell_run).compiled_code
            calculated_by_to_clear = self.dash.cells_calculated_by(cell_run)
            if compiled_code:
                if op := self.process_clear_statements(calculated_by_to_clear):
                    statements.append(op)
                statements.append(ExecOp(cell_run, compiled_code))
            else:
                # If we evaluated an origin cell and it no longer has code, we need to clear it:
                if cell_run in self.cell_ids and not compiled_code:
                    if op := self.process_clear_statements(calculated_by_to_clear):
                        statements.append(op)
                    statements.append(ClearOp([cell_run]))
                else:
                    self.ts.done(cell_run)

        return statements

    def process_clear_statements(
        self, calculated_by_to_clear: set[Address]
    ) -> ClearOp | None:
        if calculated_by_to_clear:
            for cell_id in calculated_by_to_clear:
                self.dash.unlink(cell_id)
            return ClearOp([*calculated_by_to_clear])

    def done(self, *nodes: Any) -> None:
        for node in nodes:
            try:
                self.ts.done(node)
            except ValueError:
                pass

    def is_active(self) -> bool:
        return bool(self.cycle_error_ops) or self.ts.is_active()
