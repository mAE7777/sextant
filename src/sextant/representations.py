"""Small executable representation interfaces. Unknown input stays unknown.

These interpreters implement finite, inspectable semantics. They do not establish
that a proposed graph or program describes the world. Programs are dispatched to
the OS sandbox by the application, never evaluated in this process.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from fractions import Fraction
from typing import Any

from sextant.domain import DomainError, Representation, canonical, finite_number


class Unknown(Exception):
    pass


def expression(node: Any, inputs: dict[str, Any], *, depth: int = 0) -> Any:
    if depth > 32:
        raise DomainError("expression_depth", "expression exceeds the finite interpreter depth")
    if not isinstance(node, dict):
        return node
    if set(node) == {"var"}:
        if not isinstance(node["var"], str):
            raise DomainError("expression", "variable name must be a string")
        if node["var"] not in inputs:
            raise Unknown(str(node["var"]))
        return inputs[node["var"]]
    if set(node) == {"literal"}:
        return node["literal"]
    if set(node) != {"op", "args"} or not isinstance(node["args"], list):
        raise DomainError("expression", "use {var}, {literal}, or {op,args}")
    values = [expression(v, inputs, depth=depth + 1) for v in node["args"]]
    op = node["op"]
    if not isinstance(op, str):
        raise DomainError("expression", "operator must be a string")
    if op in ("and", "or", "xor", "not"):
        if not all(isinstance(v, bool) for v in values):
            raise DomainError("expression_type", "logical operators require boolean inputs")
        if op == "not" and len(values) == 1:
            return not values[0]
        if op == "xor" and len(values) == 2:
            return values[0] != values[1]
        if op == "and":
            return all(values)
        if op == "or":
            return any(values)
    if op in ("eq", "ne") and len(values) == 2:
        equal = canonical(values[0]) == canonical(values[1])
        return equal if op == "eq" else not equal
    if op in ("add", "sub", "mul", "div", "lt", "le", "gt", "ge") and len(values) == 2:
        if not all(finite_number(v) for v in values):
            raise DomainError("expression_type", "numeric operators require finite numbers")
        functions: dict[str, Callable[[Any, Any], Any]] = {
            "add": operator.add, "sub": operator.sub, "mul": operator.mul, "div": operator.truediv,
            "lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge,
        }
        try:
            result = functions[op](*values)
            canonical(result)
            return result
        except (ArithmeticError, ValueError) as exc:
            raise DomainError("undefined_expression", "expression is undefined on these inputs") from exc
    raise DomainError("expression_operator", "unsupported operator or arity")


def evaluate(representation: Representation, inputs: dict[str, Any]) -> dict[str, Any]:
    """Return an explicit result/loss, not a prose-only interface bearing an executable label."""
    body = representation.body
    try:
        if representation.kind == "table":
            columns = body["inputs"]
            if not isinstance(columns, list) or not all(isinstance(k, str) for k in columns) or len(set(columns)) != len(columns):
                raise DomainError("table", "inputs must name table columns")
            rows = body["rows"]
            if not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get("input"), dict)
                                                or set(r["input"]) != set(columns) or "output" not in r for r in rows):
                raise DomainError("table", "rows must have an input object matching the columns and an output")
            if any(k not in inputs for k in columns):
                raise Unknown("missing table column")
            matches = [r["output"] for r in rows
                       if canonical(r["input"]) == canonical({k: inputs[k] for k in columns})]
            if not matches:
                raise Unknown("input outside the finite table")
            if len({canonical(m) for m in matches}) != 1:
                raise DomainError("table_conflict", "table contains conflicting outputs")
            result = matches[0]
        elif representation.kind == "argument_graph":
            # Only typed expressions supply executable inference. Narrative support edges
            # can be retained in metadata but cannot impersonate such an inference.
            values = dict(inputs)
            nodes = body["nodes"]
            if not isinstance(nodes, list) or any(not isinstance(n, dict) or not isinstance(n.get("id"), str)
                                                 or not n["id"] or "expression" not in n for n in nodes):
                raise DomainError("graph", "nodes must contain string IDs and expressions")
            if not isinstance(body["outputs"], dict):
                raise DomainError("graph", "outputs must map names to expressions")
            ids = [n["id"] for n in nodes]
            if len(set(ids)) != len(ids) or set(ids) & set(inputs):
                raise DomainError("graph_identity", "duplicate or shadowing node IDs")
            pending = nodes[:]
            while pending:
                progressed = False
                for node in pending[:]:
                    try:
                        value = expression(node["expression"], values)
                    except Unknown:
                        continue
                    values[node["id"]] = value
                    pending.remove(node)
                    progressed = True
                if not progressed:
                    raise Unknown("cycle or unresolved premise")
            result = {name: expression(expr, values) for name, expr in body["outputs"].items()}
        elif representation.kind == "conditional_policy":
            selected = []
            unknown_conditions = []
            branches = body["branches"]
            if not isinstance(branches, list) or any(not isinstance(b, dict) or "when" not in b or "then" not in b for b in branches):
                raise DomainError("policy", "branches must contain when/then objects")
            for branch in branches:
                try:
                    match = expression(branch["when"], inputs)
                except Unknown as exc:
                    unknown_conditions.append(str(exc))
                    continue
                if not isinstance(match, bool):
                    raise DomainError("policy_condition", "policy condition must return boolean")
                if match:
                    selected.append(branch["then"])
            return {"status": "partial" if unknown_conditions else "computed", "value": selected,
                    "unknown": unknown_conditions, "semantics": "all applicable branches; no implicit tie breaker",
                    "loss": representation.loss, "empirical_validity": "not established by execution"}
        elif representation.kind == "program":
            if not isinstance(body.get("program"), str):
                raise DomainError("program", "program representation requires Python source")
            return {"status": "requires_sandbox", "request": {"program": body["program"], "inputs": inputs},
                    "loss": representation.loss, "empirical_validity": "not established by execution"}
        else:
            return {"status": "unsupported", "loss": representation.loss or ["no executable adapter"]}
        canonical(result)
        return {"status": "computed", "value": result, "loss": representation.loss,
                "empirical_validity": "not established by execution"}
    except Unknown as exc:
        return {"status": "unknown", "reason": str(exc), "loss": representation.loss}
    except (KeyError, TypeError, ValueError, DomainError) as exc:
        return {"status": "unsupported", "reason": str(exc), "loss": [*representation.loss, "invalid executable interface"]}


def arithmetic(goal: str) -> dict[str, Any] | None:
    """A bounded, explicitly requested arithmetic language; no general Python eval.

    Computed exactly over rationals: every literal is read from its source text, so
    "0.1 + 0.2" is exactly 3/10, never a binary-float approximation. ``value`` is a
    JSON number only when that number states the exact result; otherwise it is the
    string "p/q". ``exact`` always carries the numerator and denominator.
    """
    text = goal.strip()
    prefixes = ("calculate ", "compute ", "计算 ", "计算：", "计算:")
    prefix = next((p for p in prefixes if text.lower().startswith(p)), None)
    if prefix is None:
        return None
    source = text[len(prefix):].strip()
    if len(source) > 512:
        return None
    limit = 10**100
    try:
        tree = ast.parse(source, mode="eval")
        count = 0

        def bounded(value: Fraction) -> Fraction:
            if abs(value) > limit or value.denominator > limit:
                raise ValueError("number outside operational range")
            return value

        def visit(node: ast.AST) -> Fraction:
            nonlocal count
            count += 1
            if count > 64:
                raise ValueError("arithmetic complexity")
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and finite_number(node.value):
                literal = ast.get_source_segment(source, node)
                if isinstance(node.value, float):
                    if literal is None:
                        raise ValueError("literal text unavailable")
                    return bounded(Fraction(literal.replace("_", "")))
                return bounded(Fraction(node.value))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                v = visit(node.operand)
                return v if isinstance(node.op, ast.UAdd) else -v
            if isinstance(node, ast.BinOp):
                operations: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
                    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
                }
                if type(node.op) in operations:
                    return bounded(Fraction(operations[type(node.op)](visit(node.left), visit(node.right))))
            raise ValueError("outside arithmetic language")

        exact = visit(tree.body)
        value: int | float | str
        if exact.denominator == 1:
            value = int(exact)
        else:
            nearest = float(exact)
            # A decimal JSON number states the exact result only if it reads back as the same rational.
            value = nearest if Fraction(repr(nearest)) == exact else f"{exact.numerator}/{exact.denominator}"
        return {"expression": source, "value": value, "exact": {"numerator": exact.numerator, "denominator": exact.denominator},
                "protocol": "bounded-arithmetic/2", "units": "dimensionless; no unstated conversion", "provenance": "constructed"}
    except (SyntaxError, ValueError, ArithmeticError):
        return None
