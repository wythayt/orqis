from __future__ import annotations

import inspect
import json
import re
import textwrap
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, get_type_hints

try:
    from typing_extensions import is_typeddict
except ImportError:  # pragma: no cover
    def is_typeddict(tp: Any) -> bool:
        return hasattr(tp, "__annotations__") and hasattr(tp, "__total__")


def unwrap_callable(obj: Any) -> Any:
    current = obj
    seen: set[int] = set()
    while current is not None and hasattr(current, "func") and id(current) not in seen:
        seen.add(id(current))
        func = getattr(current, "func")
        if func is None:
            break
        current = func
    return current


def callable_ref(obj: Any) -> str | None:
    target = unwrap_callable(obj)
    if target is None:
        return None
    if inspect.ismethod(target):
        target = target.__func__
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", getattr(target, "__name__", None))
    if qualname is None:
        qualname = target.__class__.__qualname__
    if module:
        return f"{module}.{qualname}"
    return qualname


def type_name(tp: Any) -> str:
    if tp is None:
        return "None"
    if isinstance(tp, str):
        return tp
    module = getattr(tp, "__module__", "")
    if module in {"typing", "typing_extensions"} or hasattr(tp, "__args__"):
        return str(tp)
    if getattr(tp, "__module__", "") == "builtins" and hasattr(tp, "__name__"):
        return tp.__name__
    return str(tp)


def typed_dict_annotations(schema: Any) -> dict[str, Any]:
    if schema is None:
        return {}
    if not is_typeddict(schema):
        return {}
    try:
        return get_type_hints(schema, include_extras=True)
    except Exception:
        return dict(getattr(schema, "__annotations__", {}))


def typed_dict_keys(schema: Any) -> list[str]:
    return list(typed_dict_annotations(schema).keys())


def safe_getsource(obj: Any) -> str | None:
    target = unwrap_callable(obj)
    if target is None:
        return None
    try:
        return textwrap.dedent(inspect.getsource(target))
    except (OSError, TypeError):
        return None


def sanitize_identifier(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    return sanitized or "graph"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def topological_layers(nodes: list[str], edges: Mapping[str, set[str]]) -> list[list[str]]:
    indegree = {node: 0 for node in nodes}
    for src, dests in edges.items():
        for dst in dests:
            if dst in indegree:
                indegree[dst] += 1
    remaining = set(nodes)
    layers: list[list[str]] = []
    while remaining:
        layer = sorted(node for node in remaining if indegree[node] == 0)
        if not layer:
            layers.append(sorted(remaining))
            break
        layers.append(layer)
        for node in layer:
            remaining.remove(node)
            for dst in edges.get(node, set()):
                if dst in indegree:
                    indegree[dst] -= 1
    return layers


def payload_units(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, str):
        return max(1.0, len(value) / 32.0)
    if isinstance(value, (int, float, bool)):
        return 1.0
    if isinstance(value, Mapping):
        return 2.0 + sum(payload_units(key) + payload_units(val) for key, val in value.items())
    if isinstance(value, set):
        return 1.0 + sum(payload_units(item) for item in value)
    if isinstance(value, (list, tuple)):
        return 1.0 + sum(payload_units(item) for item in value)
    return 3.0


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return to_jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return to_jsonable(value.dict())
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, set):
        return [to_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if callable(value):
        return callable_ref(value)
    if isinstance(value, type):
        return type_name(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=False)
        handle.write("\n")
