from __future__ import annotations

from typing import Any, Callable

from orqis.compiler.utils import unwrap_callable


ORQIS_ATTR = "__orqis__"
REGISTRY_KEYS = {
    "assets",
    "memory_policies",
    "executors",
    "tools",
    "tool_bindings",
    "skills",
    "subagents",
    "node_overrides",
}
NODE_METADATA_KEYS = {
    "skills",
    "tool_bindings",
    "subagent",
    "executor",
    "approval_policy",
    "resources",
    "side_effects",
    "cache_policy",
    "retry_policy",
}


def orqis(_target: Callable[..., Any] | None = None, /, **payload: Any):
    # allow both @orqis and @orqis(...), but prefer the explicit call form.
    def decorate(target: Callable[..., Any]):
        existing = extract_orqis_payload(target)
        merged = merge_orqis_payload(existing, payload)
        setattr(target, ORQIS_ATTR, merged)
        unwrapped = unwrap_callable(target)
        if unwrapped is not None and unwrapped is not target:
            setattr(unwrapped, ORQIS_ATTR, merged)
        return target

    if _target is not None:
        return decorate(_target)
    return decorate


def extract_orqis_payload(target: Any) -> dict[str, Any]:
    if target is None:
        return {}
    current = getattr(target, ORQIS_ATTR, None)
    if isinstance(current, dict):
        return dict(current)
    unwrapped = unwrap_callable(target)
    if unwrapped is None:
        return {}
    current = getattr(unwrapped, ORQIS_ATTR, None)
    if isinstance(current, dict):
        return dict(current)
    return {}


def extract_registry_sections(target: Any) -> dict[str, dict[str, Any]]:
    payload = extract_orqis_payload(target)
    sections: dict[str, dict[str, Any]] = {key: {} for key in REGISTRY_KEYS}
    for key in REGISTRY_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            sections[key].update(value)
    return sections


def extract_node_metadata(target: Any) -> dict[str, Any]:
    payload = extract_orqis_payload(target)
    return {
        key: payload[key]
        for key in NODE_METADATA_KEYS
        if key in payload
    }


def merge_orqis_payload(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in REGISTRY_KEYS and isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
            continue
        if key in {"skills", "tool_bindings"} and isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = list(dict.fromkeys([*merged[key], *value]))
            continue
        merged[key] = value
    return merged
