from __future__ import annotations

from typing import Any, get_args, get_origin

from orqis.compiler.ir import (
    ApprovalPolicyIR,
    CachePolicyIR,
    ReducerRef,
    ResourceIR,
    RetryPolicyIR,
    RetryRuleIR,
    SideEffectIR,
)
from orqis.compiler.utils import callable_ref, type_name, unwrap_callable


def normalize_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def normalize_string_sequence(value: Any) -> list[str]:
    return [str(item) for item in normalize_sequence(value)]


def channel_kind(channel: Any) -> str:
    class_name = type(channel).__name__
    if class_name == "LastValueAfterFinish":
        return "LastValue"
    if class_name in {"LastValue", "Topic", "BinaryOperatorAggregate"}:
        return class_name
    if class_name == "EphemeralValue":
        return "EphemeralValue"
    return class_name


def extract_reducer(annotation: Any, channel: Any) -> ReducerRef | None:
    if not hasattr(channel, "operator"):
        origin = get_origin(annotation)
        if origin is None or "Annotated" not in str(origin):
            return None
    reducer_fn = getattr(channel, "operator", None)
    if reducer_fn is None:
        args = get_args(annotation)
        reducer_fn = args[1] if len(args) > 1 and callable(args[1]) else None
    if reducer_fn is None:
        return None
    reducer_id = callable_ref(reducer_fn) or type_name(reducer_fn)
    associative = reducer_id in {"_operator.add", "operator.add"}
    commutative = associative and "list" not in type_name(annotation)
    identity_expr = "[]" if reducer_id in {"_operator.add", "operator.add"} and "list" in type_name(annotation) else None
    return ReducerRef(
        reducer_id="operator.add" if reducer_id in {"_operator.add", "operator.add"} else reducer_id,
        associative=associative,
        commutative=commutative,
        has_identity=identity_expr is not None,
        identity_expr=identity_expr,
        deterministic=True,
    )


def extract_retry_policy(policy: Any) -> RetryPolicyIR | None:
    if policy is None:
        return None
    policies = list(policy) if isinstance(policy, (list, tuple)) else [policy]
    rules: list[RetryRuleIR] = []
    for item in policies:
        if isinstance(item, dict):
            policy_dict = dict(item)
        else:
            policy_dict = item._asdict() if hasattr(item, "_asdict") else {}
        match = callable_ref(policy_dict.get("retry_on")) or "Exception"
        rules.append(
            RetryRuleIR(
                match=match,
                max_attempts=int(policy_dict.get("max_attempts", 1)),
                backoff_ms=int(float(policy_dict.get("initial_interval", 0.0)) * 1000),
            )
        )
    return RetryPolicyIR(rules=rules)


def extract_cache_policy(policy: Any) -> CachePolicyIR | None:
    if policy is None:
        return None
    if isinstance(policy, dict):
        return CachePolicyIR(
            ttl_seconds=policy.get("ttl"),
            key_func_ref=callable_ref(policy.get("key_func")) or policy.get("key_func_ref"),
        )
    return CachePolicyIR(
        ttl_seconds=getattr(policy, "ttl", None),
        key_func_ref=callable_ref(getattr(policy, "key_func", None)),
    )


def extract_side_effects(metadata: dict[str, Any]) -> SideEffectIR | None:
    value = metadata.get("side_effects")
    if not value:
        return None
    return SideEffectIR(
        purity=value.get("purity", "Pure"),
        effect_domains=list(value.get("effect_domains", [])),
        idempotency_key_strategy=value.get("idempotency_key_strategy"),
    )


def extract_resources(metadata: dict[str, Any]) -> ResourceIR | None:
    value = metadata.get("resources")
    if not value:
        return None
    return ResourceIR(
        cpu_class=value.get("cpu_class"),
        memory_mb=value.get("memory_mb"),
        timeout_sec=value.get("timeout_sec"),
        concurrency_limit=value.get("concurrency_limit"),
        batchable=value.get("batchable"),
    )


def extract_approval_policy(metadata: dict[str, Any]) -> ApprovalPolicyIR | None:
    value = metadata.get("approval_policy")
    if not value:
        return None
    return ApprovalPolicyIR(
        mode=value.get("mode", "auto"),
        interrupt_before=bool(value.get("interrupt_before", False)),
        interrupt_after=bool(value.get("interrupt_after", False)),
        required_scopes=normalize_string_sequence(value.get("required_scopes")),
        notes=normalize_string_sequence(value.get("notes")),
    )


def infer_branch_return_kind(branch_path: Any) -> str:
    source = callable_ref(branch_path) or ""
    if "fanout" in source.lower():
        return "SendList"
    target = unwrap_callable(branch_path)
    if target is None:
        return "NodeNames"
    try:
        import ast
        import inspect
        import textwrap

        parsed = ast.parse(textwrap.dedent(inspect.getsource(target)))
    except Exception:
        return "NodeNames"
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "Send":
                return "SendList"
            if isinstance(func, ast.Attribute) and func.attr == "Send":
                return "SendList"
    return "NodeNames"
