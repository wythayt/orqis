from __future__ import annotations

import importlib
import importlib.util
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from orqis.annotations import extract_orqis_payload, extract_registry_sections
from orqis.compiler.builder_common import (
    extract_approval_policy,
    extract_cache_policy,
    extract_resources,
    extract_retry_policy,
    extract_side_effects,
    normalize_string_sequence,
)
from orqis.compiler.ir import (
    AnalysisBundle,
    ApprovalPolicyIR,
    AssetRefIR,
    CacheAnalysisIR,
    CapabilityAnalysisIR,
    ExecutorClassIR,
    GraphIR0,
    MemoryPolicyIR,
    ResourceIR,
    SideEffectIR,
    SkillManifestIR,
    SubagentIR,
    ToolBindingIR,
    ToolIR,
)
from orqis.compiler.utils import callable_ref

DEFAULT_INLINE_BYTES = 16 * 1024

@dataclass(slots=True)
class ResolvedCapabilityView:
    # this is the canonical capability picture for one node or partition.
    subject_id: str
    subject_kind: str
    skill_ids: list[str] = field(default_factory=list)
    tool_binding_ids: list[str] = field(default_factory=list)
    tool_ids: list[str] = field(default_factory=list)
    asset_ids: list[str] = field(default_factory=list)
    externalized_assets: list[str] = field(default_factory=list)
    subagent_ids: list[str] = field(default_factory=list)
    executor_id: str | None = None
    memory_policy_ids: list[str] = field(default_factory=list)
    requires_interrupt: bool = False
    notes: list[str] = field(default_factory=list)


def enrich_graph_with_capabilities(
    graph_or_factory: Any,
    compiled_graph: CompiledStateGraph,
    lgir0: GraphIR0,
) -> tuple[list[str], list[str]]:
    # collect capability definitions from registries first, then let manifests override them.
    notes: list[str] = []
    warnings: list[str] = []
    module_name = resolve_source_module_name(graph_or_factory, compiled_graph)
    manifest_path = resolve_manifest_path(module_name)

    raw_sections: dict[str, dict[str, Any]] = {
        "assets": {},
        "memory_policies": {},
        "executors": {},
        "tools": {},
        "tool_bindings": {},
        "skills": {},
        "subagents": {},
        "node_overrides": {},
    }

    merge_raw_sections(raw_sections, extract_registry_sections(graph_or_factory))
    if extract_orqis_payload(graph_or_factory):
        notes.append("loaded capability annotations from graph factory")
    if module_name:
        module = importlib.import_module(module_name)
        module_sections = load_module_registries(module)
        merge_raw_sections(raw_sections, module_sections)
        if any(module_sections.values()):
            notes.append(f"loaded capability registries from module `{module_name}`")
    if manifest_path is not None:
        merge_raw_sections(raw_sections, load_manifest(manifest_path))
        notes.append(f"loaded capability manifest `{manifest_path}`")

    ensure_default_memory_policies(lgir0)
    ensure_default_executors(lgir0)
    apply_capability_sections(lgir0, raw_sections)
    apply_node_overrides(lgir0, raw_sections["node_overrides"])
    normalize_capabilities(lgir0)

    if module_name:
        lgir0.options["capability_module"] = module_name
    if manifest_path is not None:
        lgir0.options["capability_manifest"] = str(manifest_path)
    lgir0.options["capability_notes"] = notes
    return notes, warnings


def resolve_source_module_name(graph_or_factory: Any, compiled_graph: CompiledStateGraph) -> str | None:
    # prefer the graph factory module because it is the cleanest place to hang registries.
    module_name = getattr(graph_or_factory, "__module__", None)
    if module_name and module_name != "__main__":
        return module_name

    # fall back to the dominant node module when the caller passed a compiled graph directly.
    node_modules = []
    for spec in compiled_graph.builder.nodes.values():
        ref = callable_ref(spec.runnable)
        if not ref or "." not in ref:
            continue
        module = ref.rsplit(".", 1)[0]
        if module.startswith(("langgraph.", "langchain.", "orqis.compiler.")):
            continue
        node_modules.append(module)
    if not node_modules:
        return None
    return Counter(node_modules).most_common(1)[0][0]


def resolve_manifest_path(module_name: str | None) -> Path | None:
    if not module_name:
        return None
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return None
    module_path = Path(spec.origin)
    candidates = [
        module_path.with_suffix(".orqis.json"),
        module_path.with_name(f"{module_path.stem}.orqis.json"),
        module_path.parent / "orqis_capabilities.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"capability manifest `{path}` must contain a json object")
    return payload


def load_module_registries(module: Any) -> dict[str, Any]:
    sections = {
        "assets": getattr(module, "ORQIS_ASSETS", {}),
        "memory_policies": getattr(module, "ORQIS_MEMORY_POLICIES", {}),
        "executors": getattr(module, "ORQIS_EXECUTORS", {}),
        "tools": getattr(module, "ORQIS_TOOLS", {}),
        "tool_bindings": getattr(module, "ORQIS_TOOL_BINDINGS", {}),
        "skills": getattr(module, "ORQIS_SKILLS", {}),
        "subagents": getattr(module, "ORQIS_SUBAGENTS", {}),
        "node_overrides": getattr(module, "ORQIS_NODE_OVERRIDES", {}),
    }
    for name, value in vars(module).items():
        payload = extract_orqis_payload(value)
        if not payload:
            continue
        merge_raw_sections(sections, extract_registry_sections(value))
        if isinstance(payload.get("tool"), dict):
            tool_payload = dict(payload["tool"])
            tool_id = str(tool_payload.pop("tool_id", name))
            tool_payload.setdefault("callable_ref", callable_ref(value))
            sections["tools"][tool_id] = tool_payload
        if isinstance(payload.get("tool_binding"), dict):
            binding_payload = dict(payload["tool_binding"])
            binding_id = str(binding_payload.pop("binding_id", name))
            sections["tool_bindings"][binding_id] = binding_payload
        if isinstance(payload.get("skill"), dict):
            skill_payload = dict(payload["skill"])
            skill_id = str(skill_payload.pop("skill_id", name))
            sections["skills"][skill_id] = skill_payload
        if isinstance(payload.get("asset"), dict):
            asset_payload = dict(payload["asset"])
            asset_id = str(asset_payload.pop("asset_id", name))
            sections["assets"][asset_id] = asset_payload
        if isinstance(payload.get("executor"), dict):
            executor_payload = dict(payload["executor"])
            executor_id = str(executor_payload.pop("executor_id", name))
            sections["executors"][executor_id] = executor_payload
        if isinstance(payload.get("subagent"), dict):
            subagent_payload = dict(payload["subagent"])
            agent_id = str(subagent_payload.pop("agent_id", name))
            sections["subagents"][agent_id] = subagent_payload
        if isinstance(payload.get("memory_policy"), dict):
            policy_payload = dict(payload["memory_policy"])
            policy_id = str(policy_payload.pop("policy_id", name))
            sections["memory_policies"][policy_id] = policy_payload
    return sections


def merge_raw_sections(base: dict[str, dict[str, Any]], incoming: dict[str, Any]) -> None:
    for section, payload in incoming.items():
        if section not in base or not payload:
            continue
        if not isinstance(payload, dict):
            raise TypeError(f"capability section `{section}` must be a mapping")
        base[section].update(payload)


def ensure_default_memory_policies(lgir0: GraphIR0) -> None:
    if "thread_default" in lgir0.memory_policies:
        return
    lgir0.memory_policies["thread_default"] = MemoryPolicyIR(
        policy_id="thread_default",
        short_term_keys=["messages", "active_skills", "loaded_asset_refs", "tool_scope"],
        long_term_namespaces=["profiles", "preferences"],
        externalized_keys=["documents", "large_tool_outputs"],
        summarize_keys=["messages"],
        max_inline_bytes=DEFAULT_INLINE_BYTES,
        # keep live thread state small so checkpoints stay cheap and durable.
        notes=["default thread memory policy keeps large payloads out of checkpoints"],
    )


def ensure_default_executors(lgir0: GraphIR0) -> None:
    defaults = {
        "lambda_default": ExecutorClassIR(
            executor_id="lambda_default",
            backend="lambda",
            runtime="python3.11",
            packaging="zip",
            filesystem="tmp",
            network_access="egress",
            resource_profile=ResourceIR(memory_mb=512, timeout_sec=15),
            # this is the general-purpose backend for small stateless work.
            notes=["default backend for short stateless tasks"],
        ),
        "lambda_container_default": ExecutorClassIR(
            executor_id="lambda_container_default",
            backend="lambda_container",
            runtime="python3.11",
            packaging="container",
            filesystem="tmp",
            network_access="egress",
            resource_profile=ResourceIR(memory_mb=1024, timeout_sec=60),
            notes=["use when zip packaging is too small but lambda is still a good fit"],
        ),
        "fargate_default": ExecutorClassIR(
            executor_id="fargate_default",
            backend="fargate",
            runtime="python3.11",
            packaging="container",
            filesystem="ephemeral",
            network_access="vpc",
            long_running=True,
            supports_streaming=True,
            resource_profile=ResourceIR(memory_mb=4096, timeout_sec=900),
            notes=["use for heavyweight or script-driven tools"],
        ),
        "mcp_default": ExecutorClassIR(
            executor_id="mcp_default",
            backend="mcp",
            runtime="client",
            packaging="service",
            filesystem="none",
            network_access="internal",
            supports_streaming=True,
            notes=["adapter backend for model context protocol servers"],
        ),
    }
    for executor_id, executor in defaults.items():
        lgir0.executors.setdefault(executor_id, executor)


def apply_capability_sections(lgir0: GraphIR0, raw_sections: dict[str, dict[str, Any]]) -> None:
    for asset_id, payload in raw_sections["assets"].items():
        lgir0.assets[asset_id] = build_asset_ref(asset_id, payload)
    for policy_id, payload in raw_sections["memory_policies"].items():
        lgir0.memory_policies[policy_id] = build_memory_policy(policy_id, payload)
    for executor_id, payload in raw_sections["executors"].items():
        lgir0.executors[executor_id] = build_executor(executor_id, payload)
    for tool_id, payload in raw_sections["tools"].items():
        lgir0.tools[tool_id] = build_tool(tool_id, payload)
    for binding_id, payload in raw_sections["tool_bindings"].items():
        lgir0.tool_bindings[binding_id] = build_tool_binding(binding_id, payload)
    for skill_id, payload in raw_sections["skills"].items():
        lgir0.skills[skill_id] = build_skill(skill_id, payload)
    for agent_id, payload in raw_sections["subagents"].items():
        lgir0.subagents[agent_id] = build_subagent(agent_id, payload)


def apply_node_overrides(lgir0: GraphIR0, overrides: dict[str, Any]) -> None:
    for node_id, payload in overrides.items():
        if node_id not in lgir0.nodes:
            raise ValueError(f"unknown node override `{node_id}`")
        node = lgir0.nodes[node_id]
        if "skills" in payload:
            node.skill_ids = normalize_string_sequence(payload.get("skills"))
        if "tool_bindings" in payload:
            node.tool_binding_ids = normalize_string_sequence(payload.get("tool_bindings"))
        if "subagent" in payload:
            node.subagent_id = payload.get("subagent")
        if "executor" in payload:
            node.executor_id = payload.get("executor")
        if "approval_policy" in payload:
            node.approval_policy = extract_approval_policy({"approval_policy": payload.get("approval_policy")})
        if "resources" in payload:
            node.resources = extract_resources({"resources": payload.get("resources")})
        if "side_effects" in payload:
            node.side_effects = extract_side_effects({"side_effects": payload.get("side_effects")})


def build_asset_ref(asset_id: str, payload: dict[str, Any]) -> AssetRefIR:
    return AssetRefIR(
        asset_id=asset_id,
        version=payload.get("version"),
        kind=payload.get("kind", "text"),
        uri=payload.get("uri"),
        packaging=payload.get("packaging", "inline"),
        size_bytes=payload.get("size_bytes"),
        content_type=payload.get("content_type"),
        checksum=payload.get("checksum"),
        mutable=bool(payload.get("mutable", False)),
        load_strategy=payload.get("load_strategy", "lazy"),
        notes=normalize_string_sequence(payload.get("notes")),
    )


def build_memory_policy(policy_id: str, payload: dict[str, Any]) -> MemoryPolicyIR:
    return MemoryPolicyIR(
        policy_id=policy_id,
        short_term_keys=normalize_string_sequence(payload.get("short_term_keys")),
        long_term_namespaces=normalize_string_sequence(payload.get("long_term_namespaces")),
        externalized_keys=normalize_string_sequence(payload.get("externalized_keys")),
        summarize_keys=normalize_string_sequence(payload.get("summarize_keys")),
        max_inline_bytes=payload.get("max_inline_bytes"),
        notes=normalize_string_sequence(payload.get("notes")),
    )


def build_executor(executor_id: str, payload: dict[str, Any]) -> ExecutorClassIR:
    resource_profile = extract_resources({"resources": payload.get("resources")}) if payload.get("resources") else None
    return ExecutorClassIR(
        executor_id=executor_id,
        backend=payload.get("backend", "lambda"),
        runtime=payload.get("runtime"),
        packaging=payload.get("packaging"),
        filesystem=payload.get("filesystem"),
        network_access=payload.get("network_access", "none"),
        long_running=bool(payload.get("long_running", False)),
        supports_streaming=bool(payload.get("supports_streaming", False)),
        resource_profile=resource_profile,
        notes=normalize_string_sequence(payload.get("notes")),
    )


def build_tool(tool_id: str, payload: dict[str, Any]) -> ToolIR:
    return ToolIR(
        tool_id=tool_id,
        tool_kind=payload.get("tool_kind", "python"),
        callable_ref=payload.get("callable_ref"),
        description=payload.get("description", ""),
        args_schema=payload.get("args_schema"),
        return_schema=payload.get("return_schema"),
        side_effects=extract_side_effects({"side_effects": payload.get("side_effects")}) or SideEffectIR(purity="Pure"),
        resources=extract_resources({"resources": payload.get("resources")}),
        retry_policy=extract_retry_policy(payload.get("retry_policy")),
        cache_policy=extract_cache_policy(payload.get("cache_policy")),
        executor_id=payload.get("executor_id"),
        required_asset_ids=normalize_string_sequence(payload.get("required_asset_ids")),
        metadata=dict(payload.get("metadata", {})),
    )


def build_tool_binding(binding_id: str, payload: dict[str, Any]) -> ToolBindingIR:
    return ToolBindingIR(
        binding_id=binding_id,
        tool_id=payload.get("tool_id", binding_id),
        scope_kind=payload.get("scope_kind", "global"),
        scope_ref=payload.get("scope_ref"),
        visibility=payload.get("visibility", "allowed"),
        requires_skill_id=payload.get("requires_skill_id"),
        approval_policy=extract_approval_policy({"approval_policy": payload.get("approval_policy")}),
        argument_mapping={str(key): str(value) for key, value in dict(payload.get("argument_mapping", {})).items()},
        state_updates=normalize_string_sequence(payload.get("state_updates")),
        notes=normalize_string_sequence(payload.get("notes")),
    )


def build_skill(skill_id: str, payload: dict[str, Any]) -> SkillManifestIR:
    return SkillManifestIR(
        skill_id=skill_id,
        version=str(payload.get("version", "1")),
        description=payload.get("description", ""),
        prompt_asset_id=payload.get("prompt_asset_id"),
        asset_ids=normalize_string_sequence(payload.get("asset_ids")),
        tool_binding_ids=normalize_string_sequence(payload.get("tool_binding_ids")),
        subagent_ids=normalize_string_sequence(payload.get("subagent_ids")),
        state_schema=payload.get("state_schema"),
        memory_policy_id=payload.get("memory_policy_id"),
        load_strategy=payload.get("load_strategy", "on_demand"),
        inheritance=payload.get("inheritance", "explicit"),
        metadata=dict(payload.get("metadata", {})),
    )


def build_subagent(agent_id: str, payload: dict[str, Any]) -> SubagentIR:
    return SubagentIR(
        agent_id=agent_id,
        graph_ref=payload.get("graph_ref"),
        callable_ref=payload.get("callable_ref"),
        system_prompt_asset_id=payload.get("system_prompt_asset_id"),
        state_schema=payload.get("state_schema"),
        skill_ids=normalize_string_sequence(payload.get("skill_ids")),
        tool_binding_ids=normalize_string_sequence(payload.get("tool_binding_ids")),
        handoff_targets=normalize_string_sequence(payload.get("handoff_targets")),
        store_namespaces=normalize_string_sequence(payload.get("store_namespaces")),
        executor_id=payload.get("executor_id"),
        memory_policy_id=payload.get("memory_policy_id"),
        inheritance=payload.get("inheritance", "isolated"),
        metadata=dict(payload.get("metadata", {})),
    )


def normalize_capabilities(lgir0: GraphIR0) -> None:
    # resolve defaults first so later analysis can assume stable references.
    for tool in lgir0.tools.values():
        if tool.executor_id is None:
            tool.executor_id = default_executor_id_for_tool(tool)
        if tool.side_effects is None:
            tool.side_effects = SideEffectIR(purity="Pure", effect_domains=[])

    for subagent in lgir0.subagents.values():
        if subagent.executor_id is None:
            subagent.executor_id = infer_subagent_executor(lgir0, subagent)
        if subagent.memory_policy_id is None:
            subagent.memory_policy_id = "thread_default"

    for skill in lgir0.skills.values():
        if skill.memory_policy_id is None:
            skill.memory_policy_id = "thread_default"

    for node in lgir0.nodes.values():
        if node.executor_id is None:
            node.executor_id = infer_node_executor(lgir0, node.node_id)

    validate_capabilities(lgir0)


def default_executor_id_for_tool(tool: ToolIR) -> str:
    if tool.tool_kind == "mcp":
        return "mcp_default"
    if tool.tool_kind in {"sandbox", "script", "browser"}:
        return "fargate_default"
    if tool.resources and tool.resources.memory_mb and tool.resources.memory_mb > 3008:
        return "fargate_default"
    if tool.tool_kind in {"container", "batch"}:
        return "lambda_container_default"
    return "lambda_default"


def infer_subagent_executor(lgir0: GraphIR0, subagent: SubagentIR) -> str:
    tool_executors = {
        lgir0.tools[binding.tool_id].executor_id
        for binding_id in effective_subagent_tool_binding_ids(lgir0, subagent)
        for binding in [lgir0.tool_bindings[binding_id]]
        if binding.tool_id in lgir0.tools and lgir0.tools[binding.tool_id].executor_id
    }
    if len(tool_executors) == 1:
        return next(iter(tool_executors))
    if "fargate_default" in tool_executors:
        return "fargate_default"
    if "lambda_container_default" in tool_executors:
        return "lambda_container_default"
    if "mcp_default" in tool_executors:
        return "mcp_default"
    return "lambda_default"


def infer_node_executor(lgir0: GraphIR0, node_id: str) -> str:
    view = resolve_node_capability_view(lgir0, node_id)
    if view.executor_id:
        return view.executor_id
    return "lambda_default"


def validate_capabilities(lgir0: GraphIR0) -> None:
    for asset_id in lgir0.assets:
        if not asset_id:
            raise ValueError("asset ids must be non-empty")

    for policy_id in lgir0.memory_policies:
        if not policy_id:
            raise ValueError("memory policy ids must be non-empty")

    for executor_id in lgir0.executors:
        if not executor_id:
            raise ValueError("executor ids must be non-empty")

    for tool in lgir0.tools.values():
        if tool.executor_id not in lgir0.executors:
            raise ValueError(f"tool `{tool.tool_id}` references unknown executor `{tool.executor_id}`")
        for asset_id in tool.required_asset_ids:
            if asset_id not in lgir0.assets:
                raise ValueError(f"tool `{tool.tool_id}` references unknown asset `{asset_id}`")

    for binding in lgir0.tool_bindings.values():
        if binding.tool_id not in lgir0.tools:
            raise ValueError(f"tool binding `{binding.binding_id}` references unknown tool `{binding.tool_id}`")
        if binding.requires_skill_id and binding.requires_skill_id not in lgir0.skills:
            raise ValueError(
                f"tool binding `{binding.binding_id}` references unknown required skill `{binding.requires_skill_id}`"
            )
        if binding.scope_kind == "node" and binding.scope_ref and binding.scope_ref not in lgir0.nodes:
            raise ValueError(f"tool binding `{binding.binding_id}` references unknown node `{binding.scope_ref}`")
        if binding.scope_kind == "agent" and binding.scope_ref and binding.scope_ref not in lgir0.subagents:
            raise ValueError(f"tool binding `{binding.binding_id}` references unknown subagent `{binding.scope_ref}`")
        if binding.scope_kind == "skill" and binding.scope_ref and binding.scope_ref not in lgir0.skills:
            raise ValueError(f"tool binding `{binding.binding_id}` references unknown skill `{binding.scope_ref}`")

    for skill in lgir0.skills.values():
        if skill.prompt_asset_id and skill.prompt_asset_id not in lgir0.assets:
            raise ValueError(f"skill `{skill.skill_id}` references unknown prompt asset `{skill.prompt_asset_id}`")
        for asset_id in skill.asset_ids:
            if asset_id not in lgir0.assets:
                raise ValueError(f"skill `{skill.skill_id}` references unknown asset `{asset_id}`")
        for binding_id in skill.tool_binding_ids:
            if binding_id not in lgir0.tool_bindings:
                raise ValueError(f"skill `{skill.skill_id}` references unknown binding `{binding_id}`")
        for agent_id in skill.subagent_ids:
            if agent_id not in lgir0.subagents:
                raise ValueError(f"skill `{skill.skill_id}` references unknown subagent `{agent_id}`")
        if skill.memory_policy_id and skill.memory_policy_id not in lgir0.memory_policies:
            raise ValueError(f"skill `{skill.skill_id}` references unknown memory policy `{skill.memory_policy_id}`")

    for subagent in lgir0.subagents.values():
        if subagent.system_prompt_asset_id and subagent.system_prompt_asset_id not in lgir0.assets:
            raise ValueError(
                f"subagent `{subagent.agent_id}` references unknown prompt asset `{subagent.system_prompt_asset_id}`"
            )
        for skill_id in subagent.skill_ids:
            if skill_id not in lgir0.skills:
                raise ValueError(f"subagent `{subagent.agent_id}` references unknown skill `{skill_id}`")
        for binding_id in subagent.tool_binding_ids:
            if binding_id not in lgir0.tool_bindings:
                raise ValueError(f"subagent `{subagent.agent_id}` references unknown binding `{binding_id}`")
        if subagent.executor_id and subagent.executor_id not in lgir0.executors:
            raise ValueError(f"subagent `{subagent.agent_id}` references unknown executor `{subagent.executor_id}`")
        if subagent.memory_policy_id and subagent.memory_policy_id not in lgir0.memory_policies:
            raise ValueError(
                f"subagent `{subagent.agent_id}` references unknown memory policy `{subagent.memory_policy_id}`"
            )

    for node in lgir0.nodes.values():
        for skill_id in node.skill_ids:
            if skill_id not in lgir0.skills:
                raise ValueError(f"node `{node.node_id}` references unknown skill `{skill_id}`")
        for binding_id in node.tool_binding_ids:
            if binding_id not in lgir0.tool_bindings:
                raise ValueError(f"node `{node.node_id}` references unknown tool binding `{binding_id}`")
        if node.subagent_id and node.subagent_id not in lgir0.subagents:
            raise ValueError(f"node `{node.node_id}` references unknown subagent `{node.subagent_id}`")
        if node.executor_id and node.executor_id not in lgir0.executors:
            raise ValueError(f"node `{node.node_id}` references unknown executor `{node.executor_id}`")


def build_capability_analysis(lgir0: GraphIR0) -> tuple[list[CapabilityAnalysisIR], list[str], list[str]]:
    items: list[CapabilityAnalysisIR] = []
    notes = ["capability analysis resolves skills, tools, assets, and executors before partitioning"]
    warnings: list[str] = []

    for node_id in sorted(lgir0.nodes):
        view = resolve_node_capability_view(lgir0, node_id)
        items.append(
            CapabilityAnalysisIR(
                subject_id=node_id,
                subject_kind="node",
                executor_id=view.executor_id,
                externalized_assets=view.externalized_assets,
                loaded_skills=view.skill_ids,
                reachable_tools=view.tool_ids,
                requires_interrupt=view.requires_interrupt,
                notes=view.notes,
            )
        )
        if view.subagent_ids and (lgir0.nodes[node_id].tool_binding_ids or lgir0.nodes[node_id].skill_ids):
            warnings.append(f"node `{node_id}` combines local skill or tool scope with subagent delegation")

    for agent_id in sorted(lgir0.subagents):
        view = resolve_subagent_capability_view(lgir0, agent_id)
        items.append(
            CapabilityAnalysisIR(
                subject_id=agent_id,
                subject_kind="subagent",
                executor_id=view.executor_id,
                externalized_assets=view.externalized_assets,
                loaded_skills=view.skill_ids,
                reachable_tools=view.tool_ids,
                requires_interrupt=view.requires_interrupt,
                notes=view.notes,
            )
        )
    return items, notes, warnings


def resolve_node_capability_view(lgir0: GraphIR0, node_id: str) -> ResolvedCapabilityView:
    node = lgir0.nodes[node_id]
    skill_ids = list(node.skill_ids)
    binding_ids = set(applicable_bindings_for_node(lgir0, node_id, skill_ids))
    binding_ids.update(node.tool_binding_ids)
    subagent_ids: list[str] = []
    memory_policy_ids: set[str] = set()
    notes: list[str] = []

    for skill_id in skill_ids:
        skill = lgir0.skills[skill_id]
        binding_ids.update(skill.tool_binding_ids)
        if skill.memory_policy_id:
            memory_policy_ids.add(skill.memory_policy_id)

    if node.subagent_id:
        subagent_ids.append(node.subagent_id)
        subagent = lgir0.subagents[node.subagent_id]
        skill_ids = sorted(set(skill_ids) | set(subagent.skill_ids))
        binding_ids.update(effective_subagent_tool_binding_ids(lgir0, subagent))
        if subagent.memory_policy_id:
            memory_policy_ids.add(subagent.memory_policy_id)
        notes.append(f"delegates to subagent `{subagent.agent_id}`")

    binding_ids = {
        binding_id
        for binding_id in binding_ids
        if binding_visible_with_skills(lgir0, binding_id, skill_ids)
    }
    tool_ids = sorted({lgir0.tool_bindings[binding_id].tool_id for binding_id in binding_ids})
    asset_ids = collect_asset_ids_for_subject(lgir0, skill_ids, tool_ids, subagent_ids)
    externalized_assets = [
        asset_id
        for asset_id in asset_ids
        if should_externalize_asset(lgir0, asset_id, memory_policy_ids)
    ]
    executor_id = node.executor_id
    if executor_id is None and node.subagent_id:
        executor_id = lgir0.subagents[node.subagent_id].executor_id
    if executor_id is None:
        executor_ids = {
            lgir0.tools[tool_id].executor_id
            for tool_id in tool_ids
            if tool_id in lgir0.tools and lgir0.tools[tool_id].executor_id
        }
        if len(executor_ids) == 1:
            executor_id = next(iter(executor_ids))
        elif len(executor_ids) > 1:
            executor_id = "fargate_default"
            notes.append("multiple tool executors collapse to a heavyweight runtime boundary")
    requires_interrupt = approval_requires_interrupt(node.approval_policy) or any(
        approval_requires_interrupt(lgir0.tool_bindings[binding_id].approval_policy)
        for binding_id in binding_ids
    )
    if requires_interrupt:
        notes.append("approval policy introduces an interrupt boundary")
    if externalized_assets:
        notes.append("large or mounted assets stay out of checkpoints")
    return ResolvedCapabilityView(
        subject_id=node_id,
        subject_kind="node",
        skill_ids=sorted(set(skill_ids)),
        tool_binding_ids=sorted(binding_ids),
        tool_ids=tool_ids,
        asset_ids=asset_ids,
        externalized_assets=externalized_assets,
        subagent_ids=subagent_ids,
        executor_id=executor_id or "lambda_default",
        memory_policy_ids=sorted(memory_policy_ids or {"thread_default"}),
        requires_interrupt=requires_interrupt,
        notes=notes,
    )


def resolve_subagent_capability_view(lgir0: GraphIR0, agent_id: str) -> ResolvedCapabilityView:
    subagent = lgir0.subagents[agent_id]
    binding_ids = effective_subagent_tool_binding_ids(lgir0, subagent)
    tool_ids = sorted({lgir0.tool_bindings[binding_id].tool_id for binding_id in binding_ids})
    memory_policy_ids = [subagent.memory_policy_id or "thread_default"]
    asset_ids = collect_asset_ids_for_subject(lgir0, subagent.skill_ids, tool_ids, [agent_id])
    externalized_assets = [
        asset_id
        for asset_id in asset_ids
        if should_externalize_asset(lgir0, asset_id, set(memory_policy_ids))
    ]
    requires_interrupt = any(
        approval_requires_interrupt(lgir0.tool_bindings[binding_id].approval_policy)
        for binding_id in binding_ids
    )
    notes = []
    if subagent.inheritance != "isolated":
        notes.append(f"subagent inheritance mode is `{subagent.inheritance}`")
    if externalized_assets:
        notes.append("subagent prompt assets are loaded by reference")
    return ResolvedCapabilityView(
        subject_id=agent_id,
        subject_kind="subagent",
        skill_ids=sorted(set(subagent.skill_ids)),
        tool_binding_ids=sorted(binding_ids),
        tool_ids=tool_ids,
        asset_ids=asset_ids,
        externalized_assets=externalized_assets,
        subagent_ids=[agent_id],
        executor_id=subagent.executor_id or "lambda_default",
        memory_policy_ids=memory_policy_ids,
        requires_interrupt=requires_interrupt,
        notes=notes,
    )


def effective_subagent_tool_binding_ids(lgir0: GraphIR0, subagent: SubagentIR) -> list[str]:
    binding_ids = set(applicable_bindings_for_agent(lgir0, subagent.agent_id, subagent.skill_ids))
    binding_ids.update(subagent.tool_binding_ids)
    for skill_id in subagent.skill_ids:
        binding_ids.update(lgir0.skills[skill_id].tool_binding_ids)
    return sorted(
        binding_id
        for binding_id in binding_ids
        if binding_visible_with_skills(lgir0, binding_id, subagent.skill_ids)
    )


def collect_asset_ids_for_subject(
    lgir0: GraphIR0,
    skill_ids: list[str],
    tool_ids: list[str],
    subagent_ids: list[str],
) -> list[str]:
    asset_ids: set[str] = set()
    for skill_id in skill_ids:
        skill = lgir0.skills[skill_id]
        asset_ids.update(skill.asset_ids)
        if skill.prompt_asset_id:
            asset_ids.add(skill.prompt_asset_id)
    for tool_id in tool_ids:
        asset_ids.update(lgir0.tools[tool_id].required_asset_ids)
    for agent_id in subagent_ids:
        subagent = lgir0.subagents[agent_id]
        if subagent.system_prompt_asset_id:
            asset_ids.add(subagent.system_prompt_asset_id)
    return sorted(asset_ids)


def applicable_bindings_for_node(lgir0: GraphIR0, node_id: str, skill_ids: list[str]) -> list[str]:
    binding_ids = []
    for binding_id, binding in lgir0.tool_bindings.items():
        if binding.scope_kind == "global":
            binding_ids.append(binding_id)
        elif binding.scope_kind == "node" and binding.scope_ref == node_id:
            binding_ids.append(binding_id)
        elif binding.scope_kind == "skill" and binding.scope_ref in skill_ids:
            binding_ids.append(binding_id)
    return sorted(binding_ids)


def applicable_bindings_for_agent(lgir0: GraphIR0, agent_id: str, skill_ids: list[str]) -> list[str]:
    binding_ids = []
    for binding_id, binding in lgir0.tool_bindings.items():
        if binding.scope_kind == "global":
            binding_ids.append(binding_id)
        elif binding.scope_kind == "agent" and binding.scope_ref == agent_id:
            binding_ids.append(binding_id)
        elif binding.scope_kind == "skill" and binding.scope_ref in skill_ids:
            binding_ids.append(binding_id)
    return sorted(binding_ids)


def binding_visible_with_skills(lgir0: GraphIR0, binding_id: str, skill_ids: list[str]) -> bool:
    binding = lgir0.tool_bindings[binding_id]
    if binding.requires_skill_id and binding.requires_skill_id not in skill_ids:
        return False
    return True


def should_externalize_asset(lgir0: GraphIR0, asset_id: str, memory_policy_ids: set[str]) -> bool:
    asset = lgir0.assets[asset_id]
    if asset.packaging in {"s3", "efs", "blob"}:
        return True
    max_inline = min(
        (
            lgir0.memory_policies[policy_id].max_inline_bytes
            for policy_id in memory_policy_ids
            if policy_id in lgir0.memory_policies and lgir0.memory_policies[policy_id].max_inline_bytes is not None
        ),
        default=DEFAULT_INLINE_BYTES,
    )
    if asset.size_bytes is not None and asset.size_bytes > max_inline:
        return True
    return False


def approval_requires_interrupt(policy: ApprovalPolicyIR | None) -> bool:
    if policy is None:
        return False
    return policy.interrupt_before or policy.interrupt_after or policy.mode in {"user_approval", "interrupt"}


def partition_capability_view(lgir0: GraphIR0, members: list[str]) -> ResolvedCapabilityView:
    # union the per-node views so partitioning and planning share one resolution model.
    node_views = [resolve_node_capability_view(lgir0, member) for member in members]
    executor_ids = {view.executor_id for view in node_views if view.executor_id}
    notes = []
    if len(executor_ids) > 1:
        notes.append("partition spans multiple executors and should have been split earlier")
    return ResolvedCapabilityView(
        subject_id=",".join(members),
        subject_kind="partition",
        skill_ids=sorted({skill_id for view in node_views for skill_id in view.skill_ids}),
        tool_binding_ids=sorted({binding_id for view in node_views for binding_id in view.tool_binding_ids}),
        tool_ids=sorted({tool_id for view in node_views for tool_id in view.tool_ids}),
        asset_ids=sorted({asset_id for view in node_views for asset_id in view.asset_ids}),
        externalized_assets=sorted({asset_id for view in node_views for asset_id in view.externalized_assets}),
        subagent_ids=sorted({agent_id for view in node_views for agent_id in view.subagent_ids}),
        executor_id=next(iter(executor_ids)) if len(executor_ids) == 1 else None,
        memory_policy_ids=sorted({policy_id for view in node_views for policy_id in view.memory_policy_ids}),
        requires_interrupt=any(view.requires_interrupt for view in node_views),
        notes=notes,
    )


def executor_resource_profile(lgir0: GraphIR0, executor_id: str | None) -> ResourceIR | None:
    if executor_id is None:
        return None
    executor = lgir0.executors.get(executor_id)
    if executor is None:
        return None
    return executor.resource_profile


def apply_capability_analysis(analysis: AnalysisBundle, lgir0: GraphIR0) -> None:
    items, notes, warnings = build_capability_analysis(lgir0)
    analysis.capability_analysis.extend(items)
    analysis.notes.extend(notes)
    analysis.warnings.extend(warnings)


def capability_cache_hint(lgir0: GraphIR0, node_id: str, cache: CacheAnalysisIR) -> CacheAnalysisIR:
    view = resolve_node_capability_view(lgir0, node_id)
    if not view.externalized_assets:
        return cache
    # externalized assets are typically safe to cache because the checkpoint only stores refs.
    return CacheAnalysisIR(
        node_id=cache.node_id,
        has_cache_policy=cache.has_cache_policy,
        safe_to_cache=cache.safe_to_cache,
        recommended_boundary=cache.recommended_boundary or cache.safe_to_cache,
        reason=f"{cache.reason}; asset refs stay out of checkpoint payloads",
    )
