from __future__ import annotations

import operator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import Annotated, TypedDict

from orqis import orqis
from orqis.examples.final_experiments.bedrock import invoke_json_or_fallback, invoke_text_or_fallback


class IncidentResponseState(TypedDict):
    incident_id: str
    severity: str
    alert_summary: str
    evidence_scope: str
    investigation_objective: str
    findings: Annotated[list[str], operator.add]
    containment_actions: Annotated[list[str], operator.add]
    communication_drafts: Annotated[list[str], operator.add]
    executive_summary: str
    final_recommendation: str


class IngestInput(TypedDict):
    severity: str
    alert_summary: str


class PlanInput(TypedDict):
    incident_id: str
    severity: str
    alert_summary: str
    evidence_scope: str


class SubagentTaskInput(TypedDict):
    incident_id: str
    severity: str
    task: str
    evidence_scope: str


class SynthesisInput(TypedDict):
    findings: list[str]
    containment_actions: list[str]
    communication_drafts: list[str]


class FinalizeInput(TypedDict):
    executive_summary: str
    final_recommendation: str
    communication_drafts: list[str]


IDENTITY_SUBAGENT_PROMPT = """# Identity Forensics Subagent

- Investigate sign-in anomalies, MFA resets, and suspicious privilege changes.
- Recommend containment only when the evidence justifies it.
- Keep the coordinator payload compact and reference-heavy.
"""

NETWORK_SUBAGENT_PROMPT = """# Network Forensics Subagent

- Review VPN, proxy, and endpoint telemetry for suspicious lateral movement.
- Prefer structured findings with explicit time windows and host scope.
- Escalate to sandbox analysis when payload triage is required.
"""

COMMS_SUBAGENT_PROMPT = """# Customer Communications Subagent

- Draft customer-facing status updates that are accurate and minimally speculative.
- Align the update with the current evidence and next action owner.
- Keep wording suitable for incident response stakeholders.
"""

NETWORK_SANDBOX_BUNDLE = """# Sandbox Bundle Manifest

- unpack packet decoders
- stage malware triage rules
- export normalized findings
"""


def _subagent_schema() -> str:
    return (
        '{'
        '"findings": ["string"], '
        '"containment_actions": ["string"], '
        '"communication_drafts": ["string"]'
        '}'
    )


def _summary_schema() -> str:
    return (
        '{'
        '"executive_summary": "string", '
        '"final_recommendation": "string"'
        '}'
    )


def _normalize_list(value: object, fallback: list[str]) -> list[str]:
    return value if isinstance(value, list) and value else fallback


@orqis(
    tool={
        "tool_id": "query_identity_audit_tool",
        "tool_kind": "mcp",
        "description": "retrieve sign-in and MFA evidence for suspicious identity activity",
        "side_effects": {
            "purity": "Idempotent",
            "effect_domains": ["db"],
            "idempotency_key_strategy": "task_id",
        },
        "resources": {"memory_mb": 1024, "timeout_sec": 30},
    }
)
def query_identity_audit(incident_id: str, evidence_scope: str) -> str:
    return invoke_text_or_fallback(
        system_prompt="You summarize identity-audit evidence for an incident-response specialist.",
        user_prompt=(
            f"Incident id: {incident_id}\n"
            f"Evidence scope: {evidence_scope}\n\n"
            "Return a concise identity evidence brief with suspicious sign-in, MFA, and privilege details."
        ),
        fallback=f"identity audit for {incident_id}: {evidence_scope}",
        max_tokens=180,
    )


@orqis(
    tool={
        "tool_id": "disable_identity_session_tool",
        "tool_kind": "python",
        "description": "disable suspicious identity sessions once the coordinator approves containment",
        "side_effects": {
            "purity": "Idempotent",
            "effect_domains": ["identity"],
            "idempotency_key_strategy": "task_id",
        },
        "resources": {"memory_mb": 512, "timeout_sec": 20},
    }
)
def disable_identity_session(incident_id: str, principal: str) -> str:
    return f"disable session for {principal} in {incident_id}"


@orqis(
    tool={
        "tool_id": "query_network_timeline_tool",
        "tool_kind": "container",
        "description": "assemble network and endpoint timeline slices around the alert",
        "side_effects": {
            "purity": "Idempotent",
            "effect_domains": ["network"],
            "idempotency_key_strategy": "task_id",
        },
        "resources": {"memory_mb": 2048, "timeout_sec": 90},
    }
)
def query_network_timeline(incident_id: str, evidence_scope: str) -> str:
    return invoke_text_or_fallback(
        system_prompt="You summarize network-timeline evidence for an incident-response specialist.",
        user_prompt=(
            f"Incident id: {incident_id}\n"
            f"Evidence scope: {evidence_scope}\n\n"
            "Return a concise network brief covering VPN, proxy, and endpoint observations."
        ),
        fallback=f"network timeline for {incident_id}: {evidence_scope}",
        max_tokens=180,
    )


@orqis(
    tool={
        "tool_id": "run_network_sandbox_tool",
        "tool_kind": "script",
        "description": "run heavyweight payload triage and normalization for suspected network-delivered artifacts",
        "side_effects": {
            "purity": "Idempotent",
            "effect_domains": ["filesystem", "network"],
            "idempotency_key_strategy": "task_id",
        },
        "required_asset_ids": ["network_sandbox_bundle"],
        "resources": {"memory_mb": 4096, "timeout_sec": 300},
    }
)
def run_network_sandbox(incident_id: str, evidence_scope: str) -> str:
    return invoke_text_or_fallback(
        system_prompt="You summarize sandbox-triage findings for an incident-response specialist.",
        user_prompt=(
            f"Incident id: {incident_id}\n"
            f"Evidence scope: {evidence_scope}\n\n"
            f"Sandbox bundle:\n{NETWORK_SANDBOX_BUNDLE}\n\n"
            "Return a concise sandbox brief that states whether further payload triage is warranted."
        ),
        fallback=f"sandbox bundle run for {incident_id}: {evidence_scope}",
        max_tokens=180,
    )


@orqis(
    tool={
        "tool_id": "draft_customer_update_tool",
        "tool_kind": "python",
        "description": "draft a precise customer-facing incident status update",
        "side_effects": {
            "purity": "Effectful",
            "effect_domains": ["llm"],
            "idempotency_key_strategy": "task_id",
        },
        "resources": {"memory_mb": 512, "timeout_sec": 15},
    }
)
def draft_customer_update(incident_id: str, executive_summary: str) -> str:
    return invoke_text_or_fallback(
        system_prompt="You draft customer-facing incident updates.",
        user_prompt=(
            f"Incident id: {incident_id}\n"
            f"Executive summary: {executive_summary}\n\n"
            "Return one short customer-safe update."
        ),
        fallback=f"draft update for {incident_id}: {executive_summary[:48]}",
        max_tokens=140,
    )


def ingest_alert(state: IngestInput) -> dict[str, object]:
    normalized = " ".join(state["alert_summary"].split())
    if state["severity"] in {"critical", "high"}:
        scope = "collect 24h of identity, VPN, proxy, and endpoint evidence"
    else:
        scope = "collect 8h of identity and network evidence"
    return {
        "alert_summary": normalized,
        "evidence_scope": scope,
    }


def plan_response(state: PlanInput) -> dict[str, object]:
    objective = (
        f"Contain and explain the {state['severity']} incident for {state['incident_id']} "
        f"using evidence scope: {state['evidence_scope']}"
    )
    return {"investigation_objective": objective}


def fanout_subagents(state: IncidentResponseState) -> list[Send]:
    return [
        Send(
            "identity_subagent",
            {
                "incident_id": state["incident_id"],
                "severity": state["severity"],
                "task": "Review identity activity, MFA resets, and suspicious session behavior.",
                "evidence_scope": state["evidence_scope"],
            },
        ),
        Send(
            "network_subagent",
            {
                "incident_id": state["incident_id"],
                "severity": state["severity"],
                "task": "Review VPN, proxy, and endpoint telemetry for coordinated host activity.",
                "evidence_scope": state["evidence_scope"],
            },
        ),
        Send(
            "communications_subagent",
            {
                "incident_id": state["incident_id"],
                "severity": state["severity"],
                "task": "Prepare a stakeholder-safe update that matches the current evidence.",
                "evidence_scope": state["evidence_scope"],
            },
        ),
    ]


@orqis(
    subagent="identity_forensics",
    side_effects={
        "purity": "Effectful",
        "effect_domains": ["llm"],
        "idempotency_key_strategy": "task_id",
    },
)
def identity_subagent(state: SubagentTaskInput) -> dict[str, object]:
    identity_context = query_identity_audit(state["incident_id"], state["evidence_scope"])
    containment_hint = (
        disable_identity_session(state["incident_id"], "affected-admin-principal")
        if state["severity"] in {"critical", "high"}
        else ""
    )
    fallback = {
        "findings": [
            "identity: multiple MFA resets were followed by impossible-travel sign-in attempts",
            f"identity scope: {state['evidence_scope']}",
        ],
        "containment_actions": (
            ["disable suspicious Okta sessions for the affected admin principal"]
            if state["severity"] in {"critical", "high"}
            else []
        ),
        "communication_drafts": [],
    }
    payload = invoke_json_or_fallback(
        system_prompt="You are the identity-forensics subagent in an incident-response workflow.",
        user_prompt=(
            f"Prompt:\n{IDENTITY_SUBAGENT_PROMPT}\n\n"
            f"Incident id: {state['incident_id']}\n"
            f"Severity: {state['severity']}\n"
            f"Task: {state['task']}\n"
            f"Evidence scope: {state['evidence_scope']}\n"
            f"Identity context: {identity_context}\n"
            f"Containment command if justified: {containment_hint}\n\n"
            "Return the delegated subagent result."
        ),
        schema_description=_subagent_schema(),
        fallback=fallback,
        max_tokens=420,
    )
    return {
        "findings": _normalize_list(payload.get("findings"), fallback["findings"]),
        "containment_actions": _normalize_list(payload.get("containment_actions"), fallback["containment_actions"]),
        "communication_drafts": _normalize_list(payload.get("communication_drafts"), fallback["communication_drafts"]),
    }


@orqis(
    subagent="network_forensics",
    side_effects={
        "purity": "Effectful",
        "effect_domains": ["llm"],
        "idempotency_key_strategy": "task_id",
    },
)
def network_subagent(state: SubagentTaskInput) -> dict[str, object]:
    network_context = query_network_timeline(state["incident_id"], state["evidence_scope"])
    sandbox_context = run_network_sandbox(state["incident_id"], state["evidence_scope"])
    fallback = {
        "findings": [
            "network: outbound VPN activity expanded to an unrecognized host group after the alert",
            "network: endpoint telemetry suggests the payload should be sandboxed before customer-facing remediation",
        ],
        "containment_actions": ["quarantine the suspicious host group from the VPN segment"],
        "communication_drafts": [],
    }
    payload = invoke_json_or_fallback(
        system_prompt="You are the network-forensics subagent in an incident-response workflow.",
        user_prompt=(
            f"Prompt:\n{NETWORK_SUBAGENT_PROMPT}\n\n"
            f"Incident id: {state['incident_id']}\n"
            f"Severity: {state['severity']}\n"
            f"Task: {state['task']}\n"
            f"Evidence scope: {state['evidence_scope']}\n"
            f"Network context: {network_context}\n"
            f"Sandbox context: {sandbox_context}\n\n"
            "Return the delegated subagent result."
        ),
        schema_description=_subagent_schema(),
        fallback=fallback,
        max_tokens=420,
    )
    return {
        "findings": _normalize_list(payload.get("findings"), fallback["findings"]),
        "containment_actions": _normalize_list(payload.get("containment_actions"), fallback["containment_actions"]),
        "communication_drafts": _normalize_list(payload.get("communication_drafts"), fallback["communication_drafts"]),
    }


@orqis(
    subagent="customer_communications",
    side_effects={
        "purity": "Effectful",
        "effect_domains": ["llm"],
        "idempotency_key_strategy": "task_id",
    },
)
def communications_subagent(state: SubagentTaskInput) -> dict[str, object]:
    draft = draft_customer_update(
        state["incident_id"],
        f"{state['task']} | scope: {state['evidence_scope']}",
    )
    fallback = {
        "findings": ["comms: external update should avoid host-level attribution until evidence is confirmed"],
        "containment_actions": [],
        "communication_drafts": [draft],
    }
    payload = invoke_json_or_fallback(
        system_prompt="You are the customer-communications subagent in an incident-response workflow.",
        user_prompt=(
            f"Prompt:\n{COMMS_SUBAGENT_PROMPT}\n\n"
            f"Incident id: {state['incident_id']}\n"
            f"Severity: {state['severity']}\n"
            f"Task: {state['task']}\n"
            f"Evidence scope: {state['evidence_scope']}\n"
            f"Draft candidate: {draft}\n\n"
            "Return the delegated subagent result."
        ),
        schema_description=_subagent_schema(),
        fallback=fallback,
        max_tokens=320,
    )
    return {
        "findings": _normalize_list(payload.get("findings"), fallback["findings"]),
        "containment_actions": _normalize_list(payload.get("containment_actions"), fallback["containment_actions"]),
        "communication_drafts": _normalize_list(payload.get("communication_drafts"), fallback["communication_drafts"]),
    }


@orqis(
    side_effects={
        "purity": "Effectful",
        "effect_domains": ["llm"],
        "idempotency_key_strategy": "task_id",
    }
)
def synthesize_recommendation(state: SynthesisInput) -> dict[str, object]:
    fallback = {
        "executive_summary": " | ".join(state["findings"]),
        "final_recommendation": (
            f"Containment: {'; '.join(state['containment_actions'])}. "
            f"Communications prepared: {len(state['communication_drafts'])} draft(s)."
        ),
    }
    payload = invoke_json_or_fallback(
        system_prompt="You synthesize delegated incident-response findings into an executive summary.",
        user_prompt=(
            f"Findings: {state['findings']}\n"
            f"Containment actions: {state['containment_actions']}\n"
            f"Communication drafts: {state['communication_drafts']}\n\n"
            "Return the synthesis result."
        ),
        schema_description=_summary_schema(),
        fallback=fallback,
        max_tokens=320,
    )
    return {
        "executive_summary": str(payload.get("executive_summary") or fallback["executive_summary"]),
        "final_recommendation": str(payload.get("final_recommendation") or fallback["final_recommendation"]),
    }


def finalize_incident(state: FinalizeInput) -> dict[str, object]:
    if state["communication_drafts"]:
        final_recommendation = (
            f"{state['final_recommendation']} First update: {state['communication_drafts'][0]}"
        )
    else:
        final_recommendation = state["final_recommendation"]
    return {"final_recommendation": f"{state['executive_summary']} || {final_recommendation}"}


@orqis(
    assets={
        "identity_subagent_prompt": {
            "version": "2026-07-01",
            "kind": "prompt",
            "packaging": "s3",
            "uri": "s3://orqis-assets/subagents/identity_forensics.md",
            "size_bytes": 24000,
            "content_type": "text/markdown",
            "load_strategy": "lazy",
        },
        "network_subagent_prompt": {
            "version": "2026-07-01",
            "kind": "prompt",
            "packaging": "s3",
            "uri": "s3://orqis-assets/subagents/network_forensics.md",
            "size_bytes": 26000,
            "content_type": "text/markdown",
            "load_strategy": "lazy",
        },
        "communications_subagent_prompt": {
            "version": "2026-07-01",
            "kind": "prompt",
            "packaging": "inline",
            "size_bytes": len(COMMS_SUBAGENT_PROMPT.encode("utf-8")),
            "content_type": "text/markdown",
            "load_strategy": "on_demand",
        },
        "network_sandbox_bundle": {
            "version": "2026-07-01",
            "kind": "script_bundle",
            "packaging": "efs",
            "uri": "efs://orqis-assets/network-sandbox-bundle",
            "size_bytes": 180000000,
            "content_type": "application/octet-stream",
            "load_strategy": "lazy",
        },
    },
    memory_policies={
        "incident_thread": {
            "short_term_keys": [
                "incident_id",
                "severity",
                "investigation_objective",
                "executive_summary",
                "final_recommendation",
            ],
            "long_term_namespaces": ["incident_memory", "customer_updates"],
            "externalized_keys": [
                "findings",
                "containment_actions",
                "communication_drafts",
                "loaded_asset_refs",
            ],
            "summarize_keys": ["findings", "communication_drafts"],
            "max_inline_bytes": 6144,
        },
        "communications_thread": {
            "short_term_keys": ["communication_drafts", "executive_summary"],
            "long_term_namespaces": ["customer_updates"],
            "externalized_keys": ["communication_drafts"],
            "summarize_keys": ["communication_drafts"],
            "max_inline_bytes": 4096,
        },
    },
    tool_bindings={
        "identity_audit_binding": {
            "tool_id": "query_identity_audit_tool",
            "scope_kind": "agent",
            "scope_ref": "identity_forensics",
            "visibility": "allowed",
        },
        "disable_identity_session_binding": {
            "tool_id": "disable_identity_session_tool",
            "scope_kind": "agent",
            "scope_ref": "identity_forensics",
            "visibility": "required",
            "approval_policy": {
                "mode": "user_approval",
                "interrupt_before": True,
                "required_scopes": ["identity:disable"],
            },
        },
        "network_timeline_binding": {
            "tool_id": "query_network_timeline_tool",
            "scope_kind": "agent",
            "scope_ref": "network_forensics",
            "visibility": "allowed",
        },
        "network_sandbox_binding": {
            "tool_id": "run_network_sandbox_tool",
            "scope_kind": "skill",
            "scope_ref": "network_investigation",
            "visibility": "allowed",
            "requires_skill_id": "network_investigation",
        },
        "draft_customer_update_binding": {
            "tool_id": "draft_customer_update_tool",
            "scope_kind": "skill",
            "scope_ref": "customer_comms",
            "visibility": "allowed",
            "requires_skill_id": "customer_comms",
        },
    },
    skills={
        "identity_investigation": {
            "version": "1",
            "description": "Identity-focused incident response workflow and containment policy.",
            "prompt_asset_id": "identity_subagent_prompt",
            "tool_binding_ids": [
                "identity_audit_binding",
                "disable_identity_session_binding",
            ],
            "memory_policy_id": "incident_thread",
        },
        "network_investigation": {
            "version": "1",
            "description": "Network and endpoint evidence review for containment decisions.",
            "prompt_asset_id": "network_subagent_prompt",
            "asset_ids": ["network_sandbox_bundle"],
            "tool_binding_ids": [
                "network_timeline_binding",
                "network_sandbox_binding",
            ],
            "memory_policy_id": "incident_thread",
        },
        "customer_comms": {
            "version": "1",
            "description": "Stakeholder-safe incident communications workflow.",
            "prompt_asset_id": "communications_subagent_prompt",
            "tool_binding_ids": ["draft_customer_update_binding"],
            "memory_policy_id": "communications_thread",
        },
    },
    subagents={
        "identity_forensics": {
            "callable_ref": "orqis.examples.final_experiments.subagents.identity_subagent",
            "system_prompt_asset_id": "identity_subagent_prompt",
            "skill_ids": ["identity_investigation"],
            "tool_binding_ids": [
                "identity_audit_binding",
                "disable_identity_session_binding",
            ],
            "store_namespaces": ["identity_findings"],
            "memory_policy_id": "incident_thread",
            "inheritance": "inherits_parent_context",
        },
        "network_forensics": {
            "callable_ref": "orqis.examples.final_experiments.subagents.network_subagent",
            "system_prompt_asset_id": "network_subagent_prompt",
            "skill_ids": ["network_investigation"],
            "tool_binding_ids": [
                "network_timeline_binding",
                "network_sandbox_binding",
            ],
            "store_namespaces": ["network_findings", "sandbox_runs"],
            "memory_policy_id": "incident_thread",
            "inheritance": "isolated",
        },
        "customer_communications": {
            "callable_ref": "orqis.examples.final_experiments.subagents.communications_subagent",
            "system_prompt_asset_id": "communications_subagent_prompt",
            "skill_ids": ["customer_comms"],
            "tool_binding_ids": ["draft_customer_update_binding"],
            "store_namespaces": ["customer_updates"],
            "memory_policy_id": "communications_thread",
            "inheritance": "inherits_parent_context",
        },
    },
)
def build_graph():
    builder = StateGraph(IncidentResponseState)
    builder.add_node(
        "ingest_alert",
        ingest_alert,
        input_schema=IngestInput,
        metadata={
            "side_effects": {"purity": "Pure", "effect_domains": []},
            "resources": {"memory_mb": 256, "timeout_sec": 10},
        },
    )
    builder.add_node(
        "plan_response",
        plan_response,
        input_schema=PlanInput,
        metadata={
            "side_effects": {"purity": "Pure", "effect_domains": []},
            "resources": {"memory_mb": 512, "timeout_sec": 15},
        },
    )
    builder.add_node(
        "identity_subagent",
        identity_subagent,
        input_schema=SubagentTaskInput,
        metadata={
            "side_effects": {"purity": "Effectful", "effect_domains": ["llm"]},
        },
    )
    builder.add_node(
        "network_subagent",
        network_subagent,
        input_schema=SubagentTaskInput,
        metadata={
            "side_effects": {"purity": "Effectful", "effect_domains": ["llm"]},
        },
    )
    builder.add_node(
        "communications_subagent",
        communications_subagent,
        input_schema=SubagentTaskInput,
        metadata={
            "side_effects": {"purity": "Effectful", "effect_domains": ["llm"]},
        },
    )
    builder.add_node(
        "synthesize_recommendation",
        synthesize_recommendation,
        input_schema=SynthesisInput,
        defer=True,
        metadata={
            "side_effects": {"purity": "Effectful", "effect_domains": ["llm"]},
            "resources": {"memory_mb": 512, "timeout_sec": 20},
        },
    )
    builder.add_node(
        "finalize_incident",
        finalize_incident,
        input_schema=FinalizeInput,
        metadata={
            "side_effects": {"purity": "Pure", "effect_domains": []},
            "resources": {"memory_mb": 256, "timeout_sec": 10},
        },
    )
    builder.add_edge(START, "ingest_alert")
    builder.add_edge("ingest_alert", "plan_response")
    builder.add_conditional_edges(
        "plan_response",
        fanout_subagents,
        ["identity_subagent", "network_subagent", "communications_subagent"],
    )
    builder.add_edge("identity_subagent", "synthesize_recommendation")
    builder.add_edge("network_subagent", "synthesize_recommendation")
    builder.add_edge("communications_subagent", "synthesize_recommendation")
    builder.add_edge("synthesize_recommendation", "finalize_incident")
    builder.add_edge("finalize_incident", END)
    return builder.compile(name="incident_response_swarm")


def get_sample_input() -> IncidentResponseState:
    return {
        "incident_id": "inc-8841",
        "severity": "high",
        "alert_summary": (
            "Multiple MFA resets were followed by impossible-travel sign-ins and unusual VPN "
            "activity against an administrative account."
        ),
        "evidence_scope": "",
        "investigation_objective": "",
        "findings": [],
        "containment_actions": [],
        "communication_drafts": [],
        "executive_summary": "",
        "final_recommendation": "",
    }
