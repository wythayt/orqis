from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from typing_extensions import Literal, TypedDict

from orqis import orqis
from orqis.examples.final_experiments.bedrock import invoke_json_or_fallback, invoke_text_or_fallback


class ServiceDeskState(TypedDict):
    case_id: str
    account_tier: str
    request_text: str
    normalized_text: str
    route_decision: str
    evidence_refs: list[str]
    follow_up_plan: list[str]
    response_draft: str
    final_response: str


class IntakeInput(TypedDict):
    request_text: str


class TriageInput(TypedDict):
    account_tier: str
    normalized_text: str


class SpecialistInput(TypedDict):
    case_id: str
    account_tier: str
    normalized_text: str


class FinalizeInput(TypedDict):
    route_decision: str
    response_draft: str
    evidence_refs: list[str]
    follow_up_plan: list[str]


BILLING_PLAYBOOK = """# Billing Operations Playbook

- Investigate invoice, refund, credit, and unapplied payment requests.
- Prefer evidence from the current billing period and the previous completed cycle.
- For enterprise customers, provide both the adjustment summary and the follow-up owner.
"""

IDENTITY_ACCESS_RUNBOOK = """# Identity Access Runbook

- Review recent login failures, MFA challenges, and identity-provider configuration changes.
- Explain customer-visible remediation steps and the owning team.
- Escalate broad authentication outages to the IAM response rotation.
"""

VENDOR_SECURITY_CHECKLIST = """# Vendor Security Review Checklist

- Confirm questionnaire ownership, renewal timelines, and required control evidence.
- Identify whether legal, procurement, and security all need coordinated review.
- Summarize the next packet to send back to the customer.
"""

SECURITY_REVIEW_TEMPLATE = """# Security Review Template

Sections:
- requested evidence
- blockers
- review owner
- target completion date
"""


def _json_schema_description() -> str:
    return (
        '{'
        '"route_decision": "billing | identity_access | vendor_security", '
        '"response_draft": "string", '
        '"evidence_refs": ["string"], '
        '"follow_up_plan": ["string"]'
        '}'
    )


def _normalize_specialist_payload(
    payload: dict[str, object],
    *,
    route_decision: str,
    fallback_evidence: list[str],
    fallback_follow_up: list[str],
    fallback_response: str,
) -> dict[str, object]:
    evidence_refs = payload.get("evidence_refs")
    follow_up_plan = payload.get("follow_up_plan")
    return {
        "route_decision": route_decision,
        "response_draft": str(payload.get("response_draft") or fallback_response),
        "evidence_refs": evidence_refs if isinstance(evidence_refs, list) and evidence_refs else fallback_evidence,
        "follow_up_plan": follow_up_plan if isinstance(follow_up_plan, list) and follow_up_plan else fallback_follow_up,
    }


@orqis(
    tool={
        "tool_id": "fetch_billing_ledger_tool",
        "tool_kind": "mcp",
        "description": "query invoice, credit, and payment context from the billing ledger",
        "side_effects": {
            "purity": "Idempotent",
            "effect_domains": ["db"],
            "idempotency_key_strategy": "task_id",
        },
        "resources": {"memory_mb": 512, "timeout_sec": 20},
    }
)
def fetch_billing_ledger(account_tier: str, request_text: str) -> str:
    return invoke_text_or_fallback(
        system_prompt="You summarize billing-ledger context for service-desk specialists.",
        user_prompt=(
            f"Account tier: {account_tier}\n"
            f"Customer request: {request_text}\n\n"
            "Return a concise billing-context brief with the most likely invoice, refund, or credit angle."
        ),
        fallback=f"billing ledger context for {account_tier}: {request_text[:48]}",
        max_tokens=160,
    )


@orqis(
    tool={
        "tool_id": "query_identity_audit_tool",
        "tool_kind": "mcp",
        "description": "inspect identity-provider audit context for access and MFA requests",
        "side_effects": {
            "purity": "Idempotent",
            "effect_domains": ["db"],
            "idempotency_key_strategy": "task_id",
        },
        "resources": {"memory_mb": 768, "timeout_sec": 25},
    }
)
def query_identity_audit(account_tier: str, request_text: str) -> str:
    return invoke_text_or_fallback(
        system_prompt="You summarize identity-audit context for service-desk specialists.",
        user_prompt=(
            f"Account tier: {account_tier}\n"
            f"Customer request: {request_text}\n\n"
            "Return a concise identity brief covering login, SSO, MFA, and access clues."
        ),
        fallback=f"identity audit context for {account_tier}: {request_text[:48]}",
        max_tokens=160,
    )


@orqis(
    tool={
        "tool_id": "load_vendor_security_brief_tool",
        "tool_kind": "script",
        "description": "assemble the vendor due-diligence packet from large review assets",
        "side_effects": {
            "purity": "Idempotent",
            "effect_domains": ["filesystem"],
            "idempotency_key_strategy": "task_id",
        },
        "required_asset_ids": [
            "vendor_security_checklist",
            "security_review_template",
        ],
        "resources": {"memory_mb": 4096, "timeout_sec": 180},
    }
)
def load_vendor_security_brief(case_id: str, request_text: str) -> str:
    return invoke_text_or_fallback(
        system_prompt="You assemble a concise vendor-security review packet summary.",
        user_prompt=(
            f"Case id: {case_id}\n"
            f"Request: {request_text}\n\n"
            f"Checklist:\n{VENDOR_SECURITY_CHECKLIST}\n\n"
            f"Template:\n{SECURITY_REVIEW_TEMPLATE}\n\n"
            "Return the due-diligence brief the specialist should use."
        ),
        fallback=f"vendor packet for {case_id}: {request_text[:48]}",
        max_tokens=220,
    )


@orqis(
    skills=["billing_ops"],
    resources={"memory_mb": 512, "timeout_sec": 20},
    side_effects={
        "purity": "Effectful",
        "effect_domains": ["llm"],
        "idempotency_key_strategy": "task_id",
    },
)
def billing_specialist(state: SpecialistInput) -> dict[str, object]:
    context = fetch_billing_ledger(state["account_tier"], state["normalized_text"])
    fallback_response = (
        "We will reconcile the invoice delta, confirm whether a credit memo is pending, "
        "and send the account owner a billing adjustment timeline."
    )
    fallback = {
        "route_decision": "billing",
        "response_draft": fallback_response,
        "evidence_refs": [f"ledger:{state['case_id']}"],
        "follow_up_plan": ["billing analyst to confirm open adjustments within one business day"],
    }
    payload = invoke_json_or_fallback(
        system_prompt="You are the billing specialist for a customer service desk.",
        user_prompt=(
            f"Billing playbook:\n{BILLING_PLAYBOOK}\n\n"
            f"Case id: {state['case_id']}\n"
            f"Account tier: {state['account_tier']}\n"
            f"Normalized request: {state['normalized_text']}\n"
            f"Billing context: {context}\n\n"
            "Write the specialist result for the billing branch."
        ),
        schema_description=_json_schema_description(),
        fallback=fallback,
        max_tokens=320,
    )
    return _normalize_specialist_payload(
        payload,
        route_decision="billing",
        fallback_evidence=[f"ledger:{state['case_id']}"],
        fallback_follow_up=["billing analyst to confirm open adjustments within one business day"],
        fallback_response=fallback_response,
    )


@orqis(
    skills=["identity_access_ops"],
    resources={"memory_mb": 768, "timeout_sec": 25},
    side_effects={
        "purity": "Effectful",
        "effect_domains": ["llm"],
        "idempotency_key_strategy": "task_id",
    },
)
def identity_specialist(state: SpecialistInput) -> dict[str, object]:
    context = query_identity_audit(state["account_tier"], state["normalized_text"])
    fallback_response = (
        "We will review recent SSO and MFA events, confirm whether the issue is tenant-specific, "
        "and provide the next remediation step for the identity admin."
    )
    fallback = {
        "route_decision": "identity_access",
        "response_draft": fallback_response,
        "evidence_refs": [f"idp-audit:{state['case_id']}"],
        "follow_up_plan": ["identity engineer to validate login and MFA telemetry"],
    }
    payload = invoke_json_or_fallback(
        system_prompt="You are the identity-access specialist for a customer service desk.",
        user_prompt=(
            f"Identity runbook:\n{IDENTITY_ACCESS_RUNBOOK}\n\n"
            f"Case id: {state['case_id']}\n"
            f"Account tier: {state['account_tier']}\n"
            f"Normalized request: {state['normalized_text']}\n"
            f"Identity context: {context}\n\n"
            "Write the specialist result for the identity-access branch."
        ),
        schema_description=_json_schema_description(),
        fallback=fallback,
        max_tokens=320,
    )
    return _normalize_specialist_payload(
        payload,
        route_decision="identity_access",
        fallback_evidence=[f"idp-audit:{state['case_id']}"],
        fallback_follow_up=["identity engineer to validate login and MFA telemetry"],
        fallback_response=fallback_response,
    )


@orqis(
    skills=["vendor_security_ops"],
    resources={"memory_mb": 4096, "timeout_sec": 180},
    side_effects={
        "purity": "Effectful",
        "effect_domains": ["llm"],
        "idempotency_key_strategy": "task_id",
    },
)
def vendor_security_specialist(state: SpecialistInput) -> dict[str, object]:
    brief = load_vendor_security_brief(state["case_id"], state["normalized_text"])
    fallback_response = (
        "We will assemble the vendor due-diligence packet, align legal and security reviewers, "
        "and return the completed control evidence package with a target completion date."
    )
    fallback = {
        "route_decision": "vendor_security",
        "response_draft": fallback_response,
        "evidence_refs": [
            f"vendor-brief:{state['case_id']}",
            "asset:vendor_security_checklist",
            "asset:security_review_template",
        ],
        "follow_up_plan": [
            "procurement to confirm renewal deadline",
            "security reviewer to complete questionnaire evidence mapping",
        ],
    }
    payload = invoke_json_or_fallback(
        system_prompt="You are the vendor-security specialist for a customer service desk.",
        user_prompt=(
            f"Vendor checklist:\n{VENDOR_SECURITY_CHECKLIST}\n\n"
            f"Security review template:\n{SECURITY_REVIEW_TEMPLATE}\n\n"
            f"Case id: {state['case_id']}\n"
            f"Account tier: {state['account_tier']}\n"
            f"Normalized request: {state['normalized_text']}\n"
            f"Vendor brief: {brief}\n\n"
            "Write the specialist result for the vendor-security branch."
        ),
        schema_description=_json_schema_description(),
        fallback=fallback,
        max_tokens=360,
    )
    return _normalize_specialist_payload(
        payload,
        route_decision="vendor_security",
        fallback_evidence=[
            f"vendor-brief:{state['case_id']}",
            "asset:vendor_security_checklist",
            "asset:security_review_template",
        ],
        fallback_follow_up=[
            "procurement to confirm renewal deadline",
            "security reviewer to complete questionnaire evidence mapping",
        ],
        fallback_response=fallback_response,
    )


def intake_request(state: IntakeInput) -> dict[str, object]:
    normalized = " ".join(state["request_text"].split()).lower()
    return {"normalized_text": normalized}


def triage_request(state: TriageInput) -> dict[str, object]:
    text = state["normalized_text"]
    if any(keyword in text for keyword in ("invoice", "refund", "credit", "billing", "charge")):
        route = "billing"
    elif any(keyword in text for keyword in ("login", "sso", "mfa", "access", "okta")):
        route = "identity_access"
    else:
        route = "vendor_security"
    return {"route_decision": route}


def route_specialist(state: ServiceDeskState) -> Literal[
    "billing_specialist",
    "identity_specialist",
    "vendor_security_specialist",
]:
    if state["route_decision"] == "billing":
        return "billing_specialist"
    if state["route_decision"] == "identity_access":
        return "identity_specialist"
    return "vendor_security_specialist"


def finalize_response(state: FinalizeInput) -> dict[str, object]:
    evidence = "; ".join(state["evidence_refs"])
    follow_up = " | ".join(state["follow_up_plan"])
    return {
        "final_response": (
            f"Route: {state['route_decision']}. {state['response_draft']} "
            f"Evidence: {evidence}. Follow-up: {follow_up}."
        )
    }


@orqis(
    assets={
        "billing_playbook": {
            "version": "2026-07-01",
            "kind": "prompt",
            "packaging": "inline",
            "size_bytes": len(BILLING_PLAYBOOK.encode("utf-8")),
            "content_type": "text/markdown",
            "load_strategy": "on_demand",
        },
        "identity_access_runbook": {
            "version": "2026-07-01",
            "kind": "prompt",
            "packaging": "inline",
            "size_bytes": len(IDENTITY_ACCESS_RUNBOOK.encode("utf-8")),
            "content_type": "text/markdown",
            "load_strategy": "on_demand",
        },
        "vendor_security_checklist": {
            "version": "2026-07-01",
            "kind": "prompt",
            "packaging": "s3",
            "uri": "s3://orqis-assets/vendor-security/checklist.md",
            "size_bytes": 42000,
            "content_type": "text/markdown",
            "load_strategy": "lazy",
        },
        "security_review_template": {
            "version": "2026-07-01",
            "kind": "template",
            "packaging": "s3",
            "uri": "s3://orqis-assets/vendor-security/template.md",
            "size_bytes": 18000,
            "content_type": "text/markdown",
            "load_strategy": "lazy",
        },
    },
    memory_policies={
        "service_desk_thread": {
            "short_term_keys": [
                "route_decision",
                "normalized_text",
                "response_draft",
                "final_response",
            ],
            "long_term_namespaces": ["customer_accounts", "vendor_reviews"],
            "externalized_keys": ["evidence_refs", "tool_packets"],
            "summarize_keys": ["response_draft"],
            "max_inline_bytes": 8192,
        }
    },
    tool_bindings={
        "billing_context_binding": {
            "tool_id": "fetch_billing_ledger_tool",
            "scope_kind": "skill",
            "scope_ref": "billing_ops",
            "visibility": "allowed",
            "requires_skill_id": "billing_ops",
        },
        "identity_audit_binding": {
            "tool_id": "query_identity_audit_tool",
            "scope_kind": "skill",
            "scope_ref": "identity_access_ops",
            "visibility": "allowed",
            "requires_skill_id": "identity_access_ops",
        },
        "vendor_security_briefing_binding": {
            "tool_id": "load_vendor_security_brief_tool",
            "scope_kind": "skill",
            "scope_ref": "vendor_security_ops",
            "visibility": "required",
            "requires_skill_id": "vendor_security_ops",
        },
    },
    skills={
        "billing_ops": {
            "version": "1",
            "description": "Playbook for invoice, refund, and credit reconciliation.",
            "prompt_asset_id": "billing_playbook",
            "tool_binding_ids": ["billing_context_binding"],
            "memory_policy_id": "service_desk_thread",
        },
        "identity_access_ops": {
            "version": "1",
            "description": "Runbook for SSO, MFA, and access troubleshooting.",
            "prompt_asset_id": "identity_access_runbook",
            "tool_binding_ids": ["identity_audit_binding"],
            "memory_policy_id": "service_desk_thread",
        },
        "vendor_security_ops": {
            "version": "1",
            "description": "Vendor due-diligence and questionnaire handling workflow.",
            "prompt_asset_id": "vendor_security_checklist",
            "asset_ids": ["security_review_template"],
            "tool_binding_ids": ["vendor_security_briefing_binding"],
            "memory_policy_id": "service_desk_thread",
        },
    },
)
def build_graph():
    builder = StateGraph(ServiceDeskState)
    builder.add_node(
        "intake_request",
        intake_request,
        input_schema=IntakeInput,
        metadata={
            "side_effects": {"purity": "Pure", "effect_domains": []},
            "resources": {"memory_mb": 256, "timeout_sec": 10},
        },
    )
    builder.add_node(
        "triage_request",
        triage_request,
        input_schema=TriageInput,
        metadata={
            "side_effects": {"purity": "Pure", "effect_domains": []},
            "resources": {"memory_mb": 256, "timeout_sec": 10},
        },
    )
    builder.add_node(
        "billing_specialist",
        billing_specialist,
        input_schema=SpecialistInput,
        metadata={
            "side_effects": {"purity": "Effectful", "effect_domains": ["llm"]},
        },
    )
    builder.add_node(
        "identity_specialist",
        identity_specialist,
        input_schema=SpecialistInput,
        metadata={
            "side_effects": {"purity": "Effectful", "effect_domains": ["llm"]},
        },
    )
    builder.add_node(
        "vendor_security_specialist",
        vendor_security_specialist,
        input_schema=SpecialistInput,
        metadata={
            "side_effects": {"purity": "Effectful", "effect_domains": ["llm"]},
        },
    )
    builder.add_node(
        "finalize_response",
        finalize_response,
        input_schema=FinalizeInput,
        metadata={
            "side_effects": {"purity": "Pure", "effect_domains": []},
            "resources": {"memory_mb": 256, "timeout_sec": 10},
        },
    )
    builder.add_edge(START, "intake_request")
    builder.add_edge("intake_request", "triage_request")
    builder.add_conditional_edges(
        "triage_request",
        route_specialist,
        ["billing_specialist", "identity_specialist", "vendor_security_specialist"],
    )
    builder.add_edge("billing_specialist", "finalize_response")
    builder.add_edge("identity_specialist", "finalize_response")
    builder.add_edge("vendor_security_specialist", "finalize_response")
    builder.add_edge("finalize_response", END)
    return builder.compile(name="service_desk_router")


def get_sample_input() -> ServiceDeskState:
    return {
        "case_id": "case-24017",
        "account_tier": "enterprise",
        "request_text": (
            "Our renewal team needs the latest vendor due diligence package and the "
            "security questionnaire completed before the contract renewal review."
        ),
        "normalized_text": "",
        "route_decision": "",
        "evidence_refs": [],
        "follow_up_plan": [],
        "response_draft": "",
        "final_response": "",
    }
