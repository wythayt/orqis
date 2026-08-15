# Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant#optional-track-loaded-skills-and-enforce-tool-constraints
from __future__ import annotations

from typing import Any, Callable, NotRequired, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain.messages import SystemMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from orqis import orqis
from orqis.examples.final_experiments.bedrock import BedrockToolCallingChatModel, current_bedrock_config


class CustomState(AgentState):
    skills_loaded: NotRequired[list[str]]


class Skill(TypedDict):
    name: str
    description: str
    content: str


SKILLS: list[Skill] = [
    {
        "name": "sales_analytics",
        "description": "Database schema and business logic for sales data analysis including customers, orders, and revenue.",
        "content": """# Sales Analytics Schema

## Tables

### customers
- customer_id (PRIMARY KEY)
- name
- email
- signup_date
- status (active/inactive)
- customer_tier (bronze/silver/gold/platinum)

### orders
- order_id (PRIMARY KEY)
- customer_id (FOREIGN KEY -> customers)
- order_date
- status (pending/completed/cancelled/refunded)
- total_amount
- sales_region (north/south/east/west)

### order_items
- item_id (PRIMARY KEY)
- order_id (FOREIGN KEY -> orders)
- product_id
- quantity
- unit_price
- discount_percent

## Business Logic

**Active customers**: status = 'active' AND signup_date <= CURRENT_DATE - INTERVAL '90 days'

**Revenue calculation**: Only count orders with status = 'completed'. Use total_amount from orders table, which already accounts for discounts.

**Customer lifetime value (CLV)**: Sum of all completed order amounts for a customer.

**High-value orders**: Orders with total_amount > 1000

## Example Query

-- Get top 10 customers by revenue in the last quarter
SELECT
    c.customer_id,
    c.name,
    c.customer_tier,
    SUM(o.total_amount) as total_revenue
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
  AND o.order_date >= CURRENT_DATE - INTERVAL '3 months'
GROUP BY c.customer_id, c.name, c.customer_tier
ORDER BY total_revenue DESC
LIMIT 10;
""",
    },
    {
        "name": "inventory_management",
        "description": "Database schema and business logic for inventory tracking including products, warehouses, and stock levels.",
        "content": """# Inventory Management Schema

## Tables

### products
- product_id (PRIMARY KEY)
- product_name
- sku
- category
- unit_cost
- reorder_point (minimum stock level before reordering)
- discontinued (boolean)

### warehouses
- warehouse_id (PRIMARY KEY)
- warehouse_name
- location
- capacity

### inventory
- inventory_id (PRIMARY KEY)
- product_id (FOREIGN KEY -> products)
- warehouse_id (FOREIGN KEY -> warehouses)
- quantity_on_hand
- last_updated

### stock_movements
- movement_id (PRIMARY KEY)
- product_id (FOREIGN KEY -> products)
- warehouse_id (FOREIGN KEY -> warehouses)
- movement_type (inbound/outbound/transfer/adjustment)
- quantity (positive for inbound, negative for outbound)
- movement_date
- reference_number

## Business Logic

**Available stock**: quantity_on_hand from inventory table where quantity_on_hand > 0

**Products needing reorder**: Products where total quantity_on_hand across all warehouses is less than or equal to the product's reorder_point

**Active products only**: Exclude products where discontinued = true unless specifically analyzing discontinued items

**Stock valuation**: quantity_on_hand * unit_cost for each product

## Example Query

-- Find products below reorder point across all warehouses
SELECT
    p.product_id,
    p.product_name,
    p.reorder_point,
    SUM(i.quantity_on_hand) as total_stock,
    p.unit_cost,
    (p.reorder_point - SUM(i.quantity_on_hand)) as units_to_reorder
FROM products p
JOIN inventory i ON p.product_id = i.product_id
WHERE p.discontinued = false
GROUP BY p.product_id, p.product_name, p.reorder_point, p.unit_cost
HAVING SUM(i.quantity_on_hand) <= p.reorder_point
ORDER BY units_to_reorder DESC;
""",
    },
]

SKILL_ASSETS = {
    skill["name"]: {
        "version": "2026-05-27",
        "kind": "prompt",
        "packaging": "inline",
        "size_bytes": len(skill["content"].encode("utf-8")),
        "content_type": "text/markdown",
        "load_strategy": "on_demand",
    }
    for skill in SKILLS
}
SKILL_MANIFESTS = {
    skill["name"]: {
        "version": "1",
        "description": skill["description"],
        "prompt_asset_id": skill["name"],
        "memory_policy_id": "sql_agent_thread",
        "load_strategy": "on_demand",
    }
    for skill in SKILLS
}


@orqis(
    tool={
        "tool_id": "load_skill_tool",
        "tool_kind": "python",
        "description": "load a named skill into the agent context",
        "executor_id": "lambda_agent_tools",
        "side_effects": {"purity": "Pure", "effect_domains": []},
    }
)
@tool
def load_skill(skill_name: str, runtime: ToolRuntime) -> Command:
    """Load a named skill into the current agent thread."""

    for skill in SKILLS:
        if skill["name"] == skill_name:
            skill_content = f"Loaded skill: {skill_name}\n\n{skill['content']}"
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=skill_content,
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                    "skills_loaded": [skill_name],
                }
            )

    available = ", ".join(s["name"] for s in SKILLS)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Skill {skill_name} not found. Available skills: {available}",
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


@orqis(
    tool={
        "tool_id": "write_sql_query_tool",
        "tool_kind": "python",
        "description": "write and validate SQL after the right skill has been loaded",
        "executor_id": "lambda_agent_tools",
        "side_effects": {
            "purity": "Idempotent",
            "effect_domains": ["db"],
            "idempotency_key_strategy": "task_id",
        },
    }
)
@tool
def write_sql_query(
    query: str,
    vertical: str,
    runtime: ToolRuntime,
) -> str:
    """Write and validate a SQL query once the required skill is loaded."""

    skills_loaded = runtime.state.get("skills_loaded", [])
    if vertical not in skills_loaded:
        return (
            f"Error: You must load the '{vertical}' skill first "
            f"to understand the database schema before writing queries. "
            f"Use load_skill('{vertical}') to load the schema."
        )

    return (
        f"SQL Query for {vertical}:\n\n"
        f"```sql\n{query}\n```\n\n"
        f"✓ Query validated against {vertical} schema\n"
        f"Ready to execute against the database."
    )


class SkillMiddleware(AgentMiddleware[CustomState]):
    state_schema = CustomState
    tools = [load_skill, write_sql_query]

    def __init__(self):
        skills_list = []
        for skill in SKILLS:
            skills_list.append(f"- **{skill['name']}**: {skill['description']}")
        self.skills_prompt = "\n".join(skills_list)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        skills_addendum = (
            f"\n\n## Available Skills\n\n{self.skills_prompt}\n\n"
            "Use the load_skill tool when you need detailed information "
            "about handling a specific type of request."
        )
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)


def build_agent_model() -> BedrockToolCallingChatModel:
    cfg = current_bedrock_config()
    return BedrockToolCallingChatModel(
        model_id=cfg["model_id"],
        region_name=cfg["region_name"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )


@orqis(
    assets=SKILL_ASSETS,
    memory_policies={
        "sql_agent_thread": {
            "short_term_keys": ["messages", "skills_loaded", "active_skills", "tool_scope"],
            "long_term_namespaces": ["sql_profiles"],
            "externalized_keys": ["loaded_skill_assets"],
            "summarize_keys": ["messages"],
            "max_inline_bytes": 12288,
        }
    },
    executors={
        "lambda_agent_model": {
            "backend": "lambda",
            "runtime": "python3.11",
            "packaging": "zip",
            "filesystem": "tmp",
            "network_access": "egress",
            "supports_streaming": True,
            "resources": {"memory_mb": 1024, "timeout_sec": 30},
        },
        "lambda_agent_tools": {
            "backend": "lambda",
            "runtime": "python3.11",
            "packaging": "zip",
            "filesystem": "tmp",
            "network_access": "egress",
            "resources": {"memory_mb": 1024, "timeout_sec": 30},
        },
    },
    tool_bindings={
        "load_skill_binding": {
            "tool_id": "load_skill_tool",
            "scope_kind": "node",
            "scope_ref": "tools",
            "visibility": "required",
        },
        "write_sales_sql_binding": {
            "tool_id": "write_sql_query_tool",
            "scope_kind": "skill",
            "scope_ref": "sales_analytics",
            "visibility": "allowed",
            "requires_skill_id": "sales_analytics",
        },
        "write_inventory_sql_binding": {
            "tool_id": "write_sql_query_tool",
            "scope_kind": "skill",
            "scope_ref": "inventory_management",
            "visibility": "allowed",
            "requires_skill_id": "inventory_management",
        },
    },
    skills=SKILL_MANIFESTS,
    node_overrides={
        "model": {
            "skills": [skill["name"] for skill in SKILLS],
            "executor": "lambda_agent_model",
            "resources": {"memory_mb": 1024, "timeout_sec": 30},
            "side_effects": {
                "purity": "Effectful",
                "effect_domains": ["llm"],
                "idempotency_key_strategy": "task_id",
            },
        },
        "tools": {
            "skills": [skill["name"] for skill in SKILLS],
            "tool_bindings": ["load_skill_binding"],
            "executor": "lambda_agent_tools",
            "resources": {"memory_mb": 1024, "timeout_sec": 30},
            "side_effects": {"purity": "Idempotent", "effect_domains": ["db"]},
        },
    },
)
def build_graph(model: Any | None = None, *, checkpointer: Any | None = None):
    return create_agent(
        model=model or build_agent_model(),
        system_prompt=(
            "You are a SQL query assistant that helps users "
            "write queries against business databases."
        ),
        middleware=[SkillMiddleware()],
        checkpointer=checkpointer,
    )


def get_sample_input() -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a SQL query to find all customers "
                    "who made orders over $1000 in the last month"
                ),
            }
        ]
    }


def build_runtime_agent():
    return build_graph(checkpointer=InMemorySaver())


if __name__ == "__main__":
    agent = build_runtime_agent()
    thread_id = str(uuid7())
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(get_sample_input(), config)
    for message in result["messages"]:
        if hasattr(message, "pretty_print"):
            message.pretty_print()
        else:
            print(f"{message.type}: {message.content}")
