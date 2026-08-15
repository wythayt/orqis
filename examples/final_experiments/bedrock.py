from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar, Token
from functools import lru_cache
from time import perf_counter
from typing import Any, Callable, Sequence
from uuid import uuid4

from botocore.config import Config
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool


DEFAULT_MODEL_ID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
_BEDROCK_TELEMETRY_CALLBACK: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar(
    "orqis_bedrock_telemetry_callback",
    default=None,
)


def bedrock_enabled() -> bool:
    return bool(os.environ.get("ORQIS_BEDROCK_MODEL_ID"))


def current_bedrock_config() -> dict[str, Any]:
    return {
        "model_id": os.environ.get("ORQIS_BEDROCK_MODEL_ID", DEFAULT_MODEL_ID),
        "region_name": os.environ.get("ORQIS_BEDROCK_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION"),
        "temperature": float(os.environ.get("ORQIS_BEDROCK_TEMPERATURE", "0.0")),
        "max_tokens": max(1, int(os.environ.get("ORQIS_BEDROCK_MAX_OUTPUT_TOKENS", "512"))),
    }


@lru_cache(maxsize=4)
def _bedrock_client(region_name: str | None):
    import boto3

    kwargs = {"region_name": region_name} if region_name else {}
    return boto3.client(
        "bedrock-runtime",
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
        **kwargs,
    )


def _response_text(response: dict[str, Any]) -> str:
    content = response.get("output", {}).get("message", {}).get("content", [])
    return "\n".join(
        part.get("text", "").strip()
        for part in content
        if isinstance(part, dict) and part.get("text")
    ).strip()


def _record_bedrock_telemetry(event: dict[str, Any]) -> None:
    callback = _BEDROCK_TELEMETRY_CALLBACK.get()
    if callback is None:
        return
    callback(dict(event))


@contextmanager
def bedrock_telemetry_session(callback: Callable[[dict[str, Any]], None]):
    token: Token[Callable[[dict[str, Any]], None] | None] = _BEDROCK_TELEMETRY_CALLBACK.set(callback)
    try:
        yield
    finally:
        _BEDROCK_TELEMETRY_CALLBACK.reset(token)


def converse_text(
    *,
    system_prompt: str | None,
    user_prompt: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    model_id: str | None = None,
    region_name: str | None = None,
) -> str:
    cfg = current_bedrock_config()
    resolved_model_id = model_id or cfg["model_id"]
    resolved_region_name = region_name or cfg["region_name"]
    resolved_max_tokens = max_tokens or cfg["max_tokens"]
    resolved_temperature = cfg["temperature"] if temperature is None else temperature
    started = perf_counter()
    response = _bedrock_client(resolved_region_name).converse(
        modelId=resolved_model_id,
        system=[{"text": system_prompt}] if system_prompt else [],
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={
            "maxTokens": resolved_max_tokens,
            "temperature": resolved_temperature,
        },
    )
    text = _response_text(response)
    usage = response.get("usage", {})
    _record_bedrock_telemetry(
        {
            "provider": "bedrock",
            "model_id": resolved_model_id,
            "region_name": resolved_region_name,
            "latency_ms": round((perf_counter() - started) * 1000, 3),
            "input_tokens": int(usage.get("inputTokens", 0) or 0),
            "output_tokens": int(usage.get("outputTokens", 0) or 0),
            "total_tokens": int(usage.get("totalTokens", 0) or 0),
            "response_chars": len(text),
        }
    )
    return text


def _extract_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidate_texts: list[str] = []
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
        candidate = match.group(1).strip()
        if candidate:
            candidate_texts.append(candidate)
    stripped = text.strip()
    if stripped:
        candidate_texts.append(stripped)

    for candidate_text in candidate_texts:
        for match in re.finditer(r"\{", candidate_text):
            try:
                parsed, _ = decoder.raw_decode(candidate_text[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise ValueError("model response did not contain a JSON object")


def _repair_json_response(
    *,
    response_text: str,
    schema_description: str,
    model_id: str | None = None,
    region_name: str | None = None,
    max_tokens: int = 768,
) -> str:
    return converse_text(
        system_prompt=(
            "You repair model outputs into exactly one valid JSON object. "
            "Return JSON only, without markdown fences or commentary."
        ),
        user_prompt=(
            "Convert the following response into exactly one JSON object.\n\n"
            f"Required schema:\n{schema_description}\n\n"
            "If a field is missing, use an empty string, empty list, or empty object as appropriate.\n\n"
            "Response to repair:\n"
            f"{response_text}"
        ),
        model_id=model_id,
        region_name=region_name,
        max_tokens=max_tokens,
        temperature=0.0,
    )


def invoke_text_or_fallback(
    *,
    system_prompt: str | None,
    user_prompt: str,
    fallback: str | Callable[[], str],
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    if not bedrock_enabled():
        return fallback() if callable(fallback) else fallback
    return converse_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def invoke_json_or_fallback(
    *,
    system_prompt: str | None,
    user_prompt: str,
    schema_description: str,
    fallback: dict[str, Any] | Callable[[], dict[str, Any]],
    max_tokens: int = 768,
    temperature: float = 0.0,
) -> dict[str, Any]:
    if not bedrock_enabled():
        return fallback() if callable(fallback) else dict(fallback)
    cfg = current_bedrock_config()
    response_text = converse_text(
        system_prompt=system_prompt,
        user_prompt=(
            f"{user_prompt}\n\n"
            "Return exactly one JSON object and no prose.\n"
            f"Required schema:\n{schema_description}"
        ),
        model_id=cfg["model_id"],
        region_name=cfg["region_name"],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    try:
        return _extract_first_json_object(response_text)
    except ValueError:
        repaired_text = _repair_json_response(
            response_text=response_text,
            schema_description=schema_description,
            model_id=cfg["model_id"],
            region_name=cfg["region_name"],
            max_tokens=max_tokens,
        )
        try:
            return _extract_first_json_object(repaired_text)
        except ValueError:
            return fallback() if callable(fallback) else dict(fallback)


def _tool_spec(tool: dict[str, Any] | type | Callable[..., Any] | BaseTool) -> dict[str, str]:
    if isinstance(tool, BaseTool):
        return {"name": tool.name, "description": tool.description or ""}
    if isinstance(tool, dict):
        function_spec = tool.get("function", tool)
        return {
            "name": str(function_spec.get("name", "tool")),
            "description": str(function_spec.get("description", "")),
        }
    if hasattr(tool, "name"):
        return {"name": str(getattr(tool, "name")), "description": str(getattr(tool, "description", ""))}
    return {
        "name": getattr(tool, "__name__", "tool"),
        "description": (getattr(tool, "__doc__", "") or "").strip(),
    }


def _messages_to_text(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            lines.append(f"tool: {message.content}")
            continue
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            lines.append(f"assistant_tool_calls: {json.dumps(tool_calls, ensure_ascii=True)}")
            continue
        role = getattr(message, "type", message.__class__.__name__).lower()
        message_text = str(getattr(message, "content", ""))
        lines.append(f"{role}: {message_text}")
    return "\n".join(lines)


def _loaded_skills_from_messages(messages: list[BaseMessage]) -> list[str]:
    loaded_skills: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        content = str(message.content or "")
        if not content.startswith("Loaded skill: "):
            continue
        skill_name = content.split("Loaded skill: ", 1)[1].splitlines()[0].strip()
        if skill_name and skill_name not in loaded_skills:
            loaded_skills.append(skill_name)
    return loaded_skills


def _fallback_skill_decision(messages: list[BaseMessage]) -> AIMessage:
    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    tool_texts = [str(message.content) for message in tool_messages]
    if not tool_messages:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"call_{uuid4().hex[:8]}",
                    "name": "load_skill",
                    "args": {"skill_name": "sales_analytics"},
                    "type": "tool_call",
                }
            ],
        )
    if not any(text.startswith("SQL Query for ") for text in tool_texts):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"call_{uuid4().hex[:8]}",
                    "name": "write_sql_query",
                    "args": {
                        "query": (
                            "SELECT customer_id, total_amount "
                            "FROM orders WHERE status = 'completed' "
                            "AND total_amount > 1000"
                        ),
                        "vertical": "sales_analytics",
                    },
                    "type": "tool_call",
                }
            ],
        )
    return AIMessage(content="Done: loaded the sales skill and produced the SQL query.", tool_calls=[])


def _default_skill_response(messages: list[BaseMessage]) -> str:
    sql_vertical: str | None = None
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        content = str(message.content or "")
        if content.startswith("SQL Query for "):
            sql_vertical = content.split("SQL Query for ", 1)[1].split(":", 1)[0].strip()
            break
    if sql_vertical:
        return f"Done: loaded the {sql_vertical} skill and produced the SQL query."
    loaded_skills = _loaded_skills_from_messages(messages)
    if len(loaded_skills) == 1:
        return f"Done: loaded the {loaded_skills[0]} skill."
    return "Done."


class BedrockToolCallingChatModel(BaseChatModel):
    model_id: str = DEFAULT_MODEL_ID
    region_name: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512
    bound_tools: tuple[dict[str, str], ...] = ()
    bound_tool_choice: str | None = None

    @property
    def _llm_type(self) -> str:
        return "bedrock_converse_tool_model"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        del kwargs
        return self.model_copy(
            update={
                "bound_tools": tuple(_tool_spec(tool) for tool in tools),
                "bound_tool_choice": tool_choice,
            }
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        if not bedrock_enabled():
            message = _fallback_skill_decision(messages)
            return ChatResult(generations=[ChatGeneration(message=message)])

        tool_lines = "\n".join(
            f"- {tool['name']}: {tool['description']}"
            for tool in self.bound_tools
        )
        schema = (
            '{'
            '"action": "tool" | "respond", '
            '"tool_name": "tool name or empty string", '
            '"arguments": {"arg": "value"}, '
            '"response": "final response text"'
            '}'
        )
        payload = invoke_json_or_fallback(
            system_prompt=(
                "You are a Bedrock-backed controller for the ORQIS skills example. "
                "Choose the next tool call when necessary, otherwise return the final answer."
            ),
            user_prompt=(
                "Conversation so far:\n"
                f"{_messages_to_text(messages)}\n\n"
                "Available tools:\n"
                f"{tool_lines}\n\n"
                "When the user needs schema details, call load_skill first. "
                "When the relevant skill is already loaded, call write_sql_query. "
                "When calling write_sql_query, always include both 'query' and "
                "'vertical', and set 'vertical' to exactly one loaded skill name. "
                "If the task is complete, return a final textual response. "
                "Never leave the response field empty when action is respond."
            ),
            schema_description=schema,
            fallback=lambda: {
                "action": "respond",
                "tool_name": "",
                "arguments": {},
                "response": "Done: loaded the skill and produced the SQL query.",
            },
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if payload.get("action") == "tool":
            tool_name = str(payload.get("tool_name", "")).strip()
            arguments = payload.get("arguments") or {}
            loaded_skills = _loaded_skills_from_messages(messages)
            if tool_name == "write_sql_query" and "vertical" not in arguments and len(loaded_skills) == 1:
                arguments = dict(arguments)
                arguments["vertical"] = loaded_skills[0]
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{uuid4().hex[:8]}",
                        "name": tool_name,
                        "args": arguments,
                        "type": "tool_call",
                    }
                ],
            )
        else:
            response_text = str(payload.get("response", "")).strip()
            if not response_text:
                response_text = _default_skill_response(messages)
            message = AIMessage(content=response_text, tool_calls=[])
        return ChatResult(generations=[ChatGeneration(message=message)])
