from __future__ import annotations

import operator

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send
from typing_extensions import Annotated, TypedDict


class DocWorkflowState(TypedDict):
    doc_id: str
    text: str
    chunks: list[str]
    chunk_summaries: Annotated[list[str], operator.add]
    final_summary: str


class IngestInput(TypedDict):
    doc_id: str
    text: str


class SplitInput(TypedDict):
    text: str


class RemoteChunkTaskInput(TypedDict):
    doc_id: str
    chunk: str


class ToyChunkTaskInput(TypedDict):
    chunk: str


class AggregateInput(TypedDict):
    chunk_summaries: list[str]


def ingest(state: IngestInput) -> dict[str, object]:
    normalized = " ".join(state["text"].split())
    return {"text": normalized}


def split(state: SplitInput) -> dict[str, object]:
    chunk_size = 24
    text = state["text"]
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    return {"chunks": chunks}


def fanout_to_remote_chunks(state: DocWorkflowState) -> list[Send]:
    return [
        Send("summarise_chunk", {"doc_id": state["doc_id"], "chunk": chunk})
        for chunk in state["chunks"]
    ]


def fanout_to_toy_chunks(state: DocWorkflowState) -> list[Send]:
    return [Send("summarise_chunk", {"chunk": chunk}) for chunk in state["chunks"]]


def summarise_chunk_remote(state: RemoteChunkTaskInput) -> dict[str, object]:
    chunk = state["chunk"]
    return {"chunk_summaries": [f"summary<{chunk[:10]}>"]}


def summarise_chunk_toy(state: ToyChunkTaskInput) -> dict[str, object]:
    chunk = state["chunk"]
    return {"chunk_summaries": [chunk[:1] or "s"]}


def aggregate(state: AggregateInput) -> dict[str, object]:
    return {"final_summary": " | ".join(state["chunk_summaries"])}


def _build_graph(*, kind: str, graph_name: str):
    builder = StateGraph(DocWorkflowState)
    builder.add_node(
        "ingest",
        ingest,
        input_schema=IngestInput,
        metadata={
            "side_effects": {"purity": "Pure", "effect_domains": []},
            "resources": {"memory_mb": 512, "timeout_sec": 15},
        },
    )
    builder.add_node(
        "split",
        split,
        input_schema=SplitInput,
        metadata={
            "side_effects": {"purity": "Pure", "effect_domains": []},
            "resources": {"memory_mb": 512, "timeout_sec": 15},
        },
    )
    if kind == "remote":
        builder.add_node(
            "summarise_chunk",
            summarise_chunk_remote,
            input_schema=RemoteChunkTaskInput,
            retry_policy=RetryPolicy(max_attempts=4, initial_interval=0.5),
            metadata={
                "side_effects": {
                    "purity": "Effectful",
                    "effect_domains": ["llm"],
                    "idempotency_key_strategy": "task_id",
                },
                "resources": {
                    "memory_mb": 1536,
                    "timeout_sec": 90,
                    "concurrency_limit": 8,
                },
            },
        )
        fanout = fanout_to_remote_chunks
    else:
        builder.add_node(
            "summarise_chunk",
            summarise_chunk_toy,
            input_schema=ToyChunkTaskInput,
            metadata={
                "side_effects": {"purity": "Pure", "effect_domains": []},
                "resources": {"memory_mb": 256, "timeout_sec": 15},
            },
        )
        fanout = fanout_to_toy_chunks
    builder.add_node(
        "aggregate",
        aggregate,
        input_schema=AggregateInput,
        defer=True,
        metadata={
            "side_effects": {"purity": "Pure", "effect_domains": []},
            "resources": {"memory_mb": 512, "timeout_sec": 15},
        },
    )
    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "split")
    builder.add_conditional_edges("split", fanout, ["summarise_chunk"])
    builder.add_edge("summarise_chunk", "aggregate")
    builder.add_edge("aggregate", END)
    return builder.compile(name=graph_name)


def build_graph():
    return _build_graph(kind="remote", graph_name="doc_summariser")


def build_remote_graph():
    return _build_graph(kind="remote", graph_name="doc_summariser_remote")


def build_toy_graph():
    return _build_graph(kind="toy", graph_name="doc_summariser_toy")


def get_sample_input() -> DocWorkflowState:
    return {
        "doc_id": "doc-001",
        "text": (
            "LangGraph compiles workflows into a Pregel runtime with clear "
            "supersteps and dynamic fanout."
        ),
        "chunks": [],
        "chunk_summaries": [],
        "final_summary": "",
    }


def get_remote_sample_input() -> DocWorkflowState:
    return get_sample_input()


def get_toy_sample_input() -> DocWorkflowState:
    return get_sample_input()
