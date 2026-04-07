from __future__ import annotations

import operator

from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, Literal, TypedDict


# typeddict state model for the loop example graph
class CounterState(TypedDict):
    current: int
    target: int
    done: bool
    history: Annotated[list[int], operator.add]
    result: str


class SeedInput(TypedDict):
    current: int


class IncrementInput(TypedDict):
    current: int


class CheckInput(TypedDict):
    current: int
    target: int


class FinalizeInput(TypedDict):
    current: int
    history: list[int]


def seed(state: SeedInput) -> dict[str, object]:
    return {"history": [state["current"]]}


def increment(state: IncrementInput) -> dict[str, object]:
    next_value = state["current"] + 1
    return {
        "current": next_value,
        "history": [next_value],
    }


def check_target(state: CheckInput) -> dict[str, object]:
    return {"done": state["current"] >= state["target"]}


def route_after_check(state: CounterState) -> Literal["increment", "finalize"]:
    return "finalize" if state["done"] else "increment"


def finalize(state: FinalizeInput) -> dict[str, object]:
    return {
        "result": f"reached {state['current']} after {len(state['history']) - 1} loop iterations"
    }


def build_graph():
    builder = StateGraph(CounterState)
    builder.add_node(
        "seed",
        seed,
        input_schema=SeedInput,
        metadata={
            "side_effects": {"purity": "Pure", "effect_domains": []},
            "resources": {"memory_mb": 256, "timeout_sec": 10},
        },
    )
    builder.add_node(
        "increment",
        increment,
        input_schema=IncrementInput,
        metadata={
            "side_effects": {"purity": "Pure", "effect_domains": []},
            "resources": {"memory_mb": 256, "timeout_sec": 10},
        },
    )
    builder.add_node(
        "check_target",
        check_target,
        input_schema=CheckInput,
        metadata={
            "side_effects": {"purity": "Pure", "effect_domains": []},
            "resources": {"memory_mb": 256, "timeout_sec": 10},
        },
    )
    builder.add_node(
        "finalize",
        finalize,
        input_schema=FinalizeInput,
        metadata={
            "side_effects": {"purity": "Pure", "effect_domains": []},
            "resources": {"memory_mb": 256, "timeout_sec": 10},
        },
    )
    builder.add_edge(START, "seed")
    builder.add_edge("seed", "increment")
    builder.add_edge("increment", "check_target")
    builder.add_conditional_edges("check_target", route_after_check, ["increment", "finalize"])
    builder.add_edge("finalize", END)
    return builder.compile(name="iterative_counter")


def get_sample_input() -> CounterState:
    return {
        "current": 0,
        "target": 3,
        "done": False,
        "history": [],
        "result": "",
    }
