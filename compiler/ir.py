from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# summary of langgraph reducer semantics
# used to decide whether concurrent writes are safe and whether lowering must preserve application order
@dataclass(slots=True)
class ReducerRef:
    reducer_id: str
    associative: bool
    commutative: bool
    has_identity: bool
    identity_expr: str | None = None
    deterministic: bool = True

# one logical state key / channel
# tracks what nodes read from state and what gets written back across supersteps
@dataclass(slots=True)
class StateKeyIR:
    key: str
    value_type: str
    channel_kind: str
    reducer: ReducerRef | None = None
    # true means write order matters
    ordering_sensitive: bool = True
    monotonic: bool | None = None
    dedupe: bool | None = None


@dataclass(slots=True)
class StateSchemaIR:
    keys: dict[str, StateKeyIR] = field(default_factory=dict)


@dataclass(slots=True)
class CachePolicyIR:
    ttl_seconds: int | None = None
    key_func_ref: str | None = None

# one retry rule after flattening langgraph retry policy objects
@dataclass(slots=True)
class RetryRuleIR:
    match: str
    max_attempts: int
    backoff_ms: int

# retry policy attached to a node or partition
@dataclass(slots=True)
class RetryPolicyIR:
    rules: list[RetryRuleIR] = field(default_factory=list)

# side-effect summary used by retries, fusion, and partition boundaries
@dataclass(slots=True)
class SideEffectIR:
    purity: str
    effect_domains: list[str] = field(default_factory=list)
    idempotency_key_strategy: str | None = None

# execution hints that later become worker sizing hints
@dataclass(slots=True)
class ResourceIR:
    cpu_class: str | None = None
    memory_mb: int | None = None
    timeout_sec: int | None = None
    concurrency_limit: int | None = None
    batchable: bool | None = None

# graph-level view of one node before lowering to explicit pregel mechanics
# close to what stategraph.add_node() receives
@dataclass(slots=True)
class NodeIR0:
    node_id: str
    kind: str
    callable_ref: str | None
    input_schema_type: str | None
    # upper bound on what the node can read from state if the user supplied an
    # explicit input schema
    input_schema_keys: list[str] = field(default_factory=list)
    defer: bool = False
    retry_policy: RetryPolicyIR | None = None
    cache_policy: CachePolicyIR | None = None
    side_effects: SideEffectIR | None = None
    resources: ResourceIR | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    destinations_decl: list[str] = field(default_factory=list)

# graph-level control-flow edge
# either a plain static edge or a conditional route that can emit node names or send payloads
@dataclass(slots=True)
class EdgeIR0:
    kind: str
    source: str | None = None
    target: str | None = None
    # used for join edges where multiple predecessors must finish before the
    # target becomes runnable
    sources: list[str] = field(default_factory=list)
    router_ref: str | None = None
    returns: str | None = None
    # for conditional edges this maps route labels to concrete destination nodes
    path_map: dict[str, str] = field(default_factory=dict)
    may_goto: list[str] = field(default_factory=list)
    may_send: bool = False
    target_graph: str | None = None

# first-stage ir: normalized graph-shaped view of the workflow
@dataclass(slots=True)
class GraphIR0:
    ir_version: str
    graph_id: str
    entrypoints: list[str]
    finishpoints: list[str]
    context_schema: str | None
    state_schema: StateSchemaIR
    nodes: dict[str, NodeIR0] = field(default_factory=dict)
    edges: list[EdgeIR0] = field(default_factory=list)
    subgraphs: dict[str, str] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

# pregel-level channel description
# captures what nodes read, write, and trigger on at runtime
@dataclass(slots=True)
class ChannelIR1:
    name: str
    kind: str
    reducer: ReducerRef | None = None
    available_predicate: str | None = None
    # reserved channels are runtime internals such as the task queue channel
    reserved: bool = False

# what a pregel node may write or route to
# includes both normal state writes and control-flow writes such as branch channels or send fanout
@dataclass(slots=True)
class WriterIR1:
    declared_channels: list[str] = field(default_factory=list)
    route_targets: list[str] = field(default_factory=list)
    state_write_keys: list[str] = field(default_factory=list)
    may_emit_send: bool = False
    notes: list[str] = field(default_factory=list)

# pregel-level node view with explicit reads, triggers, and writers
@dataclass(slots=True)
class NodeIR1:
    node_id: str
    # concrete channels this node reads when LangGraph executes it
    reads: list[str] = field(default_factory=list)
    # channels whose updates make this node runnable in the next superstep
    triggers: list[str] = field(default_factory=list)
    writer_spec: WriterIR1 = field(default_factory=WriterIR1)
    retry_policy: RetryPolicyIR | None = None
    cache_policy: CachePolicyIR | None = None
    defer: bool = False
    subgraph_ref: str | None = None

# second-stage ir: explicit pregel runtime structure
# at this stage we reason about channels, triggers, and task planning instead of only user-authored edges
@dataclass(slots=True)
class PregelIR1:
    ir_version: str
    graph_id: str
    channels: dict[str, ChannelIR1] = field(default_factory=dict)
    nodes: dict[str, NodeIR1] = field(default_factory=dict)
    # reverse index used during planning: updated channel -> candidate nodes
    trigger_to_nodes: dict[str, list[str]] = field(default_factory=dict)
    # runtime-only channels such as the internal queue of pending `Send` tasks
    reserved_channels: dict[str, dict[str, Any]] = field(default_factory=dict)
    # mirrors langgraph's deterministic ordering requirement at a summary level
    task_path_ordering: str = "lexicographic_path_prefix_3"

# read/write summary for either a node or a route function
@dataclass(slots=True)
class ReadWriteIR:
    subject_id: str
    subject_kind: str
    read_set: list[str] = field(default_factory=list)
    # state keys expected to come from the persisted checkpoint / global state
    checkpoint_read_set: list[str] = field(default_factory=list)
    # keys expected to arrive inside a pushed task payload rather than the main
    # checkpointed state; `chunk` in a Send fanout is the main example
    task_input_keys: list[str] = field(default_factory=list)
    write_set: list[str] = field(default_factory=list)
    send_targets: list[str] = field(default_factory=list)
    send_payload_keys: list[str] = field(default_factory=list)
    inferred_from: list[str] = field(default_factory=list)
    exactness: str = "best_effort"
    notes: list[str] = field(default_factory=list)

# analysis result for one state key's merge behavior under parallelism
@dataclass(slots=True)
class ReducerAnalysisIR:
    key: str
    channel_kind: str
    reducer: ReducerRef | None
    writers: list[str] = field(default_factory=list)
    # writers that can run in parallel in a fanout region and therefore need a
    # reducer or some other deterministic merge strategy
    parallel_writers: list[str] = field(default_factory=list)
    ordering_sensitive: bool = True
    safe_parallel_merge: bool = True
    notes: list[str] = field(default_factory=list)

# detected map-reduce send region
@dataclass(slots=True)
class FanoutRegionIR:
    fanout_source: str
    router_id: str
    map_nodes: list[str] = field(default_factory=list)
    # best-effort guess of the node(s) that logically consume the fan-in result
    # once all map tasks complete
    reduce_join: list[str] = field(default_factory=list)
    send_payload_keys: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

# one strongly connected component, classified as cyclic or acyclic
@dataclass(slots=True)
class LoopIR:
    component_id: str
    members: list[str] = field(default_factory=list)
    kind: str = "acyclic"
    requires_loop_capable_orchestrator: bool = False

# result of asking whether caching is safe and useful for one node
@dataclass(slots=True)
class CacheAnalysisIR:
    node_id: str
    has_cache_policy: bool
    safe_to_cache: bool
    recommended_boundary: bool
    reason: str

# all static-analysis outputs grouped into one object
@dataclass(slots=True)
class AnalysisBundle:
    read_write: dict[str, ReadWriteIR] = field(default_factory=dict)
    route_read_write: dict[str, ReadWriteIR] = field(default_factory=dict)
    reducers: list[ReducerAnalysisIR] = field(default_factory=list)
    fanout_regions: list[FanoutRegionIR] = field(default_factory=list)
    loops: list[LoopIR] = field(default_factory=list)
    cache_analysis: list[CacheAnalysisIR] = field(default_factory=list)
    side_effects: dict[str, SideEffectIR] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

# record of one attempted partition merge
# we keep rejected merges because they explain why the partitioner stopped where it did
@dataclass(slots=True)
class FusionDecisionIR:
    edge: str
    source_partition: list[str] = field(default_factory=list)
    target_partition: list[str] = field(default_factory=list)
    accepted: bool = False
    rule_results: dict[str, bool] = field(default_factory=dict)
    estimated_cost_before: float | None = None
    estimated_cost_after: float | None = None
    notes: list[str] = field(default_factory=list)

# executable partition after greedy fusion
# this is the unit that would run inside one worker invocation
@dataclass(slots=True)
class PartitionIR2:
    partition_id: str
    members: list[str] = field(default_factory=list)
    # conditional route helpers attached to member nodes and therefore executed
    # inside the same worker
    attached_routes: list[str] = field(default_factory=list)
    retry_policy: RetryPolicyIR | None = None
    cache_policy: CachePolicyIR | None = None
    resources: ResourceIR | None = None
    side_effects: SideEffectIR | None = None
    read_set: list[str] = field(default_factory=list)
    # minimal checkpoint slice the worker must load before it can start
    checkpoint_read_set: list[str] = field(default_factory=list)
    # task-local payload fields that are merged into the worker state for PUSH
    # tasks, e.g. as per-chunk input in a map stage
    task_input_keys: list[str] = field(default_factory=list)
    write_set: list[str] = field(default_factory=list)
    emits_send: bool = False
    # if true, this partition still needs a superstep barrier before downstream
    # observers may see its writes
    requires_barrier_after: bool = True
    estimated_cost: float | None = None

# third-stage ir: partitioned execution plan with explicit barriers
@dataclass(slots=True)
class GraphIR2:
    ir_version: str
    graph_id: str
    partitions: dict[str, PartitionIR2] = field(default_factory=dict)
    # lookup table used later by the planner: node id -> partition id.
    partitioned_task_model: dict[str, dict[str, str]] = field(default_factory=dict)
    barriers: str = "pregel_superstep"
    edges: dict[str, list[str]] = field(default_factory=dict)
    fusion_decisions: list[FusionDecisionIR] = field(default_factory=list)


@dataclass(slots=True)
class WorkerPlanIR:
    lambda_name: str
    memory_mb: int
    timeout_sec: int
    concurrency_limit: int | None = None
    notes: list[str] = field(default_factory=list)

# deployment sketch for queues, workers, checkpoints, and orchestration
@dataclass(slots=True)
class ServerlessPlanIR:
    plan_version: str
    graph_id: str
    persistence: dict[str, Any] = field(default_factory=dict)
    messaging: dict[str, Any] = field(default_factory=dict)
    compute: dict[str, Any] = field(default_factory=dict)
    orchestration: dict[str, Any] = field(default_factory=dict)
    security: dict[str, Any] = field(default_factory=dict)
    # canonical event shape expected by generated workers
    task_contract_fields: list[str] = field(default_factory=list)
    checkpoint_schema: dict[str, Any] = field(default_factory=dict)
    pending_writes_schema: dict[str, Any] = field(default_factory=dict)
    # human-readable sketch of the coordinator loop after lowering
    planner_outline: list[str] = field(default_factory=list)

# observed runtime task from the debug trace of a real graph run
@dataclass(slots=True)
class TaskTraceIR:
    task_id: str
    task_kind: str
    node_id: str
    triggers: list[str] = field(default_factory=list)
    input_slice: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)

# one pregel superstep reconstructed from debug events
@dataclass(slots=True)
class StepTraceIR:
    step: int
    tasks: list[TaskTraceIR] = field(default_factory=list)
    # writes grouped by state key in the order they were observed for this step
    grouped_writes: dict[str, list[Any]] = field(default_factory=dict)
    state_after_step: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# one log entry for the compiler pipeline itself
@dataclass(slots=True)
class PassTraceIR:
    pass_name: str
    description: str
    highlights: list[str] = field(default_factory=list)

# final container returned by the compiler
# report generation, json dumping, tests, and future deployment stages all read from this
@dataclass(slots=True)
class CompilationBundle:
    graph_id: str
    pass_trace: list[PassTraceIR]
    lgir0: GraphIR0
    lgir1: PregelIR1
    analysis: AnalysisBundle
    lgir2: GraphIR2
    srv_plan: ServerlessPlanIR
    runtime_trace: list[StepTraceIR] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
