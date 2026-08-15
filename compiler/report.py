from __future__ import annotations

from pathlib import Path

from orqis.compiler.ir import CompilationBundle
from orqis.compiler.utils import ensure_directory, to_jsonable, write_json


def _field(value: object, key: str, default: object = None) -> object:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def render_report(bundle: CompilationBundle) -> str:
    lines: list[str] = []
    lines.append(f"# Compilation report: `{bundle.graph_id}`")
    lines.append("")
    lines.append(f"- resource policy: `{bundle.resource_policy}`")
    slo_profile_id = _field(bundle.srv_plan.compute.get("resource_summary", {}), "slo_profile_id")
    if slo_profile_id:
        lines.append(f"- SLO profile: `{slo_profile_id}`")
    lines.append("")
    lines.append("## Pass trace")
    for item in bundle.pass_trace:
        lines.append(f"- **{item.pass_name}**: {item.description}")
        for highlight in item.highlights:
            lines.append(f"  - {highlight}")
    lines.append("")
    lines.append("## LGIR-0")
    lines.append("")
    lines.append("### State keys")
    for key, state_key in bundle.lgir0.state_schema.keys.items():
        reducer = state_key.reducer.reducer_id if state_key.reducer else "none"
        lines.append(
            f"- `{key}`: {state_key.channel_kind}, reducer={reducer}, ordering_sensitive={state_key.ordering_sensitive}"
        )
    lines.append("")
    lines.append("### Nodes")
    for node_id, node in bundle.lgir0.nodes.items():
        lines.append(
            f"- `{node_id}`: kind={node.kind}, input={node.input_schema_keys}, defer={node.defer}, callable={node.callable_ref}"
        )
        if node.skill_ids or node.tool_binding_ids or node.subagent_id or node.executor_id:
            lines.append(
                f"  - skills={node.skill_ids}, tools={node.tool_binding_ids}, subagent={node.subagent_id}, executor={node.executor_id}"
            )
    lines.append("")
    if bundle.lgir0.assets:
        lines.append("### Assets")
        for asset_id, asset in bundle.lgir0.assets.items():
            lines.append(
                f"- `{asset_id}`: kind={asset.kind}, packaging={asset.packaging}, uri={asset.uri}, size_bytes={asset.size_bytes}, load={asset.load_strategy}"
            )
        lines.append("")
    if bundle.lgir0.skills:
        lines.append("### Skills")
        for skill_id, skill in bundle.lgir0.skills.items():
            lines.append(
                f"- `{skill_id}`: version={skill.version}, assets={skill.asset_ids}, tool_bindings={skill.tool_binding_ids}, subagents={skill.subagent_ids}, load={skill.load_strategy}"
            )
        lines.append("")
    if bundle.lgir0.tools:
        lines.append("### Tools")
        for tool_id, tool in bundle.lgir0.tools.items():
            lines.append(
                f"- `{tool_id}`: kind={tool.tool_kind}, executor={tool.executor_id}, assets={tool.required_asset_ids}, callable={tool.callable_ref}"
            )
        lines.append("")
    if bundle.lgir0.tool_bindings:
        lines.append("### Tool bindings")
        for binding_id, binding in bundle.lgir0.tool_bindings.items():
            lines.append(
                f"- `{binding_id}`: tool={binding.tool_id}, scope={binding.scope_kind}:{binding.scope_ref}, visibility={binding.visibility}, requires_skill={binding.requires_skill_id}"
            )
        lines.append("")
    if bundle.lgir0.subagents:
        lines.append("### Subagents")
        for agent_id, subagent in bundle.lgir0.subagents.items():
            lines.append(
                f"- `{agent_id}`: graph_ref={subagent.graph_ref}, skills={subagent.skill_ids}, tools={subagent.tool_binding_ids}, executor={subagent.executor_id}"
            )
        lines.append("")
    lines.append("### Edges")
    for edge in bundle.lgir0.edges:
        if edge.kind == "Static":
            lines.append(f"- Static: `{edge.source}` -> `{edge.target}`")
        elif edge.kind == "Join":
            lines.append(f"- Join: `{edge.sources}` -> `{edge.target}`")
        elif edge.kind == "Conditional":
            lines.append(
                f"- Conditional: `{edge.source}` -> {sorted(edge.path_map.values())}, returns={edge.returns}, router={edge.router_ref}"
            )
    lines.append("")
    lines.append("## LGIR-1")
    lines.append("")
    lines.append("### Channels")
    for channel_name, channel in bundle.lgir1.channels.items():
        lines.append(
            f"- `{channel_name}`: kind={channel.kind}, reserved={channel.reserved}, reducer={channel.reducer.reducer_id if channel.reducer else 'none'}"
        )
    lines.append("")
    lines.append("### Pregel nodes")
    for node_id, node in bundle.lgir1.nodes.items():
        lines.append(
            f"- `{node_id}`: reads={node.reads}, triggers={node.triggers}, static_writer_channels={node.writer_spec.declared_channels}, may_emit_send={node.writer_spec.may_emit_send}"
        )
    lines.append("")
    lines.append("## Static analysis")
    lines.append("")
    lines.append("### Read/write sets")
    for subject_id, rw in bundle.analysis.read_write.items():
        lines.append(
            f"- `{subject_id}`: reads={rw.read_set}, checkpoint_reads={rw.checkpoint_read_set}, task_inputs={rw.task_input_keys}, writes={rw.write_set}, source={rw.inferred_from}"
        )
    for subject_id, rw in bundle.analysis.route_read_write.items():
        lines.append(
            f"- `{subject_id}`: reads={rw.read_set}, sends={rw.send_targets}, payload_keys={rw.send_payload_keys}, source={rw.inferred_from}"
        )
    lines.append("")
    lines.append("### Reducers")
    for reducer in bundle.analysis.reducers:
        reducer_name = reducer.reducer.reducer_id if reducer.reducer else "none"
        lines.append(
            f"- `{reducer.key}`: reducer={reducer_name}, writers={reducer.writers}, parallel_writers={reducer.parallel_writers}, safe_parallel_merge={reducer.safe_parallel_merge}"
        )
    lines.append("")
    lines.append("### Fanout regions")
    for region in bundle.analysis.fanout_regions:
        lines.append(
            f"- source=`{region.fanout_source}`, map_nodes={region.map_nodes}, reduce_join={region.reduce_join}, payload_keys={region.send_payload_keys}"
        )
    if not bundle.analysis.fanout_regions:
        lines.append("- none")
    lines.append("")
    lines.append("### SCCs / loops")
    for loop in bundle.analysis.loops:
        lines.append(
            f"- `{loop.component_id}`: members={loop.members}, loop={loop.requires_loop_capable_orchestrator}, entry={loop.entry_nodes}, exit={loop.exit_nodes}, termination={loop.termination_style}"
        )
        if loop.cycle_edges:
            lines.append(f"  - cycle_edges: {loop.cycle_edges}")
        for note in loop.notes:
            lines.append(f"  - note: {note}")
    lines.append("")
    lines.append("### Side effects and cache")
    for node_id, side_effect in bundle.analysis.side_effects.items():
        lines.append(f"- `{node_id}`: purity={side_effect.purity}, domains={side_effect.effect_domains}")
    for cache in bundle.analysis.cache_analysis:
        lines.append(
            f"- cache `{cache.node_id}`: has_policy={cache.has_cache_policy}, safe={cache.safe_to_cache}, recommended_boundary={cache.recommended_boundary}, reason={cache.reason}"
        )
    if bundle.analysis.capability_analysis:
        lines.append("")
        lines.append("### Capabilities")
        for capability in bundle.analysis.capability_analysis:
            lines.append(
                f"- `{capability.subject_id}`: kind={capability.subject_kind}, executor={capability.executor_id}, skills={capability.loaded_skills}, tools={capability.reachable_tools}, externalized_assets={capability.externalized_assets}, interrupt={capability.requires_interrupt}"
            )
    if bundle.analysis.work_profiles:
        lines.append("")
        lines.append("### Work Profiles")
        for subject_id, profile in bundle.analysis.work_profiles.items():
            lines.append(
                f"- `{subject_id}`: body={profile.body_kind}, ops={profile.dominant_operations}, ratio={profile.work_to_overhead_ratio}, invocations={profile.observed_invocations}, peak_concurrency={profile.observed_peak_concurrency}, earned={profile.earned_partition}, hint={profile.granularity_hint}"
            )
            lines.append(
                f"  - work: static={profile.static_work_score}, payload={profile.payload_work_score}, intrinsic={profile.intrinsic_work_score}, overhead={profile.orchestration_overhead_score}"
            )
            lines.append(
                f"  - payload: avg_in={profile.avg_input_units}, max_in={profile.max_input_units}, avg_out={profile.avg_result_units}, max_out={profile.max_result_units}, expansion={profile.payload_expansion_ratio}"
            )
            if profile.earned_reasons:
                lines.append(f"  - earned because: {profile.earned_reasons}")
            if profile.unearned_reasons:
                lines.append(f"  - has not earned isolation because: {profile.unearned_reasons}")
            if profile.recommended_batch_size:
                lines.append(f"  - batching hint: {profile.recommended_batch_size} items per invocation")
            for reason in profile.theoretical_basis:
                lines.append(f"  - theory: {reason}")
    if bundle.analysis.warnings:
        lines.append("")
        lines.append("### Warnings")
        for warning in bundle.analysis.warnings:
            lines.append(f"- {warning}")
    if bundle.analysis.notes:
        lines.append("")
        lines.append("### Notes")
        for note in bundle.analysis.notes:
            lines.append(f"- {note}")
    lines.append("")
    lines.append("## LGIR-2")
    lines.append("")
    for decision in bundle.lgir2.fusion_decisions:
        lines.append(
            f"- fusion `{decision.edge}`: accepted={decision.accepted}, cost_before={decision.estimated_cost_before}, cost_after={decision.estimated_cost_after}, rules={decision.rule_results}"
        )
    lines.append("")
    for partition_id, partition in bundle.lgir2.partitions.items():
        lines.append(
            f"- `{partition_id}`: members={partition.members}, routes={partition.attached_routes}, loop_component={partition.loop_component}, checkpoint_reads={partition.checkpoint_read_set}, task_inputs={partition.task_input_keys}, writes={partition.write_set}, emits_send={partition.emits_send}, cost={partition.estimated_cost}"
        )
        if partition.asset_ids or partition.tool_binding_ids or partition.subagent_ids or partition.executor_id:
            lines.append(
                f"  - assets={partition.asset_ids}, tools={partition.tool_binding_ids}, subagents={partition.subagent_ids}, executor={partition.executor_id}"
            )
    if bundle.lgir2.partition_work_profiles:
        lines.append("")
        lines.append("### Earned Partitioning")
        for partition_id, profile in bundle.lgir2.partition_work_profiles.items():
            lines.append(
                f"- `{partition_id}`: body={profile.body_kind}, ratio={profile.work_to_overhead_ratio}, earned={profile.earned_partition}, hint={profile.granularity_hint}"
            )
            lines.append(
                f"  - partition work: intrinsic={profile.intrinsic_work_score}, overhead={profile.orchestration_overhead_score}, invocations={profile.observed_invocations}, peak_concurrency={profile.observed_peak_concurrency}"
            )
            if profile.earned_reasons:
                lines.append(f"  - earned because: {profile.earned_reasons}")
            if profile.unearned_reasons:
                lines.append(f"  - has not earned more granularity because: {profile.unearned_reasons}")
            if profile.recommended_batch_size:
                lines.append(f"  - recommended fanout batch size: {profile.recommended_batch_size}")
            for reason in profile.theoretical_basis:
                lines.append(f"  - theory: {reason}")
    if bundle.lgir2.loop_clusters:
        lines.append("")
        lines.append("### Loop clusters")
        for component_id, partitions in bundle.lgir2.loop_clusters.items():
            lines.append(f"- `{component_id}` -> {partitions}")
    lines.append("")
    lines.append("## SRV-Plan")
    lines.append("")
    lines.append(f"- mode: `{bundle.srv_plan.orchestration.get('mode')}`")
    lines.append(f"- checkpoint store: `{bundle.srv_plan.persistence.get('checkpoint_store')}`")
    lines.append(f"- worker count: `{len(bundle.srv_plan.compute.get('workers', {}))}`")
    resource_summary = bundle.srv_plan.compute.get("resource_summary", {})
    if resource_summary:
        lines.append(f"- resource optimizer: `{resource_summary.get('strategy')}`")
        lines.append(f"- SLO profile: `{resource_summary.get('slo_profile_id')}`")
        lines.append(f"- total compute envelope: `{resource_summary.get('total_compute_mb')}MB`")
    routing_nodes = bundle.srv_plan.orchestration.get("routing_nodes", [])
    if routing_nodes:
        lines.append(f"- routing nodes: `{routing_nodes}`")
    distributed_map = bundle.srv_plan.orchestration.get("distributed_map_partitions", [])
    if distributed_map:
        lines.append(f"- distributed map partitions: `{distributed_map}`")
    interrupt_partitions = bundle.srv_plan.orchestration.get("interrupt_partitions", [])
    if interrupt_partitions:
        lines.append(f"- interrupt partitions: `{interrupt_partitions}`")
    subagent_partitions = bundle.srv_plan.orchestration.get("subagent_partitions", [])
    if subagent_partitions:
        lines.append(f"- subagent partitions: `{subagent_partitions}`")
    external_asset_partitions = bundle.srv_plan.orchestration.get("external_asset_partitions", [])
    if external_asset_partitions:
        lines.append(f"- external asset partitions: `{external_asset_partitions}`")
    batching_candidates = bundle.srv_plan.orchestration.get("batching_candidates", {})
    if batching_candidates:
        lines.append(f"- batching candidates: `{batching_candidates}`")
    if bundle.srv_plan.assets:
        lines.append(f"- asset plan keys: `{sorted(bundle.srv_plan.assets)}`")
    if bundle.srv_plan.tooling:
        lines.append(f"- tooling plan keys: `{sorted(bundle.srv_plan.tooling)}`")
    if bundle.srv_plan.agent_runtime:
        lines.append(f"- agent runtime keys: `{sorted(bundle.srv_plan.agent_runtime)}`")
    lines.append("")
    lines.append("### Workers")
    for partition_id, worker in bundle.srv_plan.compute.get("workers", {}).items():
        lines.append(
            f"- `{partition_id}` -> `{worker['lambda_name']}`: memory={worker['memory_mb']}MB, timeout={worker['timeout_sec']}s, concurrency={worker['concurrency_limit']}, total_compute={worker.get('total_compute_mb')}MB"
        )
        for note in worker.get("notes", [])[:4]:
            lines.append(f"  - note: {note}")
    lines.append("")
    lines.append("### Memory optimization")
    for partition_id, worker in bundle.srv_plan.compute.get("workers", {}).items():
        optimization = worker.get("resource_optimization", {})
        if not optimization:
            continue
        workload = _field(optimization, "workload", {})
        selected = _field(optimization, "selected_memory_mb")
        initial = _field(optimization, "initial_memory_mb")
        candidate_text = "; ".join(
            (
                f"{_field(candidate, 'memory_mb')}MB:"
                f"J={_field(candidate, 'objective_score')},"
                f"p95={_field(candidate, 'estimated_p95_latency_ms')}ms,"
                f"cost={_field(candidate, 'estimated_cost_units')},"
                f"ok={_field(candidate, 'feasible')}"
            )
            for candidate in _field(optimization, "candidates", [])
        )
        lines.append(
            f"- `{partition_id}`: initial={initial}MB, selected={selected}MB, timeout={_field(optimization, 'selected_timeout_sec')}s, concurrency={_field(optimization, 'selected_concurrency_limit')}, total_compute={_field(optimization, 'total_compute_mb')}MB"
        )
        lines.append(
            f"  - workload: profile={workload.get('profile')}, invocations={workload.get('observed_invocations')}, peak_concurrency={workload.get('observed_peak_concurrency')}, fanout_width={workload.get('fanout_width')}, work_units={workload.get('work_units')}"
        )
        lines.append(f"  - reason: {_field(optimization, 'reason')}")
        lines.append(f"  - candidates: {candidate_text}")
    lines.append("")
    lines.append("### Planner outline")
    for step in bundle.srv_plan.planner_outline:
        lines.append(f"- {step}")
    if bundle.srv_plan.loop_plans:
        lines.append("")
        lines.append("### Loop plans")
        for loop_plan in bundle.srv_plan.loop_plans:
            lines.append(
                f"- `{loop_plan['component_id']}`: partitions={loop_plan['partitions']}, termination={loop_plan['termination_style']}, scheduler_hint={loop_plan['scheduler_hint']}"
            )
    router_nodes = bundle.srv_plan.agent_runtime.get("router_nodes", {})
    if router_nodes:
        lines.append("")
        lines.append("### Router runtime")
        for node_id, router in router_nodes.items():
            lines.append(
                f"- `{node_id}`: branches={router.get('branch_names')}, targets={router.get('send_targets')}, fanout={router.get('fanout')}, payload_keys={router.get('payload_keys')}"
            )
    delegation_nodes = bundle.srv_plan.agent_runtime.get("delegation_nodes", {})
    if delegation_nodes:
        lines.append("")
        lines.append("### Delegation runtime")
        for node_id, delegation in delegation_nodes.items():
            lines.append(
                f"- `{node_id}`: subagent={delegation.get('subagent_id')}, executor={delegation.get('executor_id')}, tools={delegation.get('reachable_tools')}, interrupt={delegation.get('requires_interrupt')}"
            )
    subagents = bundle.srv_plan.agent_runtime.get("subagents", {})
    if subagents:
        lines.append("")
        lines.append("### Subagent runtimes")
        for agent_id, subagent in subagents.items():
            lines.append(
                f"- `{agent_id}`: executor={subagent.get('executor_id')}, inherited={subagent.get('inheritance')}, tools={subagent.get('reachable_tools')}, delegated_from={subagent.get('delegated_from_nodes')}"
            )
    if bundle.runtime_trace:
        lines.append("")
        lines.append("## Baseline runtime trace")
        lines.append("")
        for step in bundle.runtime_trace:
            task_names = ", ".join(f"{task.task_kind}:{task.node_id}" for task in step.tasks)
            lines.append(f"- step {step.step}: {task_names}")
            lines.append(f"  - writes: {step.grouped_writes}")
            lines.append(f"  - state_after: {step.state_after_step}")
            for note in step.notes:
                lines.append(f"  - note: {note}")
    lines.append("")
    lines.append("## Next work")
    for item in bundle.next_steps:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_artifacts(bundle: CompilationBundle, output_dir: Path) -> Path:
    ensure_directory(output_dir)
    write_json(output_dir / "lgir0.json", bundle.lgir0)
    write_json(output_dir / "lgir1.json", bundle.lgir1)
    write_json(output_dir / "analysis.json", bundle.analysis)
    write_json(output_dir / "lgir2.json", bundle.lgir2)
    write_json(output_dir / "srv_plan.json", bundle.srv_plan)
    write_json(output_dir / "runtime_trace.json", bundle.runtime_trace)
    report_path = output_dir / "report.md"
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return report_path
