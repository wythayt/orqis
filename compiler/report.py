from __future__ import annotations

from pathlib import Path

from orqis.compiler.ir import CompilationBundle
from orqis.compiler.utils import ensure_directory, to_jsonable, write_json


def render_report(bundle: CompilationBundle) -> str:
    lines: list[str] = []
    lines.append(f"# Compilation report: `{bundle.graph_id}`")
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
            f"- `{loop.component_id}`: members={loop.members}, loop={loop.requires_loop_capable_orchestrator}"
        )
    lines.append("")
    lines.append("### Side effects and cache")
    for node_id, side_effect in bundle.analysis.side_effects.items():
        lines.append(f"- `{node_id}`: purity={side_effect.purity}, domains={side_effect.effect_domains}")
    for cache in bundle.analysis.cache_analysis:
        lines.append(
            f"- cache `{cache.node_id}`: has_policy={cache.has_cache_policy}, safe={cache.safe_to_cache}, recommended_boundary={cache.recommended_boundary}, reason={cache.reason}"
        )
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
            f"- `{partition_id}`: members={partition.members}, routes={partition.attached_routes}, checkpoint_reads={partition.checkpoint_read_set}, task_inputs={partition.task_input_keys}, writes={partition.write_set}, emits_send={partition.emits_send}, cost={partition.estimated_cost}"
        )
    lines.append("")
    lines.append("## SRV-Plan")
    lines.append("")
    lines.append(f"- mode: `{bundle.srv_plan.orchestration.get('mode')}`")
    lines.append(f"- checkpoint store: `{bundle.srv_plan.persistence.get('checkpoint_store')}`")
    lines.append(f"- worker count: `{len(bundle.srv_plan.compute.get('workers', {}))}`")
    lines.append("")
    lines.append("### Workers")
    for partition_id, worker in bundle.srv_plan.compute.get("workers", {}).items():
        lines.append(
            f"- `{partition_id}` -> `{worker['lambda_name']}`: memory={worker['memory_mb']}MB, timeout={worker['timeout_sec']}s, concurrency={worker['concurrency_limit']}"
        )
    lines.append("")
    lines.append("### Planner outline")
    for step in bundle.srv_plan.planner_outline:
        lines.append(f"- {step}")
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
