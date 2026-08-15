from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from orqis.compiler.passes import compile_graph, load_object
from orqis.compiler.resource_optimizer import DEFAULT_SLO_PROFILE_ID, SLO_PROFILES
from orqis.compiler.report import write_artifacts


RESOURCE_POLICIES = ("baseline", "om", "om2")


def _add_resource_policy_arg(parser: argparse.ArgumentParser, *, default: str = "om2") -> None:
    parser.add_argument(
        "--resource-policy",
        choices=RESOURCE_POLICIES,
        default=default,
        help="Resource policy: baseline copies partition hints, om uses fixed-weight memory optimization, om2 adds profile guardrails.",
    )


def _add_slo_profile_arg(parser: argparse.ArgumentParser, *, default: str = DEFAULT_SLO_PROFILE_ID) -> None:
    parser.add_argument(
        "--slo-profile",
        choices=tuple(sorted(SLO_PROFILES)),
        default=default,
        help="Named SLO profile that controls latency, headroom, and failure-risk constraints during memory optimization.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a compiled LangGraph Pregel graph.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    example = subparsers.add_parser("analyze-example", help="Run the built-in document summariser example.")
    example.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/doc_summariser"),
        help="Directory for generated IR, analysis, and report artifacts.",
    )
    _add_resource_policy_arg(example)
    _add_slo_profile_arg(example)

    toy_example = subparsers.add_parser(
        "analyze-doc-toy-example",
        help="Run the built-in toy document summariser example without remote/LLM metadata.",
    )
    toy_example.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/doc_summariser_toy"),
        help="Directory for generated IR, analysis, and report artifacts.",
    )
    _add_resource_policy_arg(toy_example)
    _add_slo_profile_arg(toy_example)

    remote_example = subparsers.add_parser(
        "analyze-doc-remote-example",
        help="Run the built-in remote document summariser example with LLM metadata.",
    )
    remote_example.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/doc_summariser_remote"),
        help="Directory for generated IR, analysis, and report artifacts.",
    )
    _add_resource_policy_arg(remote_example)
    _add_slo_profile_arg(remote_example)

    loop_example = subparsers.add_parser("analyze-loop-example", help="Run the built-in iterative loop example.")
    loop_example.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/iterative_counter"),
        help="Directory for generated IR, analysis, and report artifacts.",
    )
    _add_resource_policy_arg(loop_example)
    _add_slo_profile_arg(loop_example)

    benchmark_examples = subparsers.add_parser(
        "analyze-benchmark-examples",
        help="Run the toy doc summariser, remote doc summariser, and iterative counter examples together.",
    )
    benchmark_examples.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/compiler_benchmark_examples"),
        help="Root directory where per-example artifact folders will be written.",
    )
    _add_resource_policy_arg(benchmark_examples)
    _add_slo_profile_arg(benchmark_examples)

    benchmark_profiles = subparsers.add_parser(
        "analyze-benchmark-memory-profiles",
        help="Run the toy doc summariser, remote doc summariser, and iterative counter examples for baseline, OM, and OM2.",
    )
    benchmark_profiles.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/compiler_benchmark_memory_profiles"),
        help="Root directory where per-example and per-policy artifact folders will be written.",
    )
    _add_slo_profile_arg(benchmark_profiles)

    skills_example = subparsers.add_parser("analyze-skills-example", help="Run the built-in skills agent example.")
    skills_example.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/skills"),
        help="Directory for generated IR, analysis, and report artifacts.",
    )
    _add_resource_policy_arg(skills_example)
    _add_slo_profile_arg(skills_example)

    router_example = subparsers.add_parser(
        "analyze-router-example",
        help="Run the built-in agentic router workflow example.",
    )
    router_example.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/service_desk_router"),
        help="Directory for generated IR, analysis, and report artifacts.",
    )
    _add_resource_policy_arg(router_example)
    _add_slo_profile_arg(router_example)

    subagents_example = subparsers.add_parser(
        "analyze-subagents-example",
        help="Run the built-in subagent incident-response workflow example.",
    )
    subagents_example.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/incident_response_swarm"),
        help="Directory for generated IR, analysis, and report artifacts.",
    )
    _add_resource_policy_arg(subagents_example)
    _add_slo_profile_arg(subagents_example)

    module = subparsers.add_parser("analyze-module", help="Analyze a graph object or factory from a module path.")
    module.add_argument("graph", help="Module spec in the form package.module:attr")
    module.add_argument(
        "--sample",
        help="Optional sample-input object or zero-arg factory in the form package.module:attr",
    )
    module.add_argument(
        "--graph-id",
        help="Optional graph id override for IR and output naming.",
    )
    module.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated IR, analysis, and report artifacts.",
    )
    _add_resource_policy_arg(module)
    _add_slo_profile_arg(module)
    return parser


def _resolve_sample_input(spec: str | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    obj = load_object(spec)
    return obj() if callable(obj) else obj


def _run_builtin(
    graph_spec: str,
    sample_spec: str,
    output_dir: Path,
    *,
    graph_id: str | None = None,
    resource_policy: str = "om2",
    slo_profile: str = DEFAULT_SLO_PROFILE_ID,
) -> int:
    graph_obj = load_object(graph_spec)
    sample_input = load_object(sample_spec)()
    bundle = compile_graph(
        graph_obj,
        graph_id=graph_id,
        sample_input=sample_input,
        resource_policy=resource_policy,
        slo_profile=slo_profile,
    )
    report_path = write_artifacts(bundle, output_dir)
    print(f"Analyzed `{bundle.graph_id}` with `{resource_policy}` resource policy.")
    print(f"SLO profile: `{slo_profile}`.")
    print(f"Artifacts written to `{output_dir}`.")
    print(f"Report: `{report_path}`")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze-example":
        return _run_builtin(
            "orqis.examples.doc_summariser:build_graph",
            "orqis.examples.doc_summariser:get_sample_input",
            args.output_dir,
            resource_policy=args.resource_policy,
            slo_profile=args.slo_profile,
        )

    if args.command == "analyze-doc-toy-example":
        return _run_builtin(
            "orqis.examples.doc_summariser:build_toy_graph",
            "orqis.examples.doc_summariser:get_toy_sample_input",
            args.output_dir,
            resource_policy=args.resource_policy,
            slo_profile=args.slo_profile,
        )

    if args.command == "analyze-doc-remote-example":
        return _run_builtin(
            "orqis.examples.doc_summariser:build_remote_graph",
            "orqis.examples.doc_summariser:get_remote_sample_input",
            args.output_dir,
            resource_policy=args.resource_policy,
            slo_profile=args.slo_profile,
        )

    if args.command == "analyze-loop-example":
        return _run_builtin(
            "orqis.examples.iterative_counter:build_graph",
            "orqis.examples.iterative_counter:get_sample_input",
            args.output_dir,
            resource_policy=args.resource_policy,
            slo_profile=args.slo_profile,
        )

    if args.command == "analyze-skills-example":
        return _run_builtin(
            "orqis.examples.skills:build_graph",
            "orqis.examples.skills:get_sample_input",
            args.output_dir,
            graph_id="skills",
            resource_policy=args.resource_policy,
            slo_profile=args.slo_profile,
        )

    if args.command == "analyze-router-example":
        return _run_builtin(
            "orqis.examples.agentic.router:build_graph",
            "orqis.examples.agentic.router:get_sample_input",
            args.output_dir,
            resource_policy=args.resource_policy,
            slo_profile=args.slo_profile,
        )

    if args.command == "analyze-subagents-example":
        return _run_builtin(
            "orqis.examples.agentic.subagents:build_graph",
            "orqis.examples.agentic.subagents:get_sample_input",
            args.output_dir,
            resource_policy=args.resource_policy,
            slo_profile=args.slo_profile,
        )

    if args.command == "analyze-benchmark-examples":
        root = args.output_root
        runs = [
            (
                "orqis.examples.doc_summariser:build_toy_graph",
                "orqis.examples.doc_summariser:get_toy_sample_input",
                root / "doc_summariser_toy",
            ),
            (
                "orqis.examples.doc_summariser:build_remote_graph",
                "orqis.examples.doc_summariser:get_remote_sample_input",
                root / "doc_summariser_remote",
            ),
            (
                "orqis.examples.iterative_counter:build_graph",
                "orqis.examples.iterative_counter:get_sample_input",
                root / "iterative_counter",
            ),
        ]
        for graph_spec, sample_spec, output_dir in runs:
            _run_builtin(
                graph_spec,
                sample_spec,
                output_dir,
                resource_policy=args.resource_policy,
                slo_profile=args.slo_profile,
            )
        return 0

    if args.command == "analyze-benchmark-memory-profiles":
        root = args.output_root
        runs = [
            (
                "doc_summariser_toy",
                "orqis.examples.doc_summariser:build_toy_graph",
                "orqis.examples.doc_summariser:get_toy_sample_input",
                None,
            ),
            (
                "doc_summariser_remote",
                "orqis.examples.doc_summariser:build_remote_graph",
                "orqis.examples.doc_summariser:get_remote_sample_input",
                None,
            ),
            (
                "iterative_counter",
                "orqis.examples.iterative_counter:build_graph",
                "orqis.examples.iterative_counter:get_sample_input",
                None,
            ),
        ]
        for example_id, graph_spec, sample_spec, graph_id in runs:
            for resource_policy in RESOURCE_POLICIES:
                _run_builtin(
                    graph_spec,
                    sample_spec,
                    root / example_id / resource_policy,
                    graph_id=graph_id,
                    resource_policy=resource_policy,
                    slo_profile=args.slo_profile,
                )
        return 0

    graph_obj = load_object(args.graph)
    sample_input = _resolve_sample_input(args.sample)
    bundle = compile_graph(
        graph_obj,
        graph_id=args.graph_id,
        sample_input=sample_input,
        resource_policy=args.resource_policy,
        slo_profile=args.slo_profile,
    )
    output_dir = args.output_dir or Path("artifacts") / bundle.graph_id
    report_path = write_artifacts(bundle, output_dir)
    print(f"Analyzed `{bundle.graph_id}` with `{args.resource_policy}` resource policy.")
    print(f"SLO profile: `{args.slo_profile}`.")
    print(f"Artifacts written to `{output_dir}`.")
    print(f"Report: `{report_path}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
