from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from orqis.compiler.passes import compile_graph, load_object
from orqis.compiler.report import write_artifacts


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
    return parser


def _resolve_sample_input(spec: str | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    obj = load_object(spec)
    return obj() if callable(obj) else obj


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze-example":
        graph_obj = load_object("orqis.examples.doc_summariser:build_graph")
        sample_input = load_object("orqis.examples.doc_summariser:get_sample_input")()
        bundle = compile_graph(graph_obj, sample_input=sample_input)
        report_path = write_artifacts(bundle, args.output_dir)
        print(f"Analyzed `{bundle.graph_id}`.")
        print(f"Artifacts written to `{args.output_dir}`.")
        print(f"Report: `{report_path}`")
        return 0

    graph_obj = load_object(args.graph)
    sample_input = _resolve_sample_input(args.sample)
    bundle = compile_graph(graph_obj, graph_id=args.graph_id, sample_input=sample_input)
    output_dir = args.output_dir or Path("artifacts") / bundle.graph_id
    report_path = write_artifacts(bundle, output_dir)
    print(f"Analyzed `{bundle.graph_id}`.")
    print(f"Artifacts written to `{output_dir}`.")
    print(f"Report: `{report_path}`")
    return 0
