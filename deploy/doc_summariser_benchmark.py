from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PLAN_PATH = Path("artifacts/doc_summariser_memory_opt/srv_plan.json")
DEFAULT_RESULTS_PATH = Path("artifacts/doc_summariser_memory_opt/deployment_metrics.json")
DEFAULT_PREFIX = "orqis-doc-summariser"
AWS_LAMBDA_GB_SECOND_USD = 0.0000166667
AWS_LAMBDA_REQUEST_USD = 0.20 / 1_000_000

HANDLER_SOURCE = r'''
from __future__ import annotations

import hashlib
import json
import os
import time

PARTITION_ID = os.environ["ORQIS_PARTITION_ID"]
VARIANT = os.environ["ORQIS_VARIANT"]
CPU_ROUNDS = int(os.environ.get("ORQIS_CPU_ROUNDS", "0"))
ALLOCATE_MB = int(os.environ.get("ORQIS_ALLOCATE_MB", "0"))


def _burn_cpu(units: int) -> str:
    rounds = max(0, CPU_ROUNDS) * max(1, units)
    digest = b"orqis"
    for index in range(rounds):
        digest = hashlib.sha256(digest + str(index).encode("ascii")).digest()
    return digest.hex()[:16]


def _allocate_memory() -> int:
    if ALLOCATE_MB <= 0:
        return 0
    block = bytearray(ALLOCATE_MB * 1024 * 1024)
    for index in range(0, len(block), 4096):
        block[index] = index % 251
    return len(block)


def _normalise_text(text: str) -> str:
    return " ".join(text.split())


def _split_text(text: str, chunk_size: int) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def lambda_handler(event, context):
    started = time.perf_counter()
    state = dict(event.get("state") or {})
    input_slice = dict(event.get("input_slice") or {})
    chunk_size = int(event.get("chunk_size", 24))
    state.update(input_slice)
    memory_touched = _allocate_memory()

    if PARTITION_ID == "p_ingest_split_fanout":
        text = _normalise_text(state.get("text", ""))
        chunks = _split_text(text, chunk_size)
        _burn_cpu(max(1, len(text) // 24))
        result = {
            "writes": {"text": text, "chunks": chunks},
            "sends": [
                {
                    "node": "summarise_chunk",
                    "input_slice": {"doc_id": state.get("doc_id", ""), "chunk": chunk},
                }
                for chunk in chunks
            ],
        }
    elif PARTITION_ID == "p_summarise_chunk":
        chunk = state.get("chunk", "")
        _burn_cpu(max(1, len(chunk) // 8))
        result = {"writes": {"chunk_summaries": [f"summary<{chunk[:10]}>"]}, "sends": []}
    elif PARTITION_ID == "p_aggregate":
        summaries = state.get("chunk_summaries", [])
        _burn_cpu(max(1, len(summaries)))
        result = {"writes": {"final_summary": " | ".join(summaries)}, "sends": []}
    else:
        raise ValueError(f"unknown partition {PARTITION_ID}")

    result["meta"] = {
        "variant": VARIANT,
        "partition_id": PARTITION_ID,
        "handler_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "memory_touched_bytes": memory_touched,
    }
    return result
'''


@dataclass
class WorkerConfig:
    partition_id: str
    function_name: str
    memory_mb: int
    timeout_sec: int
    concurrency_limit: int | None
    source: str


@dataclass
class VariantConfig:
    name: str
    workers: dict[str, WorkerConfig] = field(default_factory=dict)


@dataclass
class InvokeMetric:
    variant: str
    partition_id: str
    status_code: int
    function_error: str | None
    duration_ms: float | None
    billed_duration_ms: int | None
    init_duration_ms: float | None
    max_memory_used_mb: int | None
    configured_memory_mb: int
    estimated_cost_usd: float | None
    payload: dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CommandError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy and benchmark baseline vs memory-optimized doc_summariser Lambda workers.",
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH, help="SRV plan JSON to read.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Lambda function name prefix.")
    parser.add_argument("--region", help="AWS region to pass to the AWS CLI.")
    parser.add_argument("--profile", help="AWS profile to pass to the AWS CLI.")
    parser.add_argument("--aws-cli", default="aws", help="AWS CLI executable.")
    parser.add_argument("--python-runtime", default="python3.11", help="Lambda Python runtime.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Show the two deployable variants.")
    plan.set_defaults(func=cmd_plan)

    deploy = subparsers.add_parser("deploy", help="Create or update Lambda workers for both variants.")
    deploy.add_argument("--role-arn", required=True, help="IAM role ARN used by Lambda functions.")
    deploy.add_argument("--cpu-rounds", type=int, default=0, help="Deterministic CPU work multiplier in handlers.")
    deploy.add_argument("--allocate-mb", type=int, default=0, help="Temporary memory to touch in each invocation.")
    deploy.add_argument(
        "--apply-reserved-concurrency",
        action="store_true",
        help="Apply reserved concurrency to Lambda functions. Disabled by default for small AWS accounts.",
    )
    deploy.set_defaults(func=cmd_deploy)

    invoke = subparsers.add_parser("invoke", help="Invoke already deployed variants and write metrics.")
    add_invoke_args(invoke)
    invoke.set_defaults(func=cmd_invoke)

    benchmark = subparsers.add_parser("benchmark", help="Deploy, then invoke both variants.")
    benchmark.add_argument("--role-arn", required=True, help="IAM role ARN used by Lambda functions.")
    benchmark.add_argument("--cpu-rounds", type=int, default=0, help="Deterministic CPU work multiplier in handlers.")
    benchmark.add_argument("--allocate-mb", type=int, default=0, help="Temporary memory to touch in each invocation.")
    benchmark.add_argument(
        "--apply-reserved-concurrency",
        action="store_true",
        help="Apply reserved concurrency to Lambda functions. Disabled by default for small AWS accounts.",
    )
    add_invoke_args(benchmark)
    benchmark.set_defaults(func=cmd_benchmark)

    cleanup = subparsers.add_parser("cleanup", help="Delete both variants' Lambda functions.")
    cleanup.set_defaults(func=cmd_cleanup)
    return parser


def add_invoke_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs", type=int, default=5, help="Workflow runs per variant.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup workflow runs per variant.")
    parser.add_argument("--chunk-size", type=int, default=24, help="Chunk size used by the deployed split worker.")
    parser.add_argument("--text-repeat", type=int, default=1, help="Repeat the sample text to create wider fanout.")
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULTS_PATH, help="Metrics JSON output path.")
    parser.add_argument("--sleep-sec", type=float, default=0.0, help="Sleep between workflow runs.")


def cmd_plan(args: argparse.Namespace) -> int:
    variants = load_variants(args.plan, args.prefix)
    print(json.dumps(variants_to_json(variants), indent=2))
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    variants = load_variants(args.plan, args.prefix)
    zip_path = build_lambda_zip()
    try:
        for variant in variants.values():
            for worker in variant.workers.values():
                deploy_function(args, worker, zip_path)
    finally:
        zip_path.unlink(missing_ok=True)
    return 0


def cmd_invoke(args: argparse.Namespace) -> int:
    variants = load_variants(args.plan, args.prefix)
    result = benchmark_variants(args, variants)
    write_json(args.result, result)
    print(f"metrics written to {args.result}")
    print(json.dumps(result["summary"], indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    deploy_status = cmd_deploy(args)
    if deploy_status != 0:
        return deploy_status
    return cmd_invoke(args)


def cmd_cleanup(args: argparse.Namespace) -> int:
    variants = load_variants(args.plan, args.prefix)
    for variant in variants.values():
        for worker in variant.workers.values():
            run_aws(
                args,
                ["lambda", "delete-function", "--function-name", worker.function_name],
                allow_failure=True,
                capture=True,
            )
            print(f"deleted if present: {worker.function_name}")
    return 0


def load_variants(plan_path: Path, prefix: str) -> dict[str, VariantConfig]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    workers = plan["compute"]["workers"]
    variants = {
        "baseline": VariantConfig(name="baseline"),
        "memory_opt": VariantConfig(name="memory_opt"),
    }
    for partition_id, worker in workers.items():
        optimization = worker.get("resource_optimization") or {}
        baseline_memory = int(optimization.get("initial_memory_mb") or worker["memory_mb"])
        baseline_timeout = int(optimization.get("initial_timeout_sec") or worker["timeout_sec"])
        baseline_concurrency = optimization.get("initial_concurrency_limit", worker.get("concurrency_limit"))
        opt_memory = int(worker["memory_mb"])
        opt_timeout = int(worker["timeout_sec"])
        opt_concurrency = worker.get("concurrency_limit")
        variants["baseline"].workers[partition_id] = WorkerConfig(
            partition_id=partition_id,
            function_name=function_name(prefix, "baseline", partition_id),
            memory_mb=baseline_memory,
            timeout_sec=baseline_timeout,
            concurrency_limit=baseline_concurrency,
            source="initial_langgraph_metadata",
        )
        variants["memory_opt"].workers[partition_id] = WorkerConfig(
            partition_id=partition_id,
            function_name=function_name(prefix, "memory-opt", partition_id),
            memory_mb=opt_memory,
            timeout_sec=opt_timeout,
            concurrency_limit=opt_concurrency,
            source="resource_optimizer_selected",
        )
    return variants


def function_name(prefix: str, variant: str, partition_id: str) -> str:
    return f"{prefix}-{variant}-{partition_id}".replace("_", "-")


def variants_to_json(variants: dict[str, VariantConfig]) -> dict[str, Any]:
    return {
        variant_name: {
            partition_id: {
                "function_name": worker.function_name,
                "memory_mb": worker.memory_mb,
                "timeout_sec": worker.timeout_sec,
                "concurrency_limit": worker.concurrency_limit,
                "source": worker.source,
            }
            for partition_id, worker in variant.workers.items()
        }
        for variant_name, variant in variants.items()
    }


def build_lambda_zip() -> Path:
    tmp = tempfile.NamedTemporaryFile(prefix="orqis-doc-summariser-", suffix=".zip", delete=False)
    tmp.close()
    path = Path(tmp.name)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("handler.py", HANDLER_SOURCE)
    return path


def deploy_function(args: argparse.Namespace, worker: WorkerConfig, zip_path: Path) -> None:
    exists = lambda_exists(args, worker.function_name)
    env = {
        "Variables": {
            "ORQIS_PARTITION_ID": worker.partition_id,
            "ORQIS_VARIANT": "baseline" if "-baseline-" in worker.function_name else "memory_opt",
            "ORQIS_CPU_ROUNDS": str(args.cpu_rounds),
            "ORQIS_ALLOCATE_MB": str(args.allocate_mb),
        }
    }
    if exists:
        run_aws(
            args,
            [
                "lambda",
                "update-function-code",
                "--function-name",
                worker.function_name,
                "--zip-file",
                f"fileb://{zip_path}",
            ],
        )
        wait_function_updated(args, worker.function_name)
        run_aws(
            args,
            [
                "lambda",
                "update-function-configuration",
                "--function-name",
                worker.function_name,
                "--runtime",
                args.python_runtime,
                "--handler",
                "handler.lambda_handler",
                "--memory-size",
                str(worker.memory_mb),
                "--timeout",
                str(worker.timeout_sec),
                "--environment",
                json.dumps(env),
            ],
        )
        wait_function_updated(args, worker.function_name)
        print(f"updated {worker.function_name}")
    else:
        run_aws(
            args,
            [
                "lambda",
                "create-function",
                "--function-name",
                worker.function_name,
                "--runtime",
                args.python_runtime,
                "--role",
                args.role_arn,
                "--handler",
                "handler.lambda_handler",
                "--zip-file",
                f"fileb://{zip_path}",
                "--memory-size",
                str(worker.memory_mb),
                "--timeout",
                str(worker.timeout_sec),
                "--environment",
                json.dumps(env),
            ],
        )
        wait_function_updated(args, worker.function_name)
        print(f"created {worker.function_name}")

    configure_reserved_concurrency(args, worker)


def lambda_exists(args: argparse.Namespace, function: str) -> bool:
    result = run_aws(
        args,
        ["lambda", "get-function", "--function-name", function],
        allow_failure=True,
        capture=True,
    )
    return result.returncode == 0


def configure_reserved_concurrency(args: argparse.Namespace, worker: WorkerConfig) -> None:
    if not getattr(args, "apply_reserved_concurrency", False):
        return
    if worker.concurrency_limit is None:
        run_aws(
            args,
            ["lambda", "delete-function-concurrency", "--function-name", worker.function_name],
            allow_failure=True,
            capture=True,
        )
        return
    run_aws(
        args,
        [
            "lambda",
            "put-function-concurrency",
            "--function-name",
            worker.function_name,
            "--reserved-concurrent-executions",
            str(worker.concurrency_limit),
        ],
    )


def wait_function_updated(args: argparse.Namespace, function: str) -> None:
    run_aws(args, ["lambda", "wait", "function-updated", "--function-name", function])


def benchmark_variants(args: argparse.Namespace, variants: dict[str, VariantConfig]) -> dict[str, Any]:
    all_results: list[dict[str, Any]] = []
    for variant in variants.values():
        for index in range(args.warmup_runs):
            run_workflow(args, variant, run_id=f"warmup-{index}", collect=False)
        for index in range(args.runs):
            run_result = run_workflow(args, variant, run_id=f"run-{index}", collect=True)
            all_results.append(run_result)
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)
    return {
        "plan": str(args.plan),
        "prefix": args.prefix,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "chunk_size": args.chunk_size,
        "text_repeat": args.text_repeat,
        "results": all_results,
        "summary": summarize_results(all_results),
    }


def run_workflow(
    args: argparse.Namespace,
    variant: VariantConfig,
    *,
    run_id: str,
    collect: bool,
) -> dict[str, Any]:
    state = sample_state(args.text_repeat)
    ingest_metric, ingest_payload = invoke_worker(
        args,
        variant.workers["p_ingest_split_fanout"],
        {"state": state, "chunk_size": args.chunk_size},
    )
    writes = ingest_payload["writes"]
    state.update(writes)
    sends = ingest_payload["sends"]

    max_workers = variant.workers["p_summarise_chunk"].concurrency_limit or len(sends) or 1
    max_workers = max(1, min(max_workers, len(sends) or 1))
    map_metrics: list[InvokeMetric] = []
    summaries: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                invoke_worker,
                args,
                variant.workers["p_summarise_chunk"],
                {"input_slice": send["input_slice"], "chunk_size": args.chunk_size},
            )
            for send in sends
        ]
        for future in concurrent.futures.as_completed(futures):
            metric, payload = future.result()
            map_metrics.append(metric)
            summaries.extend(payload["writes"]["chunk_summaries"])

    state["chunk_summaries"] = summaries
    aggregate_metric, aggregate_payload = invoke_worker(
        args,
        variant.workers["p_aggregate"],
        {"state": state, "chunk_size": args.chunk_size},
    )
    state.update(aggregate_payload["writes"])

    metrics = [ingest_metric, *map_metrics, aggregate_metric]
    workflow = {
        "variant": variant.name,
        "run_id": run_id,
        "collect": collect,
        "chunk_count": len(sends),
        "final_summary_length": len(state.get("final_summary", "")),
        "metrics": [metric_to_json(metric) for metric in metrics],
        "estimated_cost_usd": round(sum(metric.estimated_cost_usd or 0.0 for metric in metrics), 10),
        "billed_duration_ms": sum(metric.billed_duration_ms or 0 for metric in metrics),
        "handler_duration_ms": round(sum(metric.payload.get("meta", {}).get("handler_elapsed_ms", 0.0) for metric in metrics), 3),
    }
    if not collect:
        print(f"warmup {variant.name} {run_id}: chunks={len(sends)}")
    else:
        print(
            f"{variant.name} {run_id}: chunks={len(sends)} "
            f"billed_ms={workflow['billed_duration_ms']} cost=${workflow['estimated_cost_usd']}"
        )
    return workflow


def invoke_worker(
    args: argparse.Namespace,
    worker: WorkerConfig,
    event: dict[str, Any],
) -> tuple[InvokeMetric, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="orqis-invoke-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        payload_path = tmp_path / "payload.json"
        response_path = tmp_path / "response.json"
        payload_path.write_text(json.dumps(event), encoding="utf-8")
        result = run_aws(
            args,
            [
                "lambda",
                "invoke",
                "--function-name",
                worker.function_name,
                "--payload",
                f"fileb://{payload_path}",
                "--log-type",
                "Tail",
                str(response_path),
            ],
            capture=True,
        )
        invoke_meta = json.loads(result.stdout or "{}")
        payload = json.loads(response_path.read_text(encoding="utf-8") or "{}")
    log_text = decode_log_result(invoke_meta.get("LogResult"))
    report = parse_report_log(log_text)
    metric = InvokeMetric(
        variant=payload.get("meta", {}).get("variant", "unknown"),
        partition_id=worker.partition_id,
        status_code=int(invoke_meta.get("StatusCode", 0)),
        function_error=invoke_meta.get("FunctionError"),
        duration_ms=report.get("duration_ms"),
        billed_duration_ms=report.get("billed_duration_ms"),
        init_duration_ms=report.get("init_duration_ms"),
        max_memory_used_mb=report.get("max_memory_used_mb"),
        configured_memory_mb=worker.memory_mb,
        estimated_cost_usd=estimate_lambda_cost_usd(worker.memory_mb, report.get("billed_duration_ms")),
        payload=payload,
    )
    if metric.function_error:
        raise CommandError(f"{worker.function_name} failed: {payload}")
    return metric, payload


def sample_state(repeat: int) -> dict[str, Any]:
    text = (
        "LangGraph compiles workflows into a Pregel runtime with clear "
        "supersteps and dynamic fanout."
    )
    repeated = " ".join([text] * max(1, repeat))
    return {
        "doc_id": "doc-001",
        "text": repeated,
        "chunks": [],
        "chunk_summaries": [],
        "final_summary": "",
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_variant.setdefault(result["variant"], []).append(result)
    return {variant: summarize_variant(items) for variant, items in by_variant.items()}


def summarize_variant(items: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [item["estimated_cost_usd"] for item in items]
    billed = [item["billed_duration_ms"] for item in items]
    handler = [item["handler_duration_ms"] for item in items]
    partition_metrics: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        for metric in item["metrics"]:
            partition_metrics.setdefault(metric["partition_id"], []).append(metric)
    return {
        "runs": len(items),
        "avg_estimated_cost_usd": round(statistics.mean(costs), 10) if costs else 0.0,
        "avg_billed_duration_ms": round(statistics.mean(billed), 3) if billed else 0.0,
        "avg_handler_duration_ms": round(statistics.mean(handler), 3) if handler else 0.0,
        "partitions": {
            partition_id: summarize_partition_metrics(metrics)
            for partition_id, metrics in sorted(partition_metrics.items())
        },
    }


def summarize_partition_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [metric["duration_ms"] for metric in metrics if metric["duration_ms"] is not None]
    billed = [metric["billed_duration_ms"] for metric in metrics if metric["billed_duration_ms"] is not None]
    memory = [metric["max_memory_used_mb"] for metric in metrics if metric["max_memory_used_mb"] is not None]
    init = [metric["init_duration_ms"] for metric in metrics if metric["init_duration_ms"] is not None]
    return {
        "configured_memory_mb": metrics[0]["configured_memory_mb"],
        "invocations": len(metrics),
        "avg_duration_ms": round(statistics.mean(durations), 3) if durations else None,
        "p95_duration_ms": percentile(durations, 0.95),
        "avg_billed_duration_ms": round(statistics.mean(billed), 3) if billed else None,
        "max_memory_used_mb": max(memory) if memory else None,
        "cold_starts": len(init),
        "avg_init_duration_ms": round(statistics.mean(init), 3) if init else None,
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return round(ordered[index], 3)


def metric_to_json(metric: InvokeMetric) -> dict[str, Any]:
    return {
        "variant": metric.variant,
        "partition_id": metric.partition_id,
        "status_code": metric.status_code,
        "function_error": metric.function_error,
        "duration_ms": metric.duration_ms,
        "billed_duration_ms": metric.billed_duration_ms,
        "init_duration_ms": metric.init_duration_ms,
        "max_memory_used_mb": metric.max_memory_used_mb,
        "configured_memory_mb": metric.configured_memory_mb,
        "estimated_cost_usd": metric.estimated_cost_usd,
        "payload_meta": metric.payload.get("meta", {}),
    }


def estimate_lambda_cost_usd(memory_mb: int, billed_duration_ms: int | None) -> float | None:
    if billed_duration_ms is None:
        return None
    gb_seconds = (memory_mb / 1024.0) * (billed_duration_ms / 1000.0)
    return round(gb_seconds * AWS_LAMBDA_GB_SECOND_USD + AWS_LAMBDA_REQUEST_USD, 10)


REPORT_PATTERNS = {
    "duration_ms": re.compile(r"Duration: ([0-9.]+) ms"),
    "billed_duration_ms": re.compile(r"Billed Duration: ([0-9]+) ms"),
    "max_memory_used_mb": re.compile(r"Max Memory Used: ([0-9]+) MB"),
    "init_duration_ms": re.compile(r"Init Duration: ([0-9.]+) ms"),
}


def parse_report_log(log_text: str) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for key, pattern in REPORT_PATTERNS.items():
        match = pattern.search(log_text)
        if not match:
            continue
        value = match.group(1)
        report[key] = float(value) if "." in value else int(value)
    return report


def decode_log_result(value: str | None) -> str:
    if not value:
        return ""
    try:
        return base64.b64decode(value).decode("utf-8", errors="replace")
    except Exception:
        return ""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_aws(
    args: argparse.Namespace,
    command: list[str],
    *,
    capture: bool = False,
    allow_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    aws = shutil.which(args.aws_cli) or args.aws_cli
    full_command = [aws, "--no-cli-pager"]
    if args.profile:
        full_command.extend(["--profile", args.profile])
    if args.region:
        full_command.extend(["--region", args.region])
    full_command.extend(command)
    env = dict(os.environ)
    env["AWS_PAGER"] = ""
    result = subprocess.run(
        full_command,
        check=False,
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if result.returncode != 0 and not allow_failure:
        stderr = result.stderr.strip() if result.stderr else ""
        raise CommandError(f"AWS CLI failed ({result.returncode}): {' '.join(full_command)}\n{stderr}")
    return result

class CommandError(Exception):
    pass

if __name__ == "__main__":
    raise SystemExit(main())
