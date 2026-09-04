from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import tempfile
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


DEFAULT_PREFIX = "orqis-full"
DEFAULT_RUNTIME = "python3.11"
DEFAULT_HANDLER = "orqis.scripts.full_aws_runtime.lambda_handler"
DEFAULT_OUTPUT_DIR = Path("artifacts/full_aws_evaluation")
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "deployment_manifest.json"
DEFAULT_RESULTS_JSON = DEFAULT_OUTPUT_DIR / "full_aws_results.json"
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "full_aws_summary.csv"
DEFAULT_STATE_INLINE_LIMIT_BYTES = 300_000
DEFAULT_COORDINATOR_MEMORY_MB = 512
DEFAULT_COORDINATOR_TIMEOUT_SEC = 900
DEFAULT_BARRIER_MEMORY_MB = 512
DEFAULT_BARRIER_TIMEOUT_SEC = 900
DEFAULT_MIN_WORKER_TIMEOUT_SEC = 60
DEFAULT_MAX_WORKER_MEMORY_MB = 0
DEFAULT_STEPFUNCTIONS_TRANSITION_USD = 0.000025
DEFAULT_COMPILER_ARTIFACT_DIR = DEFAULT_OUTPUT_DIR / "compiler_plans"
DEFAULT_CHECKPOINT_READ_REQUEST_USD = 0.0
DEFAULT_CHECKPOINT_WRITE_REQUEST_USD = 0.0
DEFAULT_CHECKPOINT_READ_GB_USD = 0.0
DEFAULT_CHECKPOINT_WRITE_GB_USD = 0.0
DEFAULT_BEDROCK_INPUT_1K_TOKEN_USD = 0.0
DEFAULT_BEDROCK_OUTPUT_1K_TOKEN_USD = 0.0
AWS_LAMBDA_GB_SECOND_USD = 0.0000166667
AWS_LAMBDA_REQUEST_USD = 0.20 / 1_000_000
REPORT_PATTERNS = {
    "duration_ms": re.compile(r"Duration: ([0-9.]+) ms"),
    "billed_duration_ms": re.compile(r"Billed Duration: ([0-9]+) ms"),
    "max_memory_used_mb": re.compile(r"Max Memory Used: ([0-9]+) MB"),
    "init_duration_ms": re.compile(r"Init Duration: ([0-9.]+) ms"),
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    args.func(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy and run the final ORQIS evaluation workflows on materialized AWS resources."
    )
    subparsers = parser.add_subparsers(dest="command")

    catalog = subparsers.add_parser("catalog", help="Print the workflows, candidates, and datasets available to deploy.")
    catalog.set_defaults(func=cmd_catalog)

    compile_plans = subparsers.add_parser("compile-plans", help="Rerun ORQIS compilation for the evaluated workflows and SLO profiles.")
    add_selection_args(compile_plans)
    compile_plans.add_argument("--output-dir", type=Path, default=DEFAULT_COMPILER_ARTIFACT_DIR)
    compile_plans.add_argument("--compiler-use-bedrock", action="store_true")
    compile_plans.set_defaults(func=cmd_compile_plans)

    roles = subparsers.add_parser("bootstrap-roles", help="Create or update Lambda and Step Functions execution roles.")
    add_aws_args(roles)
    roles.add_argument("--lambda-role-name", default="orqis-full-aws-evaluation-lambda-role")
    roles.add_argument("--sfn-role-name", default="orqis-full-aws-evaluation-sfn-role")
    roles.add_argument("--prefix", default=DEFAULT_PREFIX)
    roles.add_argument("--state-table", default="")
    roles.add_argument("--state-bucket", default="")
    roles.set_defaults(func=cmd_bootstrap_roles)

    role = subparsers.add_parser("bootstrap-role", help="Create or update a Lambda execution role for this harness.")
    add_aws_args(role)
    role.add_argument("--role-name", default="orqis-full-aws-evaluation-role")
    role.add_argument("--prefix", default=DEFAULT_PREFIX)
    role.add_argument("--state-table", default="")
    role.add_argument("--state-bucket", default="")
    role.set_defaults(func=cmd_bootstrap_role)

    deploy = subparsers.add_parser("deploy", help="Package source code and create or update AWS resources.")
    add_aws_args(deploy)
    add_selection_args(deploy)
    deploy.add_argument("--prefix", default=DEFAULT_PREFIX)
    deploy.add_argument("--role-arn", required=True)
    deploy.add_argument("--sfn-role-arn", default="")
    deploy.add_argument("--orchestration", choices=["stepfunctions", "coordinator"], default="stepfunctions")
    deploy.add_argument("--state-machine-type", choices=["STANDARD"], default="STANDARD")
    deploy.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    deploy.add_argument("--package", type=Path, default=DEFAULT_OUTPUT_DIR / "lambda_package.zip")
    deploy.add_argument("--asl-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "asl")
    deploy.add_argument("--compiler-artifact-dir", type=Path, default=DEFAULT_COMPILER_ARTIFACT_DIR)
    deploy.add_argument("--skip-compile-plans", action="store_true")
    deploy.add_argument("--compiler-use-bedrock", action="store_true")
    deploy.add_argument("--dependency-dir", type=Path, action="append", default=[])
    deploy.add_argument("--runtime", default=DEFAULT_RUNTIME)
    deploy.add_argument("--architecture", choices=["arm64", "x86_64"], default="x86_64")
    deploy.add_argument("--state-table", default="orqis-full-aws-checkpoints")
    deploy.add_argument("--state-bucket", default="")
    deploy.add_argument(
        "--inline-checkpoints",
        action="store_true",
        help="Keep checkpoints in Step Functions/Lambda payloads instead of DynamoDB/S3.",
    )
    deploy.add_argument(
        "--skip-state-resource-check",
        action="store_true",
        help="Do not create or verify DynamoDB/S3 checkpoint resources during deploy.",
    )
    deploy.add_argument("--state-inline-limit-bytes", type=int, default=DEFAULT_STATE_INLINE_LIMIT_BYTES)
    deploy.add_argument("--min-worker-timeout-sec", type=int, default=DEFAULT_MIN_WORKER_TIMEOUT_SEC)
    deploy.add_argument(
        "--max-worker-memory-mb",
        type=int,
        default=DEFAULT_MAX_WORKER_MEMORY_MB,
        help="Optional deploy-time cap for worker memory. Use 0 to preserve candidate memory vectors exactly.",
    )
    deploy.add_argument(
        "--deploy-coordinator",
        action="store_true",
        help="Also deploy the fallback coordinator Lambda when using Step Functions orchestration.",
    )
    deploy.add_argument("--coordinator-memory-mb", type=int, default=DEFAULT_COORDINATOR_MEMORY_MB)
    deploy.add_argument("--coordinator-timeout-sec", type=int, default=DEFAULT_COORDINATOR_TIMEOUT_SEC)
    deploy.add_argument("--barrier-memory-mb", type=int, default=DEFAULT_BARRIER_MEMORY_MB)
    deploy.add_argument("--barrier-timeout-sec", type=int, default=DEFAULT_BARRIER_TIMEOUT_SEC)
    deploy.add_argument("--max-loop-steps", type=int, default=8)
    deploy.add_argument("--bedrock-model-id", default=os.environ.get("ORQIS_BEDROCK_MODEL_ID", ""))
    deploy.add_argument("--bedrock-region", default=os.environ.get("ORQIS_BEDROCK_REGION", ""))
    deploy.add_argument("--bedrock-temperature", type=float, default=float(os.environ.get("ORQIS_BEDROCK_TEMPERATURE", "0.0")))
    deploy.add_argument(
        "--bedrock-max-output-tokens",
        type=int,
        default=int(os.environ.get("ORQIS_BEDROCK_MAX_OUTPUT_TOKENS", "512")),
    )
    deploy.add_argument("--skip-reserved-concurrency", action="store_true")
    deploy.set_defaults(func=cmd_deploy)

    run = subparsers.add_parser("run", help="Invoke deployed coordinators and write raw results plus summaries.")
    add_aws_args(run)
    add_selection_args(run)
    run.add_argument("--orchestration", choices=["auto", "stepfunctions", "coordinator"], default="auto")
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    run.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    run.add_argument("--cold-probe-runs", type=int, default=0)
    run.add_argument("--warmup-runs", type=int, default=1)
    run.add_argument("--runs", type=int, default=5)
    run.add_argument("--load-batches", type=int, default=0)
    run.add_argument("--load-concurrency", type=int, default=0)
    run.add_argument("--execution-timeout-sec", type=int, default=1200)
    run.add_argument("--poll-interval-sec", type=float, default=2.0)
    run.add_argument("--log-fetch-timeout-sec", type=float, default=30.0)
    run.add_argument("--skip-log-enrichment", action="store_true")
    run.add_argument("--stepfunctions-transition-usd", type=float, default=DEFAULT_STEPFUNCTIONS_TRANSITION_USD)
    run.add_argument("--checkpoint-read-request-usd", type=float, default=DEFAULT_CHECKPOINT_READ_REQUEST_USD)
    run.add_argument("--checkpoint-write-request-usd", type=float, default=DEFAULT_CHECKPOINT_WRITE_REQUEST_USD)
    run.add_argument("--checkpoint-read-gb-usd", type=float, default=DEFAULT_CHECKPOINT_READ_GB_USD)
    run.add_argument("--checkpoint-write-gb-usd", type=float, default=DEFAULT_CHECKPOINT_WRITE_GB_USD)
    run.add_argument("--bedrock-input-1k-token-usd", type=float, default=DEFAULT_BEDROCK_INPUT_1K_TOKEN_USD)
    run.add_argument("--bedrock-output-1k-token-usd", type=float, default=DEFAULT_BEDROCK_OUTPUT_1K_TOKEN_USD)
    run.set_defaults(func=cmd_run)

    cleanup = subparsers.add_parser("cleanup", help="Delete Lambda functions from a deployment manifest.")
    add_aws_args(cleanup)
    cleanup.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    cleanup.add_argument("--delete-state-table", action="store_true")
    cleanup.add_argument("--empty-and-delete-state-bucket", action="store_true")
    cleanup.set_defaults(func=cmd_cleanup)

    return parser


def add_aws_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE", ""))


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workflow", action="append", default=[], help="Workflow id or comma-separated workflow ids.")
    parser.add_argument("--candidate", action="append", default=[], help="Candidate id or comma-separated candidate ids.")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset id or comma-separated dataset ids.")


def cmd_catalog(_args: argparse.Namespace) -> None:
    for workflow_id, module in experiment_modules().items():
        print(workflow_id)
        print("  candidates:")
        for candidate in get_candidates(module):
            memory = ",".join(str(value) for value in candidate.memory_vector_mb)
            print(f"    {candidate.candidate:6s} {candidate.group:18s} partitions={candidate.partitioning_vector} memory={memory}")
        print("  datasets:")
        print("    " + ", ".join(sorted(get_datasets(module))))


def cmd_compile_plans(args: argparse.Namespace) -> None:
    selected = selected_workflows(args)
    artifacts = compile_current_plans(
        args.output_dir,
        selected,
        set(flatten_filters(args.candidate)),
        use_bedrock=bool(args.compiler_use_bedrock),
    )
    validate_compiler_rows_against_artifacts(args.output_dir, selected, set(flatten_filters(args.candidate)))
    print(f"Wrote {len(artifacts)} compiler artifact sets to {args.output_dir}.")


def cmd_bootstrap_roles(args: argparse.Namespace) -> None:
    session = boto3_session(args)
    iam = session.client("iam")
    sts = session.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    region = args.region or session.region_name or "us-east-1"
    lambda_role = upsert_role(
        iam,
        role_name=args.lambda_role_name,
        service_principal="lambda.amazonaws.com",
        description="Execution role for materialized ORQIS evaluation Lambdas.",
        policy_name="orqis-full-aws-evaluation-lambda",
        policy=build_execution_policy(
            region=region,
            account_id=account_id,
            prefix=args.prefix,
            state_table=args.state_table,
            state_bucket=args.state_bucket,
        ),
    )
    sfn_role = upsert_role(
        iam,
        role_name=args.sfn_role_name,
        service_principal="states.amazonaws.com",
        description="Execution role for ORQIS evaluation Step Functions state machines.",
        policy_name="orqis-full-aws-evaluation-stepfunctions",
        policy=build_stepfunctions_policy(region=region, account_id=account_id, prefix=args.prefix),
    )
    print(json.dumps({"lambda_role_arn": lambda_role["Arn"], "sfn_role_arn": sfn_role["Arn"]}, indent=2))
    print("Role policies updated. AWS may need a short propagation delay before deployment succeeds.")


def cmd_bootstrap_role(args: argparse.Namespace) -> None:
    session = boto3_session(args)
    iam = session.client("iam")
    sts = session.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    region = args.region or session.region_name or "us-east-1"
    role = upsert_role(
        iam,
        role_name=args.role_name,
        service_principal="lambda.amazonaws.com",
        description="Execution role for materialized ORQIS evaluation Lambdas.",
        policy_name="orqis-full-aws-evaluation",
        policy=build_execution_policy(
            region=region,
            account_id=account_id,
            prefix=args.prefix,
            state_table=args.state_table,
            state_bucket=args.state_bucket,
        ),
    )
    print(role["Arn"])
    print("Role policy updated. AWS may need a short propagation delay before Lambda creation succeeds.")


def cmd_deploy(args: argparse.Namespace) -> None:
    selected = selected_workflows(args)
    if not selected:
        raise SystemExit("No workflows selected.")
    if args.orchestration == "stepfunctions" and not args.sfn_role_arn:
        raise SystemExit("--sfn-role-arn is required when --orchestration stepfunctions.")

    session = boto3_session(args)
    lambda_client = session.client("lambda")
    states_client = session.client("stepfunctions")
    ddb_client = session.client("dynamodb")
    s3_client = session.client("s3")
    sts_client = session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    region = args.region or session.region_name
    if not region:
        raise SystemExit("Set --region or AWS_REGION.")
    if args.inline_checkpoints:
        args.state_table = ""
        args.state_bucket = ""

    compiler_artifacts = []
    if not args.skip_compile_plans:
        compiler_artifacts = compile_current_plans(
            args.compiler_artifact_dir,
            selected,
            set(flatten_filters(args.candidate)),
            use_bedrock=bool(args.compiler_use_bedrock),
        )
        validate_compiler_rows_against_artifacts(
            args.compiler_artifact_dir,
            selected,
            set(flatten_filters(args.candidate)),
        )

    if not args.skip_state_resource_check:
        ensure_state_table(ddb_client, args.state_table)
        if args.state_bucket:
            ensure_bucket(s3_client, args.state_bucket, region)

    args.package.parent.mkdir(parents=True, exist_ok=True)
    build_lambda_package(args.package, args.dependency_dir)
    package_sha256 = sha256_file(args.package)

    deployment_records: list[dict[str, Any]] = []
    for workflow_id, module in selected.items():
        candidates = selected_candidates(args, module)
        for candidate in candidates:
            worker_specs = effective_worker_specs(
                candidate,
                min_timeout_sec=int(args.min_worker_timeout_sec),
                max_memory_mb=int(args.max_worker_memory_mb) if args.max_worker_memory_mb else None,
            )
            candidate_payload = serialize_candidate(candidate, worker_specs=worker_specs)
            worker_map: dict[str, dict[str, Any]] = {}
            for spec in worker_specs:
                logical_id = str(spec["logical_id"])
                worker_name = aws_name(args.prefix, workflow_id, candidate.candidate, logical_id)
                worker_env = build_worker_env(args, workflow_id, candidate.candidate, logical_id)
                worker_arn = upsert_lambda_function(
                    lambda_client,
                    function_name=worker_name,
                    role_arn=args.role_arn,
                    package_path=args.package,
                    handler=DEFAULT_HANDLER,
                    runtime=args.runtime,
                    architecture=args.architecture,
                    memory_mb=int(spec["memory_mb"]),
                    timeout_sec=int(spec["timeout_sec"]),
                    environment=worker_env,
                    description=f"ORQIS materialized worker {workflow_id} {candidate.candidate} {logical_id}",
                )
                if not args.skip_reserved_concurrency and spec.get("concurrency_limit") is not None:
                    lambda_client.put_function_concurrency(
                        FunctionName=worker_name,
                        ReservedConcurrentExecutions=int(spec["concurrency_limit"]),
                    )
                worker_map[logical_id] = {
                    "function_name": worker_name,
                    "function_arn": worker_arn,
                    "memory_mb": int(spec["memory_mb"]),
                    "compiler_memory_mb": int(spec.get("compiler_memory_mb") or spec["memory_mb"]),
                    "timeout_sec": int(spec["timeout_sec"]),
                    "compiler_timeout_sec": int(spec.get("compiler_timeout_sec") or spec["timeout_sec"]),
                    "concurrency_limit": spec.get("concurrency_limit"),
                }

            coordinator = None
            if args.orchestration == "coordinator" or args.deploy_coordinator:
                coordinator_name = aws_name(args.prefix, workflow_id, candidate.candidate, "coordinator")
                coordinator_env = build_coordinator_env(args, candidate_payload, worker_map)
                coordinator_arn = upsert_lambda_function(
                    lambda_client,
                    function_name=coordinator_name,
                    role_arn=args.role_arn,
                    package_path=args.package,
                    handler=DEFAULT_HANDLER,
                    runtime=args.runtime,
                    architecture=args.architecture,
                    memory_mb=int(args.coordinator_memory_mb),
                    timeout_sec=int(args.coordinator_timeout_sec),
                    environment=coordinator_env,
                    description=f"ORQIS materialized coordinator {workflow_id} {candidate.candidate}",
                )
                coordinator = {
                    "function_name": coordinator_name,
                    "function_arn": coordinator_arn,
                    "memory_mb": int(args.coordinator_memory_mb),
                    "timeout_sec": int(args.coordinator_timeout_sec),
                }
            barrier = None
            state_machine = None
            asl_path = None
            if args.orchestration == "stepfunctions":
                barrier_name = aws_name(args.prefix, workflow_id, candidate.candidate, "barrier")
                barrier_env = build_barrier_env(args, candidate_payload)
                barrier_arn = upsert_lambda_function(
                    lambda_client,
                    function_name=barrier_name,
                    role_arn=args.role_arn,
                    package_path=args.package,
                    handler=DEFAULT_HANDLER,
                    runtime=args.runtime,
                    architecture=args.architecture,
                    memory_mb=int(args.barrier_memory_mb),
                    timeout_sec=int(args.barrier_timeout_sec),
                    environment=barrier_env,
                    description=f"ORQIS Step Functions barrier {workflow_id} {candidate.candidate}",
                )
                asl = generate_state_machine_definition(
                    workflow=workflow_id,
                    candidate_id=candidate.candidate,
                    worker_map=worker_map,
                    barrier_arn=barrier_arn,
                    max_loop_steps=int(args.max_loop_steps),
                )
                args.asl_dir.mkdir(parents=True, exist_ok=True)
                asl_path = args.asl_dir / f"{workflow_id}_{candidate.candidate}.asl.json"
                write_json(asl_path, asl)
                state_machine_name = aws_name(args.prefix, workflow_id, candidate.candidate, "sfn", max_length=80)
                state_machine_arn = state_machine_arn_for(region, account_id, state_machine_name)
                state_machine_arn = upsert_state_machine(
                    states_client,
                    state_machine_name=state_machine_name,
                    state_machine_arn=state_machine_arn,
                    role_arn=args.sfn_role_arn,
                    definition=asl,
                    state_machine_type=args.state_machine_type,
                )
                barrier = {
                    "function_name": barrier_name,
                    "function_arn": barrier_arn,
                    "memory_mb": int(args.barrier_memory_mb),
                    "timeout_sec": int(args.barrier_timeout_sec),
                }
                state_machine = {
                    "name": state_machine_name,
                    "arn": state_machine_arn,
                    "type": args.state_machine_type,
                    "definition_path": str(asl_path),
                }
            deployment_records.append(
                {
                    "workflow": workflow_id,
                    "candidate": candidate.candidate,
                    "candidate_config": candidate_payload,
                    "orchestration": args.orchestration,
                    "coordinator": coordinator,
                    "barrier": barrier,
                    "state_machine": state_machine,
                    "workers": worker_map,
                }
            )

    manifest = {
        "created_at": now_utc(),
        "region": region,
        "prefix": args.prefix,
        "runtime": args.runtime,
        "architecture": args.architecture,
        "handler": DEFAULT_HANDLER,
        "package": str(args.package),
        "package_sha256": package_sha256,
        "orchestration": args.orchestration,
        "state_machine_type": args.state_machine_type if args.orchestration == "stepfunctions" else None,
        "state_table": args.state_table,
        "state_bucket": args.state_bucket,
        "state_inline_limit_bytes": int(args.state_inline_limit_bytes),
        "min_worker_timeout_sec": int(args.min_worker_timeout_sec),
        "max_worker_memory_mb": int(args.max_worker_memory_mb) if args.max_worker_memory_mb else None,
        "compiler_artifacts": compiler_artifacts,
        "records": deployment_records,
    }
    write_json(args.manifest, manifest)
    print(f"Deployed {len(deployment_records)} candidate deployments.")
    print(f"Manifest: {args.manifest}")


def cmd_run(args: argparse.Namespace) -> None:
    if args.load_batches > 0 and args.load_concurrency <= 0:
        raise SystemExit("--load-concurrency must be positive when --load-batches is greater than zero.")
    manifest = read_json(args.manifest)
    session = boto3_session(args)
    lambda_client = session.client("lambda")
    states_client = session.client("stepfunctions")
    logs_client = session.client("logs")
    records = selected_manifest_records(args, manifest)
    if not records:
        raise SystemExit("No deployed candidate records selected.")

    raw_entries: list[dict[str, Any]] = []
    for record in records:
        invoker = choose_invoker(args, record, lambda_client, states_client, logs_client)
        module = experiment_modules()[record["workflow"]]
        datasets = selected_datasets(args, module)
        for dataset_id in datasets:
            dataset_profile = dict(get_datasets(module)[dataset_id])
            raw_entries.extend(
                run_candidate_dataset(
                    invoker=invoker,
                    record=record,
                    dataset_profile=dataset_profile,
                    cold_probe_runs=int(args.cold_probe_runs),
                    warmup_runs=int(args.warmup_runs),
                    measured_runs=int(args.runs),
                    load_batches=int(args.load_batches),
                    load_concurrency=int(args.load_concurrency),
                )
            )

    summary_rows = build_summary_rows(raw_entries, manifest, args)
    payload = {
        "created_at": now_utc(),
        "manifest_path": str(args.manifest),
        "run_config": {
            "cold_probe_runs": int(args.cold_probe_runs),
            "warmup_runs": int(args.warmup_runs),
            "runs": int(args.runs),
            "load_batches": int(args.load_batches),
            "load_concurrency": int(args.load_concurrency),
            "orchestration": args.orchestration,
            "stepfunctions_transition_usd": float(args.stepfunctions_transition_usd),
            "log_enrichment": not bool(args.skip_log_enrichment),
            "checkpoint_cost_inputs": checkpoint_cost_config(args),
            "bedrock_cost_inputs": bedrock_cost_config(args),
        },
        "entries": raw_entries,
        "summary": summary_rows,
    }
    write_json(args.results_json, payload)
    write_csv(args.summary_csv, summary_rows)
    print(f"Wrote {len(raw_entries)} raw invocations.")
    print(f"Results: {args.results_json}")
    print(f"Summary: {args.summary_csv}")


def cmd_cleanup(args: argparse.Namespace) -> None:
    manifest = read_json(args.manifest)
    session = boto3_session(args)
    states_client = session.client("stepfunctions")
    lambda_client = session.client("lambda")
    for state_machine_arn in sorted(deployed_state_machine_arns(manifest)):
        try:
            states_client.delete_state_machine(stateMachineArn=state_machine_arn)
            print(f"deleted {state_machine_arn}")
        except states_client.exceptions.StateMachineDoesNotExist:
            pass
    for function_name in sorted(deployed_function_names(manifest)):
        try:
            lambda_client.delete_function(FunctionName=function_name)
            print(f"deleted {function_name}")
        except lambda_client.exceptions.ResourceNotFoundException:
            pass
    if args.delete_state_table and manifest.get("state_table"):
        ddb_client = session.client("dynamodb")
        try:
            ddb_client.delete_table(TableName=manifest["state_table"])
            print(f"deleted table {manifest['state_table']}")
        except ddb_client.exceptions.ResourceNotFoundException:
            pass
    if args.empty_and_delete_state_bucket and manifest.get("state_bucket"):
        s3_client = session.client("s3")
        empty_and_delete_bucket(s3_client, manifest["state_bucket"])
        print(f"deleted bucket {manifest['state_bucket']}")


def run_candidate_dataset(
    *,
    invoker: Any,
    record: dict[str, Any],
    dataset_profile: dict[str, Any],
    cold_probe_runs: int,
    warmup_runs: int,
    measured_runs: int,
    load_batches: int,
    load_concurrency: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index in range(cold_probe_runs):
        entries.append(invoker(record, dataset_profile, "cold_probe", index))
    for index in range(warmup_runs):
        entries.append(invoker(record, dataset_profile, "warmup", index))
    for index in range(measured_runs):
        entries.append(invoker(record, dataset_profile, "single", index))
    for batch_index in range(load_batches):
        started = perf_counter()
        batch_entries: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=load_concurrency) as executor:
            futures = [
                executor.submit(invoker, record, dataset_profile, "load", slot, batch_index)
                for slot in range(load_concurrency)
            ]
            for future in as_completed(futures):
                batch_entries.append(future.result())
        batch_elapsed_ms = round((perf_counter() - started) * 1000, 3)
        throughput_rps = round(load_concurrency / (batch_elapsed_ms / 1000.0), 6) if batch_elapsed_ms > 0 else None
        for entry in batch_entries:
            entry["batch_makespan_ms"] = batch_elapsed_ms
            entry["batch_throughput_rps"] = throughput_rps
            entry["batch_concurrency"] = load_concurrency
        entries.extend(sorted(batch_entries, key=lambda item: int(item["slot_index"])))
    return entries


def invoke_record(
    lambda_client: Any,
    record: dict[str, Any],
    dataset_profile: dict[str, Any],
    phase: str,
    slot_index: int,
    batch_index: int | None = None,
) -> dict[str, Any]:
    workflow = record["workflow"]
    candidate = record["candidate"]
    dataset_id = dataset_profile["dataset_id"]
    coordinator = record.get("coordinator")
    if not coordinator:
        raise SystemExit(f"{workflow} {candidate} has no deployed coordinator. Use --orchestration stepfunctions.")
    label = f"{phase}-{slot_index}" if batch_index is None else f"{phase}-{batch_index}-{slot_index}"
    run_id = f"{workflow}:{candidate}:{dataset_id}:{label}:{int(time.time() * 1000)}"
    event = {
        "dataset_profile": dataset_profile,
        "run_id": run_id,
        "invocation_label": label,
    }
    started = perf_counter()
    response = lambda_client.invoke(
        FunctionName=coordinator["function_name"],
        InvocationType="RequestResponse",
        LogType="Tail",
        Payload=json.dumps(event).encode("utf-8"),
    )
    client_elapsed_ms = round((perf_counter() - started) * 1000, 3)
    raw = response["Payload"].read().decode("utf-8") or "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"error": raw[:4096]}
    log_text = decode_log_result(response.get("LogResult"))
    report = parse_report_log(log_text)
    coordinator_metric = dict(payload.get("coordinator_metric") or {})
    coordinator_metric.update(
        {
            "function_name": coordinator["function_name"],
            "configured_memory_mb": int(coordinator["memory_mb"]),
            "configured_timeout_sec": int(coordinator["timeout_sec"]),
            "duration_ms": report.get("duration_ms", coordinator_metric.get("handler_elapsed_ms")),
            "billed_duration_ms": report.get("billed_duration_ms"),
            "init_duration_ms": report.get("init_duration_ms"),
            "max_memory_used_mb": report.get("max_memory_used_mb"),
            "estimated_cost_usd": estimate_lambda_cost_usd(
                coordinator["memory_mb"], report.get("billed_duration_ms")
            ),
            "client_elapsed_ms": client_elapsed_ms,
            "status_code": int(response.get("StatusCode", 0) or 0),
            "function_error": response.get("FunctionError"),
        }
    )
    payload["coordinator_metric"] = coordinator_metric
    payload["client_elapsed_ms"] = client_elapsed_ms
    payload["lambda_function_error"] = response.get("FunctionError")
    payload["lambda_log_tail"] = log_text[-4096:] if response.get("FunctionError") else ""
    payload["lambda_total_estimated_cost_usd"] = round(
        float(payload.get("estimated_cost_usd") or 0.0) + float(coordinator_metric.get("estimated_cost_usd") or 0.0),
        10,
    )
    return {
        "phase": phase,
        "workflow": workflow,
        "candidate": candidate,
        "dataset_id": dataset_id,
        "slot_index": slot_index,
        "batch_index": batch_index,
        "run_id": run_id,
        "payload": payload,
    }


def choose_invoker(
    args: argparse.Namespace,
    record: dict[str, Any],
    lambda_client: Any,
    states_client: Any,
    logs_client: Any,
) -> Any:
    mode = args.orchestration
    if mode == "auto":
        mode = "stepfunctions" if record.get("state_machine") else "coordinator"
    if mode == "stepfunctions":
        if not record.get("state_machine"):
            raise SystemExit(f"{record['workflow']} {record['candidate']} has no deployed state machine.")

        def _invoke(
            selected_record: dict[str, Any],
            dataset_profile: dict[str, Any],
            phase: str,
            slot_index: int,
            batch_index: int | None = None,
        ) -> dict[str, Any]:
            return invoke_stepfunctions_record(
                states_client,
                logs_client,
                selected_record,
                dataset_profile,
                phase,
                slot_index,
                batch_index,
                args,
            )

        return _invoke

    def _invoke(
        selected_record: dict[str, Any],
        dataset_profile: dict[str, Any],
        phase: str,
        slot_index: int,
        batch_index: int | None = None,
    ) -> dict[str, Any]:
        return invoke_record(lambda_client, selected_record, dataset_profile, phase, slot_index, batch_index)

    return _invoke


def invoke_stepfunctions_record(
    states_client: Any,
    logs_client: Any,
    record: dict[str, Any],
    dataset_profile: dict[str, Any],
    phase: str,
    slot_index: int,
    batch_index: int | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    workflow = record["workflow"]
    candidate = record["candidate"]
    dataset_id = dataset_profile["dataset_id"]
    label = f"{phase}-{slot_index}" if batch_index is None else f"{phase}-{batch_index}-{slot_index}"
    timestamp_ms = int(time.time() * 1000)
    run_id = f"{workflow}:{candidate}:{dataset_id}:{label}:{timestamp_ms}"
    execution_name = aws_name(workflow, candidate, dataset_id, label, timestamp_ms, max_length=80)
    execution_input = {
        "dataset_profile": dataset_profile,
        "run_id": run_id,
        "invocation_label": label,
    }
    started = perf_counter()
    start_response = states_client.start_execution(
        stateMachineArn=record["state_machine"]["arn"],
        name=execution_name,
        input=json.dumps(execution_input),
    )
    execution_arn = start_response["executionArn"]
    description = wait_for_execution(
        states_client,
        execution_arn,
        timeout_sec=float(args.execution_timeout_sec),
        poll_interval_sec=float(args.poll_interval_sec),
    )
    client_elapsed_ms = round((perf_counter() - started) * 1000, 3)
    history = get_execution_history(states_client, execution_arn)
    if description.get("status") == "SUCCEEDED":
        payload = json.loads(description.get("output") or "{}")
    else:
        payload = {
            "workflow": workflow,
            "candidate": candidate,
            "dataset_id": dataset_id,
            "error": description.get("status"),
            "execution_error": description.get("error"),
            "execution_cause": description.get("cause"),
        }
    payload["client_elapsed_ms"] = client_elapsed_ms
    payload["stepfunctions_metric"] = build_stepfunctions_metric(description, history, args)
    if not args.skip_log_enrichment:
        enrich_payload_from_lambda_logs(
            logs_client,
            payload,
            timeout_sec=float(args.log_fetch_timeout_sec),
            start_time=start_response["startDate"],
        )
    refresh_payload_cost_totals(payload)
    return {
        "phase": phase,
        "workflow": workflow,
        "candidate": candidate,
        "dataset_id": dataset_id,
        "slot_index": slot_index,
        "batch_index": batch_index,
        "run_id": run_id,
        "execution_arn": execution_arn,
        "payload": payload,
    }


def generate_state_machine_definition(
    *,
    workflow: str,
    candidate_id: str,
    worker_map: dict[str, dict[str, Any]],
    barrier_arn: str,
    max_loop_steps: int,
) -> dict[str, Any]:
    if workflow == "service_desk_router":
        return generate_router_state_machine(candidate_id, worker_map, barrier_arn)
    if workflow == "skills":
        return generate_skills_state_machine(candidate_id, worker_map, barrier_arn, max_loop_steps)
    if workflow == "incident_response_swarm":
        return generate_subagents_state_machine(candidate_id, worker_map, barrier_arn)
    raise ValueError(f"unsupported workflow for Step Functions generation: {workflow}")


def generate_router_state_machine(candidate_id: str, worker_map: dict[str, dict[str, Any]], barrier_arn: str) -> dict[str, Any]:
    states: dict[str, Any] = {"Init": init_state(barrier_arn, "service_desk_router", candidate_id, "Start")}
    if candidate_id in {"RT-B", "RT-C", "RT-L", "RT-R", "RT-U3"}:
        add_worker_apply(states, worker_map, barrier_arn, "Start", "p_intake_request_triage_request_fanout", ["request_text", "account_tier"], "ChooseRoute")
    elif candidate_id == "RT-U1":
        add_worker_apply(states, worker_map, barrier_arn, "Start", "p_intake_request", ["request_text"], "Triage")
        add_worker_apply(states, worker_map, barrier_arn, "Triage", "p_triage_request", ["account_tier", "normalized_text"], "ChooseRoute")
    elif candidate_id == "RT-U2":
        add_worker_apply(states, worker_map, barrier_arn, "Start", "p_intake_request_triage_request", ["request_text", "account_tier"], "MergedSpecialist")
        add_worker_apply(
            states,
            worker_map,
            barrier_arn,
            "MergedSpecialist",
            "p_merged_specialist",
            ["case_id", "account_tier", "normalized_text", "route_decision"],
            "Finalize",
        )
        add_worker_apply(states, worker_map, barrier_arn, "Finalize", "p_finalize_response", ["route_decision", "response_draft", "evidence_refs", "follow_up_plan"], "Finish")
        states["Finish"] = finish_state(barrier_arn)
        return state_machine(states)
    else:
        raise ValueError(f"unsupported router candidate: {candidate_id}")

    states["ChooseRoute"] = {
        "Type": "Choice",
        "Choices": [
            send_choice("p_billing_specialist", "BillingSpecialist"),
            send_choice("p_identity_specialist", "IdentitySpecialist"),
            send_choice("p_vendor_security_specialist", "VendorSecuritySpecialist"),
        ],
        "Default": "RouteError",
    }
    states["RouteError"] = {"Type": "Pass", "Result": "RouteError: no specialist route emitted", "ResultPath": "$.error", "Next": "Finish"}
    add_worker_apply(states, worker_map, barrier_arn, "BillingSpecialist", "p_billing_specialist", ["case_id", "account_tier", "normalized_text"], "Finalize")
    add_worker_apply(states, worker_map, barrier_arn, "IdentitySpecialist", "p_identity_specialist", ["case_id", "account_tier", "normalized_text"], "Finalize")
    add_worker_apply(states, worker_map, barrier_arn, "VendorSecuritySpecialist", "p_vendor_security_specialist", ["case_id", "account_tier", "normalized_text"], "Finalize")
    add_worker_apply(states, worker_map, barrier_arn, "Finalize", "p_finalize_response", ["route_decision", "response_draft", "evidence_refs", "follow_up_plan"], "Finish")
    states["Finish"] = finish_state(barrier_arn)
    return state_machine(states)


def generate_skills_state_machine(
    candidate_id: str,
    worker_map: dict[str, dict[str, Any]],
    barrier_arn: str,
    max_loop_steps: int,
) -> dict[str, Any]:
    states: dict[str, Any] = {"Init": init_state(barrier_arn, "skills", candidate_id, "Start")}
    if candidate_id == "SK-U1":
        add_worker_apply(states, worker_map, barrier_arn, "Start", "p_monolith", ["messages", "skills_loaded", "final_response"], "Finish")
        states["Finish"] = finish_state(barrier_arn)
        return state_machine(states)

    model_id = "p_model_fanout"
    tools_id = "p_tools_fanout"
    add_worker_apply(states, worker_map, barrier_arn, "Start", model_id, ["messages", "skills_loaded", "final_response"], "AfterModel")
    states["AfterModel"] = choice_loop_or_finish(
        next_logical_id=tools_id,
        next_state="CheckToolsLoopLimit",
        finish_state_name="Finish",
    )
    states["CheckToolsLoopLimit"] = loop_limit_choice(max_loop_steps, "Tools", "LoopLimit")
    add_worker_apply(states, worker_map, barrier_arn, "Tools", tools_id, ["messages", "skills_loaded", "final_response"], "AfterTools")
    states["AfterTools"] = choice_loop_or_finish(
        next_logical_id=model_id,
        next_state="CheckModelLoopLimit",
        finish_state_name="Finish",
    )
    states["CheckModelLoopLimit"] = loop_limit_choice(max_loop_steps, "Start", "LoopLimit")
    states["LoopLimit"] = {
        "Type": "Pass",
        "Result": f"LoopLimitError: exceeded max loop steps of {max_loop_steps}",
        "ResultPath": "$.error",
        "Next": "Finish",
    }
    states["Finish"] = finish_state(barrier_arn)
    return state_machine(states)


def generate_subagents_state_machine(candidate_id: str, worker_map: dict[str, dict[str, Any]], barrier_arn: str) -> dict[str, Any]:
    states: dict[str, Any] = {"Init": init_state(barrier_arn, "incident_response_swarm", candidate_id, "Start")}
    if candidate_id in {"IR-B", "IR-C", "IR-L", "IR-R", "IR-U3"}:
        add_worker_apply(
            states,
            worker_map,
            barrier_arn,
            "Start",
            "p_ingest_alert_plan_response_fanout",
            ["incident_id", "severity", "alert_summary"],
            "RunSubagents",
        )
    elif candidate_id == "IR-U1":
        add_worker_apply(states, worker_map, barrier_arn, "Start", "p_ingest_alert", ["severity", "alert_summary"], "PlanResponse")
        add_worker_apply(
            states,
            worker_map,
            barrier_arn,
            "PlanResponse",
            "p_plan_response_fanout",
            ["incident_id", "severity", "alert_summary", "evidence_scope"],
            "RunSubagents",
        )
    elif candidate_id == "IR-U2":
        add_worker_apply(states, worker_map, barrier_arn, "Start", "p_monolith", ["incident_id", "severity", "alert_summary"], "Finish")
        states["Finish"] = finish_state(barrier_arn)
        return state_machine(states)
    else:
        raise ValueError(f"unsupported subagent candidate: {candidate_id}")

    states["RunSubagents"] = subagent_map_state(worker_map, "ApplySubagentBranches")
    states["ApplySubagentBranches"] = apply_state(barrier_arn, "subagent_branches", "Synthesize", branch_results=True)
    add_worker_apply(
        states,
        worker_map,
        barrier_arn,
        "Synthesize",
        "p_synthesize_recommendation",
        ["findings", "containment_actions", "communication_drafts"],
        "Finalize",
    )
    add_worker_apply(
        states,
        worker_map,
        barrier_arn,
        "Finalize",
        "p_finalize_incident",
        ["executive_summary", "final_recommendation", "communication_drafts"],
        "Finish",
    )
    states["Finish"] = finish_state(barrier_arn)
    return state_machine(states)


def state_machine(states: dict[str, Any]) -> dict[str, Any]:
    return {"Comment": "Generated ORQIS evaluation state machine.", "StartAt": "Init", "States": states}


def init_state(barrier_arn: str, workflow: str, candidate_id: str, next_state: str) -> dict[str, Any]:
    return {
        "Type": "Task",
        "Resource": barrier_arn,
        "Parameters": {
            "action": "init",
            "workflow": workflow,
            "candidate": candidate_id,
            "dataset_profile.$": "$.dataset_profile",
            "run_id.$": "$.run_id",
            "invocation_label.$": "$.invocation_label",
        },
        "Next": next_state,
    }


def add_worker_apply(
    states: dict[str, Any],
    worker_map: dict[str, dict[str, Any]],
    barrier_arn: str,
    state_name: str,
    logical_id: str,
    read_keys: list[str],
    next_state: str,
) -> None:
    states[state_name] = worker_state(worker_map[logical_id]["function_arn"], logical_id, read_keys, apply_state_name(state_name))
    states[apply_state_name(state_name)] = apply_state(barrier_arn, logical_id, next_state)


def apply_state_name(state_name: str) -> str:
    return f"Apply{state_name}"


def worker_state(function_arn: str, logical_id: str, read_keys: list[str], next_state: str | None = None) -> dict[str, Any]:
    state = {
        "Type": "Task",
        "Resource": function_arn,
        "Parameters": {
            "logical_id": logical_id,
            "checkpoint_ref.$": "$.checkpoint_ref",
            "read_keys": read_keys,
            "task_input": {},
        },
        "ResultPath": "$.worker_result",
    }
    if next_state is None:
        state["End"] = True
    else:
        state["Next"] = next_state
    return state


def apply_state(barrier_arn: str, label: str, next_state: str, *, branch_results: bool = False) -> dict[str, Any]:
    parameters = context_parameters("apply", label)
    if branch_results:
        parameters["worker_results.$"] = "$.worker_results"
    else:
        parameters["worker_result.$"] = "$.worker_result"
    return {
        "Type": "Task",
        "Resource": barrier_arn,
        "Parameters": parameters,
        "Next": next_state,
    }


def finish_state(barrier_arn: str) -> dict[str, Any]:
    return {
        "Type": "Task",
        "Resource": barrier_arn,
        "Parameters": context_parameters("finish", "finish"),
        "End": True,
    }


def context_parameters(action: str, label: str) -> dict[str, Any]:
    return {
        "action": action,
        "label": label,
        "workflow.$": "$.workflow",
        "candidate.$": "$.candidate",
        "dataset_id.$": "$.dataset_id",
        "dataset_profile.$": "$.dataset_profile",
        "run_id.$": "$.run_id",
        "invocation_label.$": "$.invocation_label",
        "checkpoint_ref.$": "$.checkpoint_ref",
        "checkpoint_trace.$": "$.checkpoint_trace",
        "metrics.$": "$.metrics",
        "barrier_metrics.$": "$.barrier_metrics",
        "invoked_partitions.$": "$.invoked_partitions",
        "step_count.$": "$.step_count",
        "barrier_apply_count.$": "$.barrier_apply_count",
        "error.$": "$.error",
        "start_time_ms.$": "$.start_time_ms",
    }


def choice_loop_or_finish(*, next_logical_id: str, next_state: str, finish_state_name: str) -> dict[str, Any]:
    return {
        "Type": "Choice",
        "Choices": [
            send_choice(next_logical_id, next_state),
        ],
        "Default": finish_state_name,
    }


def send_choice(logical_id: str, next_state: str) -> dict[str, Any]:
    return {
        "And": [
            {"Variable": "$.last_sends[0].logical_id", "IsPresent": True},
            {"Variable": "$.last_sends[0].logical_id", "StringEquals": logical_id},
        ],
        "Next": next_state,
    }


def loop_limit_choice(max_loop_steps: int, continue_state: str, limit_state: str) -> dict[str, Any]:
    return {
        "Type": "Choice",
        "Choices": [
            {"Variable": "$.step_count", "NumericGreaterThanEquals": max_loop_steps, "Next": limit_state},
        ],
        "Default": continue_state,
    }


def subagent_map_state(worker_map: dict[str, dict[str, Any]], next_state: str) -> dict[str, Any]:
    iterator_states = {
        "SelectSubagent": {
            "Type": "Choice",
            "Choices": [
                {"Variable": "$.logical_id", "StringEquals": "p_identity_subagent", "Next": "IdentitySubagent"},
                {"Variable": "$.logical_id", "StringEquals": "p_network_subagent", "Next": "NetworkSubagent"},
                {"Variable": "$.logical_id", "StringEquals": "p_communications_subagent", "Next": "CommunicationsSubagent"},
            ],
            "Default": "UnsupportedSubagent",
        },
        "IdentitySubagent": map_worker_state(worker_map["p_identity_subagent"]["function_arn"], "p_identity_subagent"),
        "NetworkSubagent": map_worker_state(worker_map["p_network_subagent"]["function_arn"], "p_network_subagent"),
        "CommunicationsSubagent": map_worker_state(worker_map["p_communications_subagent"]["function_arn"], "p_communications_subagent"),
        "UnsupportedSubagent": {"Type": "Fail", "Error": "UnsupportedSubagent", "Cause": "No Lambda worker exists for the requested subagent."},
    }
    return {
        "Type": "Map",
        "ItemsPath": "$.last_sends",
        "Parameters": {
            "checkpoint_ref.$": "$.checkpoint_ref",
            "logical_id.$": "$$.Map.Item.Value.logical_id",
            "task_input.$": "$$.Map.Item.Value.task_input",
        },
        "Iterator": {"StartAt": "SelectSubagent", "States": iterator_states},
        "ResultPath": "$.worker_results",
        "Next": next_state,
    }


def map_worker_state(function_arn: str, logical_id: str) -> dict[str, Any]:
    return {
        "Type": "Task",
        "Resource": function_arn,
        "Parameters": {
            "logical_id": logical_id,
            "checkpoint_ref.$": "$.checkpoint_ref",
            "read_keys": ["incident_id", "severity", "evidence_scope"],
            "task_input.$": "$.task_input",
        },
        "End": True,
    }


def wait_for_execution(
    states_client: Any,
    execution_arn: str,
    *,
    timeout_sec: float,
    poll_interval_sec: float,
) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    terminal = {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}
    while time.time() < deadline:
        description = states_client.describe_execution(executionArn=execution_arn)
        if description.get("status") in terminal:
            return description
        time.sleep(poll_interval_sec)
    raise TimeoutError(f"Step Functions execution did not finish within {timeout_sec}s: {execution_arn}")


def get_execution_history(states_client: Any, execution_arn: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    token = None
    while True:
        kwargs = {"executionArn": execution_arn, "includeExecutionData": False}
        if token:
            kwargs["nextToken"] = token
        response = states_client.get_execution_history(**kwargs)
        events.extend(response.get("events", []))
        token = response.get("nextToken")
        if not token:
            return events


def build_stepfunctions_metric(description: dict[str, Any], history: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    start_date = description.get("startDate")
    stop_date = description.get("stopDate")
    duration_ms = datetime_delta_ms(start_date, stop_date)
    transition_count = count_state_transitions(history)
    return {
        "execution_arn": description.get("executionArn"),
        "state_machine_arn": description.get("stateMachineArn"),
        "status": description.get("status"),
        "start_date": start_date.isoformat() if hasattr(start_date, "isoformat") else str(start_date),
        "stop_date": stop_date.isoformat() if hasattr(stop_date, "isoformat") else str(stop_date),
        "execution_duration_ms": duration_ms,
        "history_event_count": len(history),
        "state_transition_count": transition_count,
        "estimated_transition_cost_usd": round(transition_count * float(args.stepfunctions_transition_usd), 10),
        "state_duration_ms_json": json_compact(state_durations_from_history(history)),
    }


def count_state_transitions(history: list[dict[str, Any]]) -> int:
    entered_suffixes = ("StateEntered",)
    return sum(1 for event in history if str(event.get("type", "")).endswith(entered_suffixes))


def state_durations_from_history(history: list[dict[str, Any]]) -> dict[str, float]:
    entered: dict[str, list[Any]] = defaultdict(list)
    durations: dict[str, list[float]] = defaultdict(list)
    for event in history:
        event_type = str(event.get("type", ""))
        name = state_name_from_event(event)
        if not name:
            continue
        if event_type.endswith("StateEntered"):
            entered[name].append(event.get("timestamp"))
        elif event_type.endswith("StateExited") and entered[name]:
            start = entered[name].pop()
            duration = datetime_delta_ms(start, event.get("timestamp"))
            if duration is not None:
                durations[name].append(duration)
    return {name: round(sum(items), 3) for name, items in sorted(durations.items())}


def state_name_from_event(event: dict[str, Any]) -> str | None:
    for value in event.values():
        if isinstance(value, dict) and "name" in value:
            return str(value["name"])
    return None


def datetime_delta_ms(start: Any, stop: Any) -> float | None:
    if start is None or stop is None:
        return None
    return round((stop - start).total_seconds() * 1000.0, 3)


def enrich_payload_from_lambda_logs(
    logs_client: Any,
    payload: dict[str, Any],
    *,
    timeout_sec: float,
    start_time: Any,
) -> None:
    metrics = lambda_report_targets(payload)
    if not metrics:
        return
    deadline = time.time() + timeout_sec
    start_time_ms = int(start_time.timestamp() * 1000) - 60_000 if hasattr(start_time, "timestamp") else None
    unresolved = list(metrics)
    while unresolved and time.time() < deadline:
        next_unresolved = []
        for metric in unresolved:
            report = fetch_lambda_report(logs_client, metric, start_time_ms=start_time_ms)
            if not report:
                next_unresolved.append(metric)
                continue
            metric.update(report)
            metric["estimated_cost_usd"] = estimate_lambda_cost_usd(
                metric.get("configured_memory_mb"),
                metric.get("billed_duration_ms"),
            )
            metric["lambda_report_found"] = True
        unresolved = next_unresolved
        if unresolved:
            time.sleep(1.5)
    for metric in unresolved:
        metric["lambda_report_found"] = False


def lambda_report_targets(payload: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    targets.extend(metric for metric in payload.get("metrics", []) if metric.get("function_name") and metric.get("aws_request_id"))
    targets.extend(
        metric for metric in payload.get("barrier_metrics", []) if metric.get("function_name") and metric.get("aws_request_id")
    )
    coordinator = payload.get("coordinator_metric")
    if isinstance(coordinator, dict) and coordinator.get("function_name") and coordinator.get("aws_request_id"):
        targets.append(coordinator)
    return targets


def fetch_lambda_report(logs_client: Any, metric: dict[str, Any], *, start_time_ms: int | None) -> dict[str, Any] | None:
    function_name = str(metric["function_name"])
    request_id = str(metric["aws_request_id"])
    kwargs: dict[str, Any] = {
        "logGroupName": f"/aws/lambda/{function_name}",
        "filterPattern": f'"REPORT" "{request_id}"',
        "limit": 5,
    }
    if start_time_ms is not None:
        kwargs["startTime"] = start_time_ms
    try:
        response = logs_client.filter_log_events(**kwargs)
    except logs_client.exceptions.ResourceNotFoundException:
        return None
    for event in response.get("events", []):
        message = str(event.get("message") or "")
        if request_id in message and "REPORT" in message:
            return parse_report_log(message)
    return None


def refresh_payload_cost_totals(payload: dict[str, Any]) -> None:
    worker_cost = sum(float(metric.get("estimated_cost_usd") or 0.0) for metric in payload.get("metrics", []))
    barrier_cost = sum(float(metric.get("estimated_cost_usd") or 0.0) for metric in payload.get("barrier_metrics", []))
    coordinator_cost = float(payload.get("coordinator_metric", {}).get("estimated_cost_usd") or 0.0)
    payload["estimated_cost_usd"] = round(worker_cost, 10)
    payload["lambda_barrier_cost_estimated_usd"] = round(barrier_cost, 10)
    payload["lambda_total_estimated_cost_usd"] = round(worker_cost + barrier_cost + coordinator_cost, 10)


def build_summary_rows(entries: list[dict[str, Any]], manifest: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest_index = {
        (record["workflow"], record["candidate"]): record
        for record in manifest["records"]
    }
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[(entry["phase"], entry["workflow"], entry["candidate"], entry["dataset_id"])].append(entry)

    rows: list[dict[str, Any]] = []
    for (phase, workflow, candidate, dataset_id), group_entries in sorted(grouped.items()):
        payloads = [entry["payload"] for entry in group_entries]
        record = manifest_index[(workflow, candidate)]
        candidate_config = record["candidate_config"]
        worker_metrics = [metric for payload in payloads for metric in payload.get("metrics", [])]
        barrier_metrics = [metric for payload in payloads for metric in payload.get("barrier_metrics", [])]
        checkpoint_metrics = [metric for payload in payloads for metric in payload.get("checkpoint_trace", [])]
        coordinator_metrics = [
            payload.get("coordinator_metric", {})
            for payload in payloads
            if isinstance(payload.get("coordinator_metric"), dict) and payload.get("coordinator_metric")
        ]
        sfn_metrics = [
            payload.get("stepfunctions_metric", {})
            for payload in payloads
            if isinstance(payload.get("stepfunctions_metric"), dict) and payload.get("stepfunctions_metric")
        ]
        checkpoint_usage = summarize_checkpoint_usage(checkpoint_metrics, len(payloads), args)
        bedrock_usage = summarize_bedrock_usage(worker_metrics, len(payloads), args)
        worker_costs = [
            sum(float(metric.get("estimated_cost_usd") or 0.0) for metric in payload.get("metrics", []))
            for payload in payloads
        ]
        barrier_costs = [
            sum(float(metric.get("estimated_cost_usd") or 0.0) for metric in payload.get("barrier_metrics", []))
            for payload in payloads
        ]
        coordinator_costs = [
            float(payload.get("coordinator_metric", {}).get("estimated_cost_usd") or 0.0)
            for payload in payloads
        ]
        sfn_transition_costs = [
            float(payload.get("stepfunctions_metric", {}).get("estimated_transition_cost_usd") or 0.0)
            for payload in payloads
        ]
        lambda_total_costs = [
            worker + coord + barrier
            for worker, coord, barrier in zip(worker_costs, coordinator_costs, barrier_costs)
        ]
        platform_total_costs = [
            lambda_cost + sfn_cost + checkpoint_usage["checkpoint_cost_estimated_mean_usd"]
            for lambda_cost, sfn_cost in zip(lambda_total_costs, sfn_transition_costs)
        ]
        total_costs = [
            lambda_cost
            + sfn_cost
            + checkpoint_usage["checkpoint_cost_estimated_mean_usd"]
            + bedrock_usage["bedrock_cost_estimated_mean_usd"]
            for lambda_cost, sfn_cost in zip(lambda_total_costs, sfn_transition_costs)
        ]
        workflow_latencies = [payload.get("workflow_elapsed_ms") for payload in payloads]
        client_latencies = [payload.get("client_elapsed_ms") for payload in payloads]
        bedrock_critical_path_latencies = [bedrock_critical_path_latency_ms(payload) for payload in payloads]
        workflow_latencies_excluding_bedrock = [
            subtract_latency_ms(latency, bedrock_latency)
            for latency, bedrock_latency in zip(workflow_latencies, bedrock_critical_path_latencies)
        ]
        client_latencies_excluding_bedrock = [
            subtract_latency_ms(latency, bedrock_latency)
            for latency, bedrock_latency in zip(client_latencies, bedrock_critical_path_latencies)
        ]
        result_scores = [result_score(payload) for payload in payloads if result_score(payload) is not None]
        row = {
            "phase": phase,
            "orchestration_mode": record.get("orchestration") or ("stepfunctions" if record.get("state_machine") else "coordinator"),
            "workflow": workflow,
            "candidate": candidate,
            "candidate_group": candidate_config.get("group"),
            "slo_profile": candidate_config.get("slo_profile") or "",
            "dataset_id": dataset_id,
            "run_count": len(group_entries),
            "partitioning_vector": json_compact(candidate_config.get("partitioning_vector")),
            "memory_vector_mb": json_compact(candidate_config.get("memory_vector_mb")),
            "compiler_memory_vector_mb": json_compact(candidate_config.get("compiler_memory_vector_mb")),
            "timeout_vector_sec": json_compact([spec["timeout_sec"] for spec in candidate_config.get("worker_defs", [])]),
            "compiler_timeout_vector_sec": json_compact(
                [spec.get("compiler_timeout_sec", spec["timeout_sec"]) for spec in candidate_config.get("worker_defs", [])]
            ),
            "concurrency_vector": json_compact([spec.get("concurrency_limit") for spec in candidate_config.get("worker_defs", [])]),
            "deployed_partition_count": len(candidate_config.get("worker_defs", [])),
            "workflow_latency_mean_ms": mean_or_none(workflow_latencies),
            "workflow_latency_p95_ms": percentile(workflow_latencies, 0.95),
            "workflow_latency_including_bedrock_mean_ms": mean_or_none(workflow_latencies),
            "workflow_latency_including_bedrock_p95_ms": percentile(workflow_latencies, 0.95),
            "workflow_latency_excluding_bedrock_mean_ms": mean_or_none(workflow_latencies_excluding_bedrock),
            "workflow_latency_excluding_bedrock_p95_ms": percentile(workflow_latencies_excluding_bedrock, 0.95),
            "client_latency_mean_ms": mean_or_none(client_latencies),
            "client_latency_p95_ms": percentile(client_latencies, 0.95),
            "client_latency_including_bedrock_mean_ms": mean_or_none(client_latencies),
            "client_latency_including_bedrock_p95_ms": percentile(client_latencies, 0.95),
            "client_latency_excluding_bedrock_mean_ms": mean_or_none(client_latencies_excluding_bedrock),
            "client_latency_excluding_bedrock_p95_ms": percentile(client_latencies_excluding_bedrock, 0.95),
            "bedrock_latency_critical_path_mean_ms": mean_or_none(bedrock_critical_path_latencies),
            "bedrock_latency_critical_path_p95_ms": percentile(bedrock_critical_path_latencies, 0.95),
            "worker_invocations_mean": mean_or_none([len(payload.get("metrics", [])) for payload in payloads]),
            "worker_duration_mean_ms": mean_or_none([metric.get("duration_ms") for metric in worker_metrics]),
            "worker_billed_duration_mean_ms": mean_or_none([metric.get("billed_duration_ms") for metric in worker_metrics]),
            "worker_init_rate": rate([metric.get("init_duration_ms") is not None for metric in worker_metrics]),
            "worker_init_mean_ms": mean_or_none([metric.get("init_duration_ms") for metric in worker_metrics]),
            "worker_max_memory_used_mb": max_or_none([metric.get("max_memory_used_mb") for metric in worker_metrics]),
            "coordinator_duration_mean_ms": mean_or_none([metric.get("duration_ms") for metric in coordinator_metrics]),
            "coordinator_billed_duration_mean_ms": mean_or_none([metric.get("billed_duration_ms") for metric in coordinator_metrics]),
            "coordinator_init_rate": rate([metric.get("init_duration_ms") is not None for metric in coordinator_metrics]),
            "coordinator_init_mean_ms": mean_or_none([metric.get("init_duration_ms") for metric in coordinator_metrics]),
            "barrier_invocations_mean": round(len(barrier_metrics) / max(len(payloads), 1), 3),
            "barrier_duration_mean_ms": mean_or_none([metric.get("duration_ms") for metric in barrier_metrics]),
            "barrier_billed_duration_mean_ms": mean_or_none([metric.get("billed_duration_ms") for metric in barrier_metrics]),
            "barrier_init_rate": rate([metric.get("init_duration_ms") is not None for metric in barrier_metrics]),
            "barrier_init_mean_ms": mean_or_none([metric.get("init_duration_ms") for metric in barrier_metrics]),
            "lambda_worker_cost_mean_usd": mean_or_none(worker_costs, digits=10),
            "lambda_barrier_cost_mean_usd": mean_or_none(barrier_costs, digits=10),
            "lambda_coordinator_cost_mean_usd": mean_or_none(coordinator_costs, digits=10),
            "lambda_total_cost_mean_usd": mean_or_none(lambda_total_costs, digits=10),
            "stepfunctions_execution_duration_mean_ms": mean_or_none(
                [metric.get("execution_duration_ms") for metric in sfn_metrics]
            ),
            "stepfunctions_transition_count_mean": mean_or_none(
                [metric.get("state_transition_count") for metric in sfn_metrics]
            ),
            "stepfunctions_cost_estimated_mean_usd": mean_or_none(sfn_transition_costs, digits=10),
            "platform_cost_estimated_mean_usd": mean_or_none(platform_total_costs, digits=10),
            "total_cost_estimated_mean_usd": mean_or_none(total_costs, digits=10),
            "cost_excluding_bedrock_estimated_mean_usd": mean_or_none(platform_total_costs, digits=10),
            "cost_including_bedrock_estimated_mean_usd": mean_or_none(total_costs, digits=10),
            "workflow_failure_rate": rate(
                [
                    bool(
                        payload.get("error")
                        or payload.get("lambda_function_error")
                        or payload.get("stepfunctions_metric", {}).get("status") not in (None, "SUCCEEDED")
                    )
                    for payload in payloads
                ]
            ),
            "timeout_run_rate": rate([int(payload.get("timeout_count", 0) or 0) > 0 for payload in payloads]),
            "worker_error_rate": rate([bool(metric.get("function_error")) for metric in worker_metrics]),
            "worker_timeout_rate": rate([bool(metric.get("timed_out")) for metric in worker_metrics]),
            "lambda_report_missing_count": sum(
                1
                for metric in [*worker_metrics, *barrier_metrics, *coordinator_metrics]
                if metric.get("lambda_report_found") is False
            ),
            "result_correct_rate": mean_or_none(result_scores, digits=6),
            "partition_duration_mean_ms_json": json_compact(mean_metric_by_partition(payloads, "duration_ms")),
            "partition_billed_duration_mean_ms_json": json_compact(mean_metric_by_partition(payloads, "billed_duration_ms")),
            "partition_cost_mean_usd_json": json_compact(mean_metric_by_partition(payloads, "estimated_cost_usd", digits=10)),
            "partition_bedrock_latency_mean_ms_json": json_compact(mean_metric_by_partition(payloads, "remote_latency_ms")),
            "partition_request_pickle_bytes_json": json_compact(mean_metric_by_partition(payloads, "request_pickle_bytes")),
            "partition_state_pickle_bytes_json": json_compact(mean_metric_by_partition(payloads, "state_pickle_bytes")),
            "partition_writes_pickle_bytes_json": json_compact(mean_metric_by_partition(payloads, "writes_pickle_bytes")),
            "partition_send_count_json": json_compact(mean_metric_by_partition(payloads, "send_count")),
            "batch_count": len({entry["batch_index"] for entry in group_entries if entry.get("batch_index") is not None}),
            "batch_concurrency": next((entry.get("batch_concurrency") for entry in group_entries if entry.get("batch_concurrency")), ""),
            "batch_makespan_mean_ms": mean_or_none([entry.get("batch_makespan_ms") for entry in group_entries]),
            "batch_throughput_mean_rps": mean_or_none([entry.get("batch_throughput_rps") for entry in group_entries], digits=6),
        }
        row.update(checkpoint_usage)
        row.update(bedrock_usage)
        rows.append(row)
    return rows


def summarize_checkpoint_usage(metrics: list[dict[str, Any]], run_count: int, args: argparse.Namespace) -> dict[str, Any]:
    reads = [metric for metric in metrics if metric.get("operation") == "read"]
    writes = [metric for metric in metrics if metric.get("operation") == "write"]
    read_bytes = sum(float(metric.get("bytes") or 0.0) for metric in reads)
    write_bytes = sum(float(metric.get("bytes") or 0.0) for metric in writes)
    read_gb = read_bytes / (1024.0 ** 3)
    write_gb = write_bytes / (1024.0 ** 3)
    total_cost = (
        len(reads) * float(args.checkpoint_read_request_usd)
        + len(writes) * float(args.checkpoint_write_request_usd)
        + read_gb * float(args.checkpoint_read_gb_usd)
        + write_gb * float(args.checkpoint_write_gb_usd)
    )
    divisor = max(run_count, 1)
    return {
        "checkpoint_read_count_mean": round(len(reads) / divisor, 3),
        "checkpoint_write_count_mean": round(len(writes) / divisor, 3),
        "checkpoint_read_bytes_mean": round(read_bytes / divisor, 3),
        "checkpoint_write_bytes_mean": round(write_bytes / divisor, 3),
        "checkpoint_read_latency_mean_ms": mean_or_none([metric.get("latency_ms") for metric in reads]),
        "checkpoint_write_latency_mean_ms": mean_or_none([metric.get("latency_ms") for metric in writes]),
        "checkpoint_backend_json": json_compact(sorted({str(metric.get("backend")) for metric in metrics})),
        "checkpoint_cost_estimated_mean_usd": round(total_cost / divisor, 10),
    }


def summarize_bedrock_usage(metrics: list[dict[str, Any]], run_count: int, args: argparse.Namespace) -> dict[str, Any]:
    input_tokens = sum(float(metric.get("remote_input_tokens") or 0.0) for metric in metrics)
    output_tokens = sum(float(metric.get("remote_output_tokens") or 0.0) for metric in metrics)
    total_tokens = sum(float(metric.get("remote_total_tokens") or 0.0) for metric in metrics)
    call_count = sum(float(metric.get("remote_call_count") or 0.0) for metric in metrics)
    latency = sum(float(metric.get("remote_latency_ms") or 0.0) for metric in metrics)
    total_cost = (
        (input_tokens / 1000.0) * float(args.bedrock_input_1k_token_usd)
        + (output_tokens / 1000.0) * float(args.bedrock_output_1k_token_usd)
    )
    divisor = max(run_count, 1)
    return {
        "bedrock_call_count_mean": round(call_count / divisor, 3),
        "bedrock_latency_total_mean_ms": round(latency / divisor, 3),
        "bedrock_input_tokens_mean": round(input_tokens / divisor, 3),
        "bedrock_output_tokens_mean": round(output_tokens / divisor, 3),
        "bedrock_total_tokens_mean": round(total_tokens / divisor, 3),
        "bedrock_cost_estimated_mean_usd": round(total_cost / divisor, 10),
    }


def bedrock_critical_path_latency_ms(payload: dict[str, Any]) -> float:
    metrics = list(payload.get("metrics") or [])
    if not metrics:
        return 0.0
    if not any(metric.get("apply_step_index") is not None for metric in metrics):
        return round(sum(float(metric.get("remote_latency_ms") or 0.0) for metric in metrics), 3)

    grouped: dict[str, list[float]] = defaultdict(list)
    for index, metric in enumerate(metrics):
        latency = float(metric.get("remote_latency_ms") or 0.0)
        if latency <= 0.0:
            continue
        step = metric.get("apply_step_index")
        group_key = str(step) if step is not None else f"metric:{index}"
        grouped[group_key].append(latency)
    return round(sum(max(values) for values in grouped.values() if values), 3)


def subtract_latency_ms(total_latency_ms: Any, excluded_latency_ms: Any) -> float | None:
    if total_latency_ms is None:
        return None
    total = float(total_latency_ms)
    excluded = float(excluded_latency_ms or 0.0)
    return round(max(0.0, total - excluded), 3)


def selected_workflows(args: argparse.Namespace) -> dict[str, Any]:
    modules = experiment_modules()
    wanted = set(flatten_filters(args.workflow))
    if not wanted:
        return modules
    missing = sorted(wanted - set(modules))
    if missing:
        raise SystemExit(f"Unknown workflow(s): {', '.join(missing)}")
    return {workflow_id: modules[workflow_id] for workflow_id in modules if workflow_id in wanted}


def selected_candidates(args: argparse.Namespace, module: Any) -> list[Any]:
    candidates = get_candidates(module)
    wanted = set(flatten_filters(args.candidate))
    if not wanted:
        return candidates
    filtered = [candidate for candidate in candidates if candidate.candidate in wanted]
    return filtered


def selected_datasets(args: argparse.Namespace, module: Any) -> list[str]:
    datasets = get_datasets(module)
    wanted = set(flatten_filters(args.dataset))
    if not wanted:
        return sorted(datasets)
    return [dataset_id for dataset_id in sorted(datasets) if dataset_id in wanted]


def selected_manifest_records(args: argparse.Namespace, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    workflows = set(flatten_filters(args.workflow))
    candidates = set(flatten_filters(args.candidate))
    records = []
    for record in manifest.get("records", []):
        if workflows and record["workflow"] not in workflows:
            continue
        if candidates and record["candidate"] not in candidates:
            continue
        records.append(record)
    return records


def compile_current_plans(
    output_dir: Path,
    selected: dict[str, Any],
    selected_candidate_ids: set[str],
    *,
    use_bedrock: bool,
) -> list[dict[str, Any]]:
    from orqis.compiler.passes import compile_graph
    from orqis.compiler.report import write_artifacts

    artifact_records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    with compiler_bedrock_context(use_bedrock):
        for workflow_id, module in selected.items():
            graph_factory, sample_factory = compiler_inputs_for_workflow(workflow_id)
            for candidate in get_candidates(module):
                if selected_candidate_ids and candidate.candidate not in selected_candidate_ids:
                    continue
                if not candidate.group.startswith("compiler"):
                    continue
                resource_policy = "baseline" if candidate.group == "compiler baseline" else "om2"
                slo_profile = candidate.slo_profile or "prototype"
                candidate_dir = output_dir / workflow_id / candidate.candidate
                bundle = compile_graph(
                    graph_factory,
                    graph_id=workflow_id,
                    sample_input=sample_factory(),
                    resource_policy=resource_policy,
                    slo_profile=slo_profile,
                )
                report_path = write_artifacts(bundle, candidate_dir)
                artifact_records.append(
                    {
                        "workflow": workflow_id,
                        "candidate": candidate.candidate,
                        "resource_policy": resource_policy,
                        "slo_profile": slo_profile,
                        "artifact_dir": str(candidate_dir),
                        "report": str(report_path),
                        "compiler_partition_count": len(bundle.lgir2.partitions),
                        "compiler_partition_ids": sorted(bundle.lgir2.partitions),
                        "srv_plan_mode": bundle.srv_plan.orchestration.get("mode"),
                    }
                )
    return artifact_records


def validate_compiler_rows_against_artifacts(
    output_dir: Path,
    selected: dict[str, Any],
    selected_candidate_ids: set[str],
) -> None:
    errors: list[str] = []
    for workflow_id, module in selected.items():
        for candidate in get_candidates(module):
            if selected_candidate_ids and candidate.candidate not in selected_candidate_ids:
                continue
            if not candidate.group.startswith("compiler"):
                continue
            lgir2_path = output_dir / workflow_id / candidate.candidate / "lgir2.json"
            if not lgir2_path.exists():
                errors.append(f"{workflow_id} {candidate.candidate}: missing {lgir2_path}")
                continue
            try:
                payload = read_json(lgir2_path)
            except json.JSONDecodeError as exc:
                errors.append(f"{workflow_id} {candidate.candidate}: invalid JSON in {lgir2_path}: {exc}")
                continue
            compiler_ids = set((payload.get("partitions") or {}).keys())
            deployed_ids = {spec.logical_id for spec in candidate.worker_defs}
            if compiler_ids != deployed_ids:
                missing = sorted(compiler_ids - deployed_ids)
                extra = sorted(deployed_ids - compiler_ids)
                errors.append(
                    f"{workflow_id} {candidate.candidate}: candidate workers do not match fresh LGIR-2 "
                    f"partitions; missing={missing}, extra={extra}"
                )
            srv_plan_path = output_dir / workflow_id / candidate.candidate / "srv_plan.json"
            if not srv_plan_path.exists():
                errors.append(f"{workflow_id} {candidate.candidate}: missing {srv_plan_path}")
                continue
            try:
                srv_plan = read_json(srv_plan_path)
            except json.JSONDecodeError as exc:
                errors.append(f"{workflow_id} {candidate.candidate}: invalid JSON in {srv_plan_path}: {exc}")
                continue
            compiler_workers = dict((srv_plan.get("compute") or {}).get("workers") or {})
            candidate_workers = {spec.logical_id: spec for spec in candidate.worker_defs}
            if set(compiler_workers) != set(candidate_workers):
                missing = sorted(set(compiler_workers) - set(candidate_workers))
                extra = sorted(set(candidate_workers) - set(compiler_workers))
                errors.append(
                    f"{workflow_id} {candidate.candidate}: candidate workers do not match fresh SRV-Plan "
                    f"workers; missing={missing}, extra={extra}"
                )
                continue
            for worker_id, compiler_worker in sorted(compiler_workers.items()):
                candidate_worker = candidate_workers[worker_id]
                expected = {
                    "memory_mb": normalize_optional_int(compiler_worker.get("memory_mb")),
                    "timeout_sec": normalize_optional_int(compiler_worker.get("timeout_sec")),
                    "concurrency_limit": normalize_optional_int(compiler_worker.get("concurrency_limit")),
                }
                deployed = {
                    "memory_mb": normalize_optional_int(candidate_worker.memory_mb),
                    "timeout_sec": normalize_optional_int(candidate_worker.timeout_sec),
                    "concurrency_limit": normalize_optional_int(candidate_worker.concurrency_limit),
                }
                if expected != deployed:
                    errors.append(
                        f"{workflow_id} {candidate.candidate} {worker_id}: candidate resources do not match "
                        f"fresh SRV-Plan; expected={expected}, deployed={deployed}"
                    )
    if errors:
        raise SystemExit("Compiler artifact validation failed:\n" + "\n".join(errors))


def normalize_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def compiler_inputs_for_workflow(workflow_id: str) -> tuple[Any, Any]:
    if workflow_id == "service_desk_router":
        from orqis.examples.final_experiments import router

        return router.build_graph, router.get_sample_input
    if workflow_id == "skills":
        from orqis.examples.final_experiments import skills

        return skills.build_graph, skills.get_sample_input
    if workflow_id == "incident_response_swarm":
        from orqis.examples.final_experiments import subagents

        return subagents.build_graph, subagents.get_sample_input
    raise ValueError(f"unsupported compiler workflow: {workflow_id}")


@contextmanager
def compiler_bedrock_context(use_bedrock: bool):
    if use_bedrock:
        yield
        return
    keys = ("ORQIS_BEDROCK_MODEL_ID", "ORQIS_BEDROCK_REGION")
    previous = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def experiment_modules() -> dict[str, Any]:
    from orqis.examples.final_experiments import router_experiment, skills_experiment, subagents_experiment

    return {
        router_experiment.WORKFLOW_ID: router_experiment,
        skills_experiment.WORKFLOW_ID: skills_experiment,
        subagents_experiment.WORKFLOW_ID: subagents_experiment,
    }


def get_candidates(module: Any) -> list[Any]:
    for name in ("ROUTER_CANDIDATES", "SKILLS_CANDIDATES", "SUBAGENT_CANDIDATES"):
        if hasattr(module, name):
            return list(getattr(module, name))
    raise AttributeError(f"candidate list not found in {module.__name__}")


def get_datasets(module: Any) -> dict[str, dict[str, Any]]:
    for name in ("ROUTER_DATASETS", "SKILLS_DATASETS", "SUBAGENT_DATASETS"):
        if hasattr(module, name):
            return dict(getattr(module, name))
    raise AttributeError(f"dataset map not found in {module.__name__}")


def serialize_candidate(candidate: Any, worker_specs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    resolved_worker_specs = worker_specs or [serialize_worker_spec(spec) for spec in candidate.worker_defs]
    return {
        "workflow": candidate.workflow,
        "group": candidate.group,
        "candidate": candidate.candidate,
        "partitioning_vector": list(candidate.partitioning_vector),
        "memory_vector_mb": [int(spec["memory_mb"]) for spec in resolved_worker_specs],
        "compiler_memory_vector_mb": list(candidate.memory_vector_mb),
        "run_on": candidate.run_on,
        "slo_profile": candidate.slo_profile,
        "worker_defs": resolved_worker_specs,
    }


def serialize_worker_spec(spec: Any, *, min_timeout_sec: int = 0, max_memory_mb: int | None = None) -> dict[str, Any]:
    if is_dataclass(spec):
        payload = asdict(spec)
    else:
        payload = dict(spec)
    compiler_timeout_sec = int(payload.get("timeout_sec") or 180)
    timeout_sec = max(compiler_timeout_sec, int(min_timeout_sec or 0))
    compiler_memory_mb = int(payload["memory_mb"])
    memory_mb = min(compiler_memory_mb, int(max_memory_mb)) if max_memory_mb else compiler_memory_mb
    return {
        "logical_id": payload["logical_id"],
        "memory_mb": memory_mb,
        "compiler_memory_mb": compiler_memory_mb,
        "timeout_sec": timeout_sec,
        "compiler_timeout_sec": compiler_timeout_sec,
        "concurrency_limit": payload.get("concurrency_limit"),
    }


def effective_worker_specs(candidate: Any, *, min_timeout_sec: int, max_memory_mb: int | None) -> list[dict[str, Any]]:
    return [
        serialize_worker_spec(spec, min_timeout_sec=min_timeout_sec, max_memory_mb=max_memory_mb)
        for spec in candidate.worker_defs
    ]


def build_worker_env(args: argparse.Namespace, workflow: str, candidate: str, logical_id: str) -> dict[str, str]:
    env = {
        "ORQIS_LAMBDA_ROLE": "worker",
        "ORQIS_WORKFLOW": workflow,
        "ORQIS_CANDIDATE": candidate,
        "ORQIS_PARTITION_ID": logical_id,
        "ORQIS_STATE_INLINE_LIMIT_BYTES": str(args.state_inline_limit_bytes),
        "ORQIS_MAX_LOOP_STEPS": str(args.max_loop_steps),
        "ORQIS_BEDROCK_TEMPERATURE": str(args.bedrock_temperature),
        "ORQIS_BEDROCK_MAX_OUTPUT_TOKENS": str(args.bedrock_max_output_tokens),
    }
    add_optional_env(env, "ORQIS_CHECKPOINT_TABLE", args.state_table)
    add_optional_env(env, "ORQIS_CHECKPOINT_BUCKET", args.state_bucket)
    add_optional_env(env, "ORQIS_BEDROCK_MODEL_ID", args.bedrock_model_id)
    add_optional_env(env, "ORQIS_BEDROCK_REGION", args.bedrock_region or args.region)
    return env


def build_coordinator_env(args: argparse.Namespace, candidate: dict[str, Any], worker_map: dict[str, Any]) -> dict[str, str]:
    env = {
        "ORQIS_LAMBDA_ROLE": "coordinator",
        "ORQIS_CANDIDATE_JSON": json_compact(candidate),
        "ORQIS_WORKER_MAP_JSON": json_compact(worker_map),
        "ORQIS_STATE_INLINE_LIMIT_BYTES": str(args.state_inline_limit_bytes),
        "ORQIS_MAX_LOOP_STEPS": str(args.max_loop_steps),
        "ORQIS_BEDROCK_TEMPERATURE": str(args.bedrock_temperature),
        "ORQIS_BEDROCK_MAX_OUTPUT_TOKENS": str(args.bedrock_max_output_tokens),
    }
    add_optional_env(env, "ORQIS_CHECKPOINT_TABLE", args.state_table)
    add_optional_env(env, "ORQIS_CHECKPOINT_BUCKET", args.state_bucket)
    add_optional_env(env, "ORQIS_BEDROCK_MODEL_ID", args.bedrock_model_id)
    add_optional_env(env, "ORQIS_BEDROCK_REGION", args.bedrock_region or args.region)
    return env


def build_barrier_env(args: argparse.Namespace, candidate: dict[str, Any]) -> dict[str, str]:
    env = {
        "ORQIS_LAMBDA_ROLE": "barrier",
        "ORQIS_CANDIDATE_JSON": json_compact(candidate),
        "ORQIS_STATE_INLINE_LIMIT_BYTES": str(args.state_inline_limit_bytes),
        "ORQIS_MAX_LOOP_STEPS": str(args.max_loop_steps),
    }
    add_optional_env(env, "ORQIS_CHECKPOINT_TABLE", args.state_table)
    add_optional_env(env, "ORQIS_CHECKPOINT_BUCKET", args.state_bucket)
    return env


def add_optional_env(env: dict[str, str], key: str, value: Any) -> None:
    if value is not None and str(value):
        env[key] = str(value)


def build_lambda_package(package_path: Path, dependency_dirs: list[Path]) -> None:
    package_dir = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "package"
        staged.mkdir()
        for dependency_dir in dependency_dirs:
            copy_dependency_dir(dependency_dir, staged)
        target_package_dir = staged / "orqis"
        for path in package_dir.rglob("*.py"):
            if should_skip_source(path):
                continue
            destination = target_package_dir / path.relative_to(package_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in staged.rglob("*"):
                if path.is_file() and not should_skip_archive_file(path):
                    archive.write(path, path.relative_to(staged))


def copy_dependency_dir(dependency_dir: Path, staged: Path) -> None:
    if not dependency_dir.exists():
        raise SystemExit(f"Dependency directory does not exist: {dependency_dir}")
    for path in dependency_dir.rglob("*"):
        if path.is_dir() or should_skip_archive_file(path):
            continue
        destination = staged / path.relative_to(dependency_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def should_skip_source(path: Path) -> bool:
    parts = set(path.parts)
    return "__pycache__" in parts or ".git" in parts


def should_skip_archive_file(path: Path) -> bool:
    return path.suffix in {".pyc", ".pyo"} or "__pycache__" in set(path.parts)


def upsert_lambda_function(
    lambda_client: Any,
    *,
    function_name: str,
    role_arn: str,
    package_path: Path,
    handler: str,
    runtime: str,
    architecture: str,
    memory_mb: int,
    timeout_sec: int,
    environment: dict[str, str],
    description: str,
) -> str:
    package_bytes = package_path.read_bytes()
    try:
        current = lambda_client.get_function(FunctionName=function_name)
    except lambda_client.exceptions.ResourceNotFoundException:
        try:
            created = lambda_client.create_function(
                FunctionName=function_name,
                Runtime=runtime,
                Role=role_arn,
                Handler=handler,
                Code={"ZipFile": package_bytes},
                Description=description,
                Timeout=timeout_sec,
                MemorySize=memory_mb,
                Architectures=[architecture],
                Environment={"Variables": environment},
            )
            wait_lambda_ready(lambda_client, function_name)
            return created["FunctionArn"]
        except lambda_client.exceptions.ResourceConflictException:
            wait_lambda_ready(lambda_client, function_name)
            current = lambda_client.get_function(FunctionName=function_name)

    lambda_client.update_function_code(FunctionName=function_name, ZipFile=package_bytes)
    wait_lambda_ready(lambda_client, function_name)
    update_config = {
        "FunctionName": function_name,
        "Runtime": runtime,
        "Role": role_arn,
        "Handler": handler,
        "Description": description,
        "Timeout": timeout_sec,
        "MemorySize": memory_mb,
        "Environment": {"Variables": environment},
    }
    if operation_supports_parameter(lambda_client, "UpdateFunctionConfiguration", "Architectures"):
        update_config["Architectures"] = [architecture]
    lambda_client.update_function_configuration(
        **update_config,
    )
    wait_lambda_ready(lambda_client, function_name)
    return current["Configuration"]["FunctionArn"]


def operation_supports_parameter(client: Any, operation_name: str, parameter_name: str) -> bool:
    operation_model = client.meta.service_model.operation_model(operation_name)
    return parameter_name in operation_model.input_shape.members


def wait_lambda_ready(lambda_client: Any, function_name: str, attempts: int = 90) -> None:
    for _attempt in range(attempts):
        config = lambda_client.get_function_configuration(FunctionName=function_name)
        state = config.get("State")
        update_status = config.get("LastUpdateStatus")
        if state == "Failed" or update_status == "Failed":
            reason = config.get("StateReason") or config.get("LastUpdateStatusReason")
            raise RuntimeError(f"Lambda {function_name} failed to become ready: {reason}")
        if state == "Active" and update_status in (None, "Successful"):
            return
        time.sleep(2)
    raise TimeoutError(f"Lambda {function_name} did not become ready.")


def ensure_state_table(ddb_client: Any, table_name: str) -> None:
    try:
        ddb_client.describe_table(TableName=table_name)
        return
    except ddb_client.exceptions.ResourceNotFoundException:
        pass
    except Exception as exc:
        if is_access_denied(exc):
            raise SystemExit(
                f"Cannot inspect DynamoDB checkpoint table `{table_name}`. "
                "Grant the deploying AWS identity `dynamodb:DescribeTable` and `dynamodb:CreateTable` "
                "for this table, or pre-create the table with partition key `run_id` (S) and sort key "
                "`checkpoint_id` (S), then rerun deploy with `--skip-state-resource-check`."
            ) from exc
        raise
    try:
        ddb_client.create_table(
            TableName=table_name,
            BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[
                {"AttributeName": "run_id", "AttributeType": "S"},
                {"AttributeName": "checkpoint_id", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "run_id", "KeyType": "HASH"},
                {"AttributeName": "checkpoint_id", "KeyType": "RANGE"},
            ],
        )
        waiter = ddb_client.get_waiter("table_exists")
        waiter.wait(TableName=table_name)
    except Exception as exc:
        if is_access_denied(exc):
            raise SystemExit(
                f"Cannot create DynamoDB checkpoint table `{table_name}`. "
                "Grant the deploying AWS identity `dynamodb:CreateTable`, `dynamodb:DescribeTable`, "
                "and `dynamodb:TagResource` if tagging is required by the account, or create the table "
                "outside this harness and rerun deploy with `--skip-state-resource-check`."
            ) from exc
        raise


def ensure_bucket(s3_client: Any, bucket_name: str, region: str) -> None:
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        return
    except Exception:
        pass
    kwargs: dict[str, Any] = {"Bucket": bucket_name}
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    try:
        s3_client.create_bucket(**kwargs)
    except Exception as exc:
        if is_access_denied(exc):
            raise SystemExit(
                f"Cannot create or inspect S3 checkpoint bucket `{bucket_name}`. "
                "Grant the deploying AWS identity S3 bucket-management permissions, or pre-create the "
                "bucket and rerun deploy with `--skip-state-resource-check`."
            ) from exc
        raise


def is_access_denied(exc: Exception) -> bool:
    return aws_error_code(exc) in {"AccessDenied", "AccessDeniedException", "UnauthorizedOperation"}


def aws_error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict):
            return str(error.get("Code") or "")
    return ""


def empty_and_delete_bucket(s3_client: Any, bucket_name: str) -> None:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name):
        objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if objects:
            s3_client.delete_objects(Bucket=bucket_name, Delete={"Objects": objects})
    s3_client.delete_bucket(Bucket=bucket_name)


def upsert_state_machine(
    states_client: Any,
    *,
    state_machine_name: str,
    state_machine_arn: str,
    role_arn: str,
    definition: dict[str, Any],
    state_machine_type: str,
) -> str:
    definition_json = json.dumps(definition, indent=2)
    try:
        states_client.describe_state_machine(stateMachineArn=state_machine_arn)
    except states_client.exceptions.StateMachineDoesNotExist:
        created = states_client.create_state_machine(
            name=state_machine_name,
            definition=definition_json,
            roleArn=role_arn,
            type=state_machine_type,
        )
        return created["stateMachineArn"]
    states_client.update_state_machine(
        stateMachineArn=state_machine_arn,
        definition=definition_json,
        roleArn=role_arn,
    )
    return state_machine_arn


def state_machine_arn_for(region: str, account_id: str, state_machine_name: str) -> str:
    return f"arn:aws:states:{region}:{account_id}:stateMachine:{state_machine_name}"


def upsert_role(
    iam: Any,
    *,
    role_name: str,
    service_principal: str,
    description: str,
    policy_name: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    assume_role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": service_principal},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    try:
        role = iam.get_role(RoleName=role_name)["Role"]
        iam.update_assume_role_policy(RoleName=role_name, PolicyDocument=json.dumps(assume_role_policy))
    except iam.exceptions.NoSuchEntityException:
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy),
            Description=description,
        )["Role"]
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=policy_name,
        PolicyDocument=json.dumps(policy),
    )
    return role


def build_execution_policy(
    *,
    region: str,
    account_id: str,
    prefix: str,
    state_table: str,
    state_bucket: str,
) -> dict[str, Any]:
    lambda_arn = f"arn:aws:lambda:{region}:{account_id}:function:{aws_name(prefix)}*"
    ddb_resources = []
    s3_resources = []
    if state_table:
        ddb_resources.append(f"arn:aws:dynamodb:{region}:{account_id}:table/{state_table}")
    if state_bucket:
        s3_resources.extend([f"arn:aws:s3:::{state_bucket}", f"arn:aws:s3:::{state_bucket}/*"])
    statements: list[dict[str, Any]] = [
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
            ],
            "Resource": f"arn:aws:logs:{region}:{account_id}:*",
        },
        {
            "Effect": "Allow",
            "Action": ["lambda:InvokeFunction"],
            "Resource": lambda_arn,
        },
        {
            "Effect": "Allow",
            "Action": ["bedrock:InvokeModel"],
            "Resource": "*",
        },
    ]
    if ddb_resources:
        statements.append(
            {
                "Effect": "Allow",
                "Action": ["dynamodb:DescribeTable", "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem", "dynamodb:Query"],
                "Resource": ddb_resources,
            }
        )
    if s3_resources:
        statements.append(
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                "Resource": s3_resources,
            }
        )
    return {"Version": "2012-10-17", "Statement": statements}


def build_stepfunctions_policy(*, region: str, account_id: str, prefix: str) -> dict[str, Any]:
    lambda_arn = f"arn:aws:lambda:{region}:{account_id}:function:{aws_name(prefix)}*"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["lambda:InvokeFunction"],
                "Resource": [lambda_arn, f"{lambda_arn}:*"],
            }
        ],
    }


def boto3_session(args: argparse.Namespace) -> Any:
    import boto3

    kwargs = {}
    if getattr(args, "profile", ""):
        kwargs["profile_name"] = args.profile
    if getattr(args, "region", None):
        kwargs["region_name"] = args.region
    return boto3.Session(**kwargs)


def deployed_function_names(manifest: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for record in manifest.get("records", []):
        if record.get("coordinator"):
            names.add(record["coordinator"]["function_name"])
        if record.get("barrier"):
            names.add(record["barrier"]["function_name"])
        for worker in record.get("workers", {}).values():
            names.add(worker["function_name"])
    return names


def deployed_state_machine_arns(manifest: dict[str, Any]) -> set[str]:
    arns: set[str] = set()
    for record in manifest.get("records", []):
        if record.get("state_machine"):
            arns.add(record["state_machine"]["arn"])
    return arns


def build_result_meta(payload: dict[str, Any]) -> dict[str, Any]:
    meta = payload.get("result_meta")
    return meta if isinstance(meta, dict) else {}


def result_score(payload: dict[str, Any]) -> float | None:
    meta = build_result_meta(payload)
    if "route_match" in meta:
        return 1.0 if meta.get("route_match") else 0.0
    if "task_success" in meta:
        return 1.0 if meta.get("task_success") else 0.0
    return None


def mean_metric_by_partition(payloads: list[dict[str, Any]], key: str, digits: int = 3) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for payload in payloads:
        for metric in payload.get("metrics", []):
            logical_id = str(metric.get("logical_id", ""))
            if metric.get(key) is not None:
                values[logical_id].append(float(metric[key]))
    return {logical_id: round(statistics.mean(items), digits) for logical_id, items in sorted(values.items()) if items}


def flatten_filters(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                result.append(item)
    return result


def parse_report_log(log_text: str) -> dict[str, float | int]:
    report: dict[str, float | int] = {}
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
    return base64.b64decode(value).decode("utf-8", errors="replace")


def estimate_lambda_cost_usd(memory_mb: int | float | None, billed_duration_ms: int | float | None) -> float | None:
    if memory_mb is None or billed_duration_ms is None:
        return None
    billed_seconds = float(billed_duration_ms) / 1000.0
    memory_gb = float(memory_mb) / 1024.0
    return round((memory_gb * billed_seconds * AWS_LAMBDA_GB_SECOND_USD) + AWS_LAMBDA_REQUEST_USD, 10)


def mean_or_none(values: list[Any], digits: int = 3) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(statistics.mean(numeric), digits)


def max_or_none(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return max(numeric)


def percentile(values: list[Any], fraction: float) -> float | None:
    numeric = sorted(float(value) for value in values if value is not None)
    if not numeric:
        return None
    if len(numeric) == 1:
        return round(numeric[0], 3)
    index = math_ceil(fraction * len(numeric)) - 1
    index = min(max(index, 0), len(numeric) - 1)
    return round(numeric[index], 3)


def math_ceil(value: float) -> int:
    return int(-(-value // 1))


def rate(values: list[Any]) -> float:
    if not values:
        return 0.0
    return round(sum(1 for value in values if value) / len(values), 6)


def checkpoint_cost_config(args: argparse.Namespace) -> dict[str, float]:
    return {
        "read_request_usd": float(args.checkpoint_read_request_usd),
        "write_request_usd": float(args.checkpoint_write_request_usd),
        "read_gb_usd": float(args.checkpoint_read_gb_usd),
        "write_gb_usd": float(args.checkpoint_write_gb_usd),
    }


def bedrock_cost_config(args: argparse.Namespace) -> dict[str, float]:
    return {
        "input_1k_token_usd": float(args.bedrock_input_1k_token_usd),
        "output_1k_token_usd": float(args.bedrock_output_1k_token_usd),
    }


def aws_name(*parts: Any, max_length: int = 64) -> str:
    raw = "-".join(str(part) for part in parts if str(part))
    name = re.sub(r"[^A-Za-z0-9-_]+", "-", raw).strip("-").lower()
    if len(name) <= max_length:
        return name
    suffix = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{name[: max_length - len(suffix) - 1].rstrip('-')}-{suffix}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: normalize_csv_value(row.get(key)) for key in fieldnames})


def normalize_csv_value(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return value
    return json_compact(value)


if __name__ == "__main__":
    raise SystemExit(main())
