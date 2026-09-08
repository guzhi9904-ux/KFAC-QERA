from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import load_config, validate_config
from .pipeline import (
    collect_shard,
    compute_roots,
    create_plan,
    doctor,
    evaluate,
    prepare_data,
    run_all,
    solve_shard,
    write_evaluation_tables,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Official-QERA-aligned Full-A versus diagonal-A isolation")
    result.add_argument("--config", required=True, type=Path)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    data = commands.add_parser("prepare-data")
    data.add_argument("--role", choices=["all", "calibration", "wikitext2"], default="all")
    data.add_argument("--allow-download", action="store_true")
    commands.add_parser("plan")
    for name in ("collect", "roots", "solve"):
        command = commands.add_parser(name)
        command.add_argument("--shard", required=True, type=int)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--only", help="One configuration, e.g. QERA_FULL_R32")
    evaluation.add_argument("--dual-gpu", action="store_true", help="Balance evaluation weights over two GPUs")
    evaluation.add_argument("--batch-size", type=int, help="Evaluation-only batch size; preserves the saved plan")
    evaluation.add_argument("--ce-chunk-tokens", type=int, default=2048, help="CE token rows per window (does not change context length)")
    commands.add_parser("summarize")
    commands.add_parser("run")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    validate_config(config)
    match args.command:
        case "doctor":
            output = doctor(config)
        case "prepare-data":
            roles = ("calibration", "wikitext2") if args.role == "all" else (args.role,)
            output = prepare_data(config, roles, args.allow_download)
        case "plan":
            output = create_plan(config)
        case "collect":
            output = collect_shard(config, args.shard)
        case "roots":
            output = compute_roots(config, args.shard)
        case "solve":
            output = solve_shard(config, args.shard)
        case "evaluate":
            output = evaluate(
                config, args.only, dual_gpu=args.dual_gpu,
                batch_size=args.batch_size, ce_chunk_tokens=args.ce_chunk_tokens,
            )
        case "summarize":
            output = write_evaluation_tables(config)
        case "run":
            output = run_all(config)
        case _:
            raise AssertionError(args.command)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if isinstance(output, dict) and output.get("status") == "FAIL":
        return 1
    if isinstance(output, dict) and output.get("status") == "NEEDS_DATA":
        return 2
    return 0
