#!/usr/bin/env python3
"""Run the identity-G extension without changing the completed diagonal-G protocol."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from qera_official_artifact_eval import run as engine


ENGINE_PATH = Path(engine.__file__).resolve()
ENGINE_SHA256 = engine.sha256_file(ENGINE_PATH)
IDENTITY_G_PROTOCOL = {
    **engine.FIXED_PROTOCOL,
    "methods": ["wq", "diag_gi", "full_gi"],
    "runner_mode": "identity-g-extension",
    "evaluation_engine_sha256": ENGINE_SHA256,
}

# The unchanged engine hashes __file__ into protocol.json. Bind that field to
# this entrypoint and bind the imported engine separately in FIXED_PROTOCOL.
engine.FIXED_PROTOCOL = IDENTITY_G_PROTOCOL
engine.__file__ = str(Path(__file__).resolve())


def _argument_value(arguments: list[str], option: str) -> str:
    try:
        index = arguments.index(option)
        return arguments[index + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"Missing required {option}") from error


def _identity_only_arguments(arguments: list[str]) -> list[str]:
    """Default the artifact stage to GI only while retaining Wq as a loader dependency."""
    if "evaluate" not in arguments or "--only" in arguments:
        return arguments
    stage = _argument_value(arguments, "--stage") if "--stage" in arguments else "bf16"
    if stage != "artifacts":
        return arguments
    config_path = Path(_argument_value(arguments, "--config"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    additions = []
    for artifact in config["artifact_runs"]:
        for method in ("DIAG_GI", "FULL_GI"):
            for rank in config["ranks"]:
                additions.extend(("--only", f"{artifact['name']}:{method}_R{rank}"))
    return [*arguments, *additions]


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        arguments = _identity_only_arguments(arguments)
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(json.dumps({"status": "ERROR", "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return engine.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
