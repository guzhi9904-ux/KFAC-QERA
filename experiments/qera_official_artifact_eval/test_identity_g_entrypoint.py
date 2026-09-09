from __future__ import annotations

import subprocess
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent
ENTRYPOINT = ROOT / "run_identity_g.py"
CONFIG = ROOT / "configs/llama3.1-8b-server-identity-g.yaml"


def test_identity_g_entrypoint_help():
    result = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "official word-PPL evaluation" in result.stdout


def test_identity_g_config_binds_unchanged_engine():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["methods"] == ["wq", "diag_gi", "full_gi"]
    assert config["runner_mode"] == "identity-g-extension"
    assert config["evaluation_engine_sha256"] == "f84633b333b4bdeb995ee19eacba8e1542789b42fb995895c71aee95c1232e75"
    assert config["output_dir"].endswith("official-word-ppl-4096-existing-artifacts-identity-g")


def test_artifact_stage_defaults_to_all_identity_g_configurations(tmp_path):
    config = {
        "ranks": [8, 16],
        "artifact_runs": [{"name": "mxint4"}, {"name": "mxint3"}],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    script = (
        "import json,sys; "
        f"sys.path.insert(0, {str(ROOT.parent)!r}); "
        "from qera_official_artifact_eval.run_identity_g import _identity_only_arguments; "
        f"print(json.dumps(_identity_only_arguments(['--config',{str(path)!r},'evaluate','--stage','artifacts'])))"
    )
    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    arguments = __import__("json").loads(result.stdout)
    selected = [arguments[index + 1] for index, value in enumerate(arguments) if value == "--only"]
    assert selected == [
        "mxint4:DIAG_GI_R8",
        "mxint4:DIAG_GI_R16",
        "mxint4:FULL_GI_R8",
        "mxint4:FULL_GI_R16",
        "mxint3:DIAG_GI_R8",
        "mxint3:DIAG_GI_R16",
        "mxint3:FULL_GI_R8",
        "mxint3:FULL_GI_R16",
    ]
