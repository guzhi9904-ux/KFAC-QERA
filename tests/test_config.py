from pathlib import Path

from qera_exp.config import load_config


def test_config_inheritance_and_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MODEL_PATH", "/models/test")
    monkeypatch.setenv("RUN_DIR", str(tmp_path / "run"))
    base = tmp_path / "base.yaml"
    base.write_text(
        """
schema_version: 1
experiment: {output_dir: "${RUN_DIR}"}
model:
  name_or_path: "${MODEL_PATH}"
  device: cuda:0
  dtype: bfloat16
  target_suffixes: [self_attn.q_proj]
quantization: {width: 4, block_size: 32}
statistics:
  methods: [AD_GI]
  ranks: [8]
  maximum_rank: 8
data:
  calibration: {split: train, text_column: text, sequence_length: 32, windows: 2}
  wikitext2: {split: test, text_column: text, sequence_length: 32, windows: 2}
  c4: {split: validation, text_column: text, sequence_length: 32, windows: 2}
""".strip(),
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\nstatistics:\n  ranks: [4, 8]\n", encoding="utf-8")
    config = load_config(child)
    assert config["model"]["name_or_path"] == "/models/test"
    assert config["experiment"]["output_dir"] == str(tmp_path / "run")
    assert config["statistics"]["ranks"] == [4, 8]
