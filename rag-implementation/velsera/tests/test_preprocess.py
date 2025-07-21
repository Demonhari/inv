from pathlib import Path
import json, subprocess

ROOT = Path(__file__).resolve().parents[1]

def test_preprocess_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("OUT_DIR", str(tmp_path))
    subprocess.check_call(["python", "preprocess.py"], cwd=ROOT)

    assert (tmp_path / "dataset.parquet").exists()
    for split in ("train.jsonl", "test.jsonl"):
        f = tmp_path / split
        assert f.exists() and f.stat().st_size > 0
        # quick schema check
        rec = json.loads(f.open().readline())
        assert {"abstract", "label"} <= rec.keys()
