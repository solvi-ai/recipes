"""Runs this recipe's scripts and compares their output with expected/ (offline: no model, no network)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def run(script, *args, env=None):
    out = subprocess.run([sys.executable, script, *args], cwd=HERE, capture_output=True, text=True, timeout=900,
                         env={**os.environ, **(env or {})})
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def expected(name):
    return (HERE / "expected" / name).read_text(encoding="utf-8")


def test_routing_toy():
    assert run("routing_toy.py") == expected("routing_toy.out.txt")


@pytest.mark.skipif(not (HERE / "data" / "banking77" / "test.csv").exists(),
                    reason="Banking77 not downloaded (python fetch_data.py); takes about 1.5 minutes")
def test_banking77_stream():
    assert run("banking77_stream.py") == expected("banking77_stream.out.txt")
