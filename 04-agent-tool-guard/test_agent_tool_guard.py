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


@pytest.mark.parametrize("script", ["cancel_order_guard", "payment_guard"])
def test_guard(script):
    assert run(f"{script}.py") == expected(f"{script}.out.txt")
