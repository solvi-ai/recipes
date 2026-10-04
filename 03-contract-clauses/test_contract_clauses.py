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


def test_quotes_offline():
    """Part 2 of contract_clauses.py: a typographic variant of a quote is found, a paraphrase is rejected."""
    out = run("contract_clauses.py", "--offline", env={"LLM_URL": ""})
    assert out.startswith("Part 1 skipped")
    part2 = expected("contract_clauses.out.txt").split("\n\n", 1)[1]
    assert out.split("\n\n", 1)[1] == part2


@pytest.mark.skipif(not os.environ.get("LLM_URL"), reason="needs an LLM: set LLM_URL (and LLM_KEY)")
def test_llm_part():
    """With a model the answers may differ from expected/ (another model, another day): check the invariants only."""
    out = run("contract_clauses.py")
    lines = [x for x in out.splitlines() if x.startswith("  ") and "contract[" in x]
    assert lines and all("literal: True" in x for x in lines)
