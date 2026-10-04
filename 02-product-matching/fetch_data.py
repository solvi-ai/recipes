"""Download Abt-Buy into ./data/abtbuy/ at run time. The data is NOT in this repository and must not be added to it.

    python fetch_data.py

Source: the copy `matchbench/Abt-Buy` on Hugging Face (tableA.csv = Abt offers, tableB.csv = Buy offers, train / valid /
test.csv = labelled candidate pairs), about 1 MB; the benchmark goes back to Köpcke, Thor, Rahm, "Evaluation of entity
resolution approaches on real-world match problems", VLDB 2010 (Leipzig benchmark datasets).
Licence: none stated. That is why this recipe only ever downloads it, and keeps neither the offers nor anything
quoting them (outputs here print counts and scores only)."""
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/matchbench/Abt-Buy/resolve/main/"
OUT = Path(__file__).resolve().parent / "data" / "abtbuy"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("tableA.csv", "tableB.csv", "train.csv", "valid.csv", "test.csv"):
        dest = OUT / name
        if not dest.exists() or dest.stat().st_size == 0:
            urllib.request.urlretrieve(BASE + name, dest)
        print(f"{dest} ({dest.stat().st_size:,} bytes)")
