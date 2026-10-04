"""Download Banking77 (PolyAI, CC BY 4.0) into ./data/banking77/. Nothing is stored in this repository.

    python fetch_data.py

Source: https://github.com/PolyAI-LDN/task-specific-datasets (banking_data/train.csv, test.csv), about 1 MB.
Licence: Creative Commons Attribution 4.0 International. Casanueva et al., "Efficient Intent Detection with Dual
Sentence Encoders", 2020."""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
OUT = Path(__file__).resolve().parent / "data" / "banking77"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "test.csv"):
        dest = OUT / name
        if not dest.exists() or dest.stat().st_size == 0:
            urllib.request.urlretrieve(BASE + name, dest)
        print(f"{dest} ({dest.stat().st_size:,} bytes)")
