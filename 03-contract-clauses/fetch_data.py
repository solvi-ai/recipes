"""Download CUAD v1 (The Atticus Project, CC BY 4.0) and keep its held-out test file in ./data/cuad/test.json.

    python fetch_data.py

Source: https://github.com/TheAtticusProject/cuad (data.zip, 18 MB; only test.json, 7 MB, is kept).
Licence: Creative Commons Attribution 4.0 International. Hendrycks, Burns, Chen, Ball, "CUAD: An Expert-Annotated NLP
Dataset for Legal Contract Review", NeurIPS 2021 Datasets and Benchmarks."""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
OUT = Path(__file__).resolve().parent / "data" / "cuad"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "test.json"
    if not dest.exists():
        with urllib.request.urlopen(URL) as r:
            zf = zipfile.ZipFile(io.BytesIO(r.read()))
        dest.write_bytes(zf.read("test.json"))
    print(f"{dest} ({dest.stat().st_size:,} bytes)")
