# ml/pipeline/p01_download.py
"""Stage 1: download DDInter severity CSVs and stage local inputs."""
from __future__ import annotations
import os
import shutil
import urllib.request
import pandas as pd
from ml.config import load_config


def ddinter_urls(cfg: dict) -> dict:
    base = cfg["ddinter"]["base_url"]
    return {c: f"{base}{c}.csv" for c in cfg["ddinter"]["categories"]}


def main():
    cfg = load_config()
    raw = cfg["paths"]["raw_dir"]
    os.makedirs(os.path.join(raw, "ddinter"), exist_ok=True)

    for cat, url in ddinter_urls(cfg).items():
        dest = os.path.join(raw, "ddinter", f"code_{cat}.csv")
        if os.path.exists(dest):
            print(f"[skip] {dest}")
            continue
        print(f"[get ] {url}")
        urllib.request.urlretrieve(url, dest)

    # Stage drugs.csv and DrugBank edges into raw/ so the pipeline is self-contained.
    shutil.copy(cfg["paths"]["drugs_csv"], os.path.join(raw, "drugs.csv"))
    shutil.copy(cfg["paths"]["drugbank_edges_csv"], os.path.join(raw, "drugbank_edges.csv"))

    # Report combined DDInter row counts as a sanity signal.
    frames = []
    for cat in cfg["ddinter"]["categories"]:
        p = os.path.join(raw, "ddinter", f"code_{cat}.csv")
        frames.append(pd.read_csv(p))
    total = pd.concat(frames, ignore_index=True)
    print(f"[done] DDInter rows downloaded: {len(total)} across {len(frames)} categories")


if __name__ == "__main__":
    main()
