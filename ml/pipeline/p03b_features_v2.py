# ml/pipeline/p03b_features_v2.py
"""Stage 3b: per-drug structural matrix (per config) + bio entity sets."""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from ml.config import load_config
from ml.assemble import drug_struct_vector
from ml.bio_features import load_bio_sets, ENTITY_TYPES


def struct_tag(cfg) -> str:
    return cfg["features"]["structural"]


def main():
    cfg = load_config()
    raw, proc = cfg["paths"]["raw_dir"], cfg["paths"]["processed_dir"]
    os.makedirs(proc, exist_ok=True)
    drugs = pd.read_csv(os.path.join(raw, "drugs.csv")).drop_duplicates("drug_id")
    ids = drugs["drug_id"].tolist()

    mat = np.stack([drug_struct_vector(s, cfg) for s in drugs["smiles"].astype(str)])
    out = os.path.join(proc, f"struct_{struct_tag(cfg)}.npz")
    np.savez_compressed(out, ids=np.array(ids), X=mat.astype(np.float32))
    print(f"[done] struct {mat.shape} ({struct_tag(cfg)}) -> {out}")

    bio_path = os.path.join(proc, "bio_sets.json")
    if not os.path.exists(bio_path):
        sets = load_bio_sets(cfg["bio"]["merged_csv"])
        serializable = {d: {t: sorted(v[t]) for t in ENTITY_TYPES} for d, v in sets.items()}
        json.dump(serializable, open(bio_path, "w"))
        print(f"[done] bio sets for {len(serializable)} drugs -> {bio_path}")
    else:
        print(f"[skip] bio sets already exist -> {bio_path}")


if __name__ == "__main__":
    main()
