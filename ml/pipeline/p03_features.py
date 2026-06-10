# ml/pipeline/p03_features.py
"""Stage 3: precompute the per-drug feature matrix for all drugs in drugs.csv."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from ml.config import load_config
from ml.features import drug_vector


def main():
    cfg = load_config()
    raw, proc = cfg["paths"]["raw_dir"], cfg["paths"]["processed_dir"]
    fp_bits = cfg["features"]["fp_bits"]
    radius = cfg["features"]["fp_radius"]

    drugs = pd.read_csv(os.path.join(raw, "drugs.csv")).drop_duplicates("drug_id")
    ids = drugs["drug_id"].tolist()
    mat = np.stack([drug_vector(s, fp_bits=fp_bits, radius=radius)
                    for s in drugs["smiles"].astype(str)])
    out = os.path.join(proc, "drug_features.npz")
    np.savez_compressed(out, ids=np.array(ids), X=mat.astype(np.float32))
    print(f"[done] features {mat.shape} for {len(ids)} drugs -> {out}")


if __name__ == "__main__":
    main()
