# ml/pipeline/p04_splits.py
"""Stage 4: drug-disjoint split into train/val/test_S0/test_S1/test_S2."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from ml.config import load_config
from ml.common import set_seed


def split_drugs(drug_ids, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    ids = list(drug_ids)
    rng.shuffle(ids)
    n_test = int(len(ids) * test_frac)
    return ids[n_test:], ids[:n_test]   # (train_drugs, test_drugs)


def assign_regime(a: str, b: str, test_drugs: set) -> str:
    in_test = (a in test_drugs) + (b in test_drugs)
    return {0: "S0", 1: "S1", 2: "S2"}[in_test]


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    proc = cfg["paths"]["processed_dir"]
    pairs = pd.read_parquet(os.path.join(proc, "pairs.parquet"))

    all_drugs = pd.unique(pairs[["drugA_id", "drugB_id"]].values.ravel())
    train_drugs, test_drugs = split_drugs(
        all_drugs, cfg["split"]["test_drug_frac"], cfg["seed"])
    test_set = set(test_drugs)

    regime = [assign_regime(a, b, test_set)
              for a, b in zip(pairs["drugA_id"], pairs["drugB_id"])]
    pairs = pairs.copy()
    pairs["regime"] = regime

    # S1/S2 pairs become the cold-start test sets directly.
    # S0 pairs (both drugs seen) are split into train / val / test_S0.
    rng = np.random.default_rng(cfg["seed"])
    split_col = []
    s0_frac, val_frac = cfg["split"]["s0_frac"], cfg["split"]["val_frac"]
    for r in pairs["regime"]:
        if r == "S1":
            split_col.append("test_S1")
        elif r == "S2":
            split_col.append("test_S2")
        else:  # S0
            u = rng.random()
            if u < s0_frac:
                split_col.append("test_S0")
            elif u < s0_frac + val_frac:
                split_col.append("val")
            else:
                split_col.append("train")
    pairs["split"] = split_col

    out = os.path.join(proc, "splits.parquet")
    pairs.drop(columns="regime").to_parquet(out, index=False)
    print("[done] split counts:\n", pairs["split"].value_counts())
    # Leakage assertion: no train/val drug appears in test_S2.
    train_val_drugs = set(pd.unique(
        pairs[pairs.split.isin(["train", "val"])][["drugA_id", "drugB_id"]].values.ravel()))
    s2 = pairs[pairs.split == "test_S2"]
    s2_drugs = set(pd.unique(s2[["drugA_id", "drugB_id"]].values.ravel()))
    assert train_val_drugs.isdisjoint(s2_drugs), "LEAKAGE: S2 drug seen in train/val"
    print("[ok] S2 is fully drug-disjoint from train/val")


if __name__ == "__main__":
    main()
