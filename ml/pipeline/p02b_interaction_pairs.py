# ml/pipeline/p02b_interaction_pairs.py
"""Stage 1 dataset: subsampled binary interaction pairs from the full DrugBank edge list."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from ml.config import load_config
from ml.common import set_seed
from ml.pipeline.p02_build_labels import canonical_pair, sample_negatives


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    raw, proc = cfg["paths"]["raw_dir"], cfg["paths"]["processed_dir"]
    os.makedirs(proc, exist_ok=True)

    drugs = pd.read_csv(os.path.join(raw, "drugs.csv")).drop_duplicates("drug_id")
    valid = set(drugs["drug_id"])
    edges = pd.read_csv(os.path.join(raw, "drugbank_edges.csv"))

    pos = set()
    for a, b in zip(edges["head"], edges["tail"]):
        if a in valid and b in valid and a != b:
            pos.add(canonical_pair(a, b))
    print(f"[stage1] valid positive interaction pairs: {len(pos)}")

    target = int(cfg["stage1"]["max_pairs"])
    n_pos = target // 2
    rng = np.random.default_rng(cfg["seed"])
    pos_list = list(pos)
    if len(pos_list) > n_pos:
        sel = rng.choice(len(pos_list), size=n_pos, replace=False)
        pos_sample = [pos_list[i] for i in sel]
        print(f"[stage1] subsampled positives {len(pos_list)} -> {n_pos}")
    else:
        pos_sample = pos_list

    n_neg = int(len(pos_sample) * cfg["stage1"]["neg_ratio"])
    negs = sample_negatives(sorted(valid), pos, n=n_neg, hard_fraction=0.0,
                            features=None, seed=cfg["seed"])
    print(f"[stage1] sampled negatives: {len(negs)}")

    rows = [(a, b, 1) for a, b in pos_sample] + [(a, b, 0) for a, b in negs]
    df = pd.DataFrame(rows, columns=["drugA_id", "drugB_id", "y"])
    out = os.path.join(proc, "interaction_pairs.parquet")
    df.to_parquet(out, index=False)
    print(f"[done] wrote {len(df)} interaction pairs ({df['y'].mean():.2f} positive) -> {out}")


if __name__ == "__main__":
    main()
