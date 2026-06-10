# ml/pipeline/p02_build_labels.py
"""Stage 2: join DDInter severity onto local drugs, build the None pool, write pairs."""
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
from ml.common import normalize_name, LEVEL_MAP, set_seed
from ml.config import load_config


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def build_name_map(drugs: pd.DataFrame) -> dict:
    """normalized name -> drug_id (first occurrence wins)."""
    m = {}
    for did, name in zip(drugs["drug_id"], drugs["name"]):
        key = normalize_name(name)
        if key and key not in m:
            m[key] = did
    return m


def join_positives(dd: pd.DataFrame, name_map: dict) -> pd.DataFrame:
    """Map DDInter rows to canonical (drugA_id, drugB_id, label); drop Unknown/unmapped."""
    rows = []
    for a_name, b_name, level in zip(dd["Drug_A"], dd["Drug_B"], dd["Level"]):
        if level not in LEVEL_MAP:
            continue
        a = name_map.get(normalize_name(a_name))
        b = name_map.get(normalize_name(b_name))
        if a is None or b is None or a == b:
            continue
        ca, cb = canonical_pair(a, b)
        rows.append((ca, cb, LEVEL_MAP[level]))
    pos = pd.DataFrame(rows, columns=["drugA_id", "drugB_id", "label"])
    # If a pair appears at multiple severities, keep the most severe.
    order = {"Minor": 1, "Moderate": 2, "Major": 3}
    pos["rank"] = pos["label"].map(order)
    pos = (pos.sort_values("rank")
              .drop_duplicates(["drugA_id", "drugB_id"], keep="last")
              .drop(columns="rank")
              .reset_index(drop=True))
    return pos


def sample_negatives(drug_ids, masked: set, n: int, hard_fraction: float,
                     features, seed: int):
    """Sample n canonical None pairs not in `masked`.

    `features` (optional dict drug_id->np.ndarray fingerprint) enables hard negatives:
    structurally similar pairs (high cosine on fingerprints) that are NOT masked.
    """
    rng = np.random.default_rng(seed)
    ids = list(drug_ids)
    out = set()

    def rand_pair():
        i, j = rng.integers(0, len(ids), size=2)
        if i == j:
            return None
        a, b = canonical_pair(ids[i], ids[j])
        return (a, b) if (a, b) not in masked else None

    n_hard = int(n * hard_fraction) if features else 0
    n_rand = n - n_hard

    # Random negatives.
    tries = 0
    while len([p for p in out]) < n_rand and tries < n_rand * 50:
        p = rand_pair()
        if p and p not in out:
            out.add(p)
        tries += 1

    # Hard negatives: for random anchor drugs, pick their most-similar non-masked partner.
    if n_hard:
        mat = np.stack([features[d] for d in ids])
        norms = np.linalg.norm(mat, axis=1) + 1e-8
        tries = 0
        while len(out) < n and tries < n_hard * 50:
            ai = int(rng.integers(0, len(ids)))
            sims = (mat @ mat[ai]) / (norms * norms[ai])
            sims[ai] = -1
            order = np.argsort(-sims)
            for bi in order[:20]:
                a, b = canonical_pair(ids[ai], ids[int(bi)])
                if (a, b) not in masked and (a, b) not in out:
                    out.add((a, b))
                    break
            tries += 1

    return list(out)[:n]


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    raw, proc = cfg["paths"]["raw_dir"], cfg["paths"]["processed_dir"]
    os.makedirs(proc, exist_ok=True)

    drugs = pd.read_csv(os.path.join(raw, "drugs.csv")).drop_duplicates("drug_id")
    name_map = build_name_map(drugs)

    dd = pd.concat(
        [pd.read_csv(p) for p in glob.glob(os.path.join(raw, "ddinter", "*.csv"))],
        ignore_index=True,
    )
    pos = join_positives(dd, name_map)
    print(f"[labels] positives after join: {len(pos)} "
          f"(distribution: {pos['label'].value_counts().to_dict()})")

    # Interaction mask = DDInter positive pairs UNION DrugBank edges (canonicalized).
    masked = set(zip(pos["drugA_id"], pos["drugB_id"]))
    edges = pd.read_csv(os.path.join(raw, "drugbank_edges.csv"))
    valid = set(drugs["drug_id"])
    for a, b in zip(edges["head"], edges["tail"]):
        if a in valid and b in valid and a != b:
            masked.add(canonical_pair(a, b))
    print(f"[labels] masked interacting pairs (excluded from None pool): {len(masked)}")

    n_neg = int(cfg["negatives"]["ratio"] * len(pos))
    negs = sample_negatives(
        sorted(valid), masked, n=n_neg,
        hard_fraction=cfg["negatives"]["hard_fraction"],
        features=None, seed=cfg["seed"],
    )
    neg_df = pd.DataFrame(negs, columns=["drugA_id", "drugB_id"])
    neg_df["label"] = "None"
    print(f"[labels] sampled None pairs: {len(neg_df)}")

    pairs = pd.concat([pos, neg_df], ignore_index=True)
    out = os.path.join(proc, "pairs.parquet")
    pairs.to_parquet(out, index=False)
    print(f"[done] wrote {len(pairs)} pairs -> {out}")


if __name__ == "__main__":
    main()
