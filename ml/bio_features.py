# ml/bio_features.py
"""Biological mechanistic-overlap features from DrugBank target/enzyme/etc. sets."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

ENTITY_TYPES = ["targets", "enzymes", "transporters", "carriers"]

# Fixed feature order (used everywhere: train, eval, serving).
BIO_FEATURE_NAMES = []
for _t in ENTITY_TYPES:
    BIO_FEATURE_NAMES += [f"{_t}_shared", f"{_t}_jaccard", f"{_t}_count_sum", f"{_t}_count_absdiff"]
N_BIO = len(BIO_FEATURE_NAMES)


def parse_entity_cell(cell) -> set:
    """Space-separated DrugBank entity IDs -> set; NaN/empty -> empty set."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return set()
    s = str(cell).strip()
    # Defensive: some CSV-read settings surface missing values as the literal string
    # "nan" rather than a float NaN. No real DrugBank entity ID is "nan".
    if not s or s.lower() == "nan":
        return set()
    return set(s.split())


def empty_bio_sets() -> dict:
    return {t: set() for t in ENTITY_TYPES}


def load_bio_sets(merged_csv: str) -> dict:
    """drug_id -> {targets, enzymes, transporters, carriers} sets."""
    df = pd.read_csv(merged_csv, usecols=["drugbank_id"] + ENTITY_TYPES, dtype=str)
    out = {}
    for _, row in df.iterrows():
        out[row["drugbank_id"]] = {t: parse_entity_cell(row[t]) for t in ENTITY_TYPES}
    return out


def bio_pair_features(a: dict, b: dict) -> np.ndarray:
    """Order-invariant overlap features for a drug pair. a,b are bio-set dicts."""
    vals = []
    for t in ENTITY_TYPES:
        sa, sb = a.get(t, set()), b.get(t, set())
        inter = len(sa & sb)
        union = len(sa | sb)
        jacc = inter / union if union else 0.0
        vals += [float(inter), float(jacc), float(len(sa) + len(sb)), float(abs(len(sa) - len(sb)))]
    return np.array(vals, dtype=np.float32)
