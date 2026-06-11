# DDI Severity Iteration 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve cold-start DDI severity by (1) a two-stage model (binary interaction on ~1.8M DrugBank edges → 3-class severity on DDInter), (2) biological mechanistic-overlap features, (3) an upgraded structural feature set with optional ChemBERTa, all measured by an ablation table.

**Architecture:** Extends the existing `ml/` package. New per-drug structural vectors (count-Morgan + ~38 RDKit descriptors) and per-pair biological overlap features (shared targets/enzymes/transporters/carriers). A binary InteractionMLP trains on a subsampled interaction-pair dataset; a 3-class SeverityMLP trains on DDInter positives. Evaluation reports each stage plus end-to-end 4-class metrics across drug-disjoint S0/S1/S2, swept over feature configs. Serving loads both models and a SMILES→bio lookup.

**Tech Stack:** Python 3.13, PyTorch, RDKit, scikit-learn, pandas, pyarrow, matplotlib, optionally HuggingFace transformers (ChemBERTa). Interpreter: `/Users/zeyadzaher/grad-project/.venv/bin/python`.

**Conventions (every task):**
- Repo root: `/Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction`; run commands from there.
- Interpreter literal: `/Users/zeyadzaher/grad-project/.venv/bin/python`
- Branch: `ddi-severity-v2` (already checked out; do not switch).
- Data/artifacts under `ml/data/` and `ml/artifacts/` are gitignored — never commit them.
- Iteration-1 modules already exist: `ml/common.py` (`CLASSES`, `CLASS_TO_IDX`, `normalize_name`, `set_seed`), `ml/config.py` (`load_config`), `ml/features.py` (`drug_vector`, `pair_features`), `ml/pipeline/p01_download.py`, `p02_build_labels.py` (`canonical_pair`), `p03_features.py`, `p04_splits.py`, etc. `ml/data/raw/` already has `drugs.csv`, `drugbank_edges.csv`, `ddinter/`. `ml/data/processed/` has `pairs.parquet`, `splits.parquet`.

---

## File Structure

```
ml/
  config.yaml                 # MODIFY: + features.structural/use_bio, + stage1 block, + bio_source
  common.py                   # (unchanged)
  features.py                 # MODIFY: add light_drug_vector (count-Morgan + ~38 descriptors), keep pair_features
  bio_features.py             # NEW: parse entity sets, BIO_FEATURE_NAMES, bio_pair_features
  embeddings.py               # NEW: ChemBERTa embeddings (lazy import, optional)
  assemble.py                 # NEW: unified pair-matrix assembly used by train/eval/serving
  models.py                   # MODIFY: add MLPHead(in_dim, n_out)
  pipeline/
    p02b_interaction_pairs.py # NEW: build subsampled binary interaction dataset
    p03b_features_v2.py       # NEW: per-drug structural matrix (config) + bio sets json
    p05b_train_stages.py      # NEW: train InteractionMLP (binary) + SeverityMLP (3-class)
    p06b_evaluate_v2.py       # NEW: two-stage + end-to-end metrics for one feature config
    p07b_export_v2.py         # NEW: export both models + metadata + SMILES->bio lookup
  run_ablation.py             # NEW: sweep configs, emit ablation_table.md
  tests/
    test_bio_features.py      # NEW
    test_features_v2.py       # NEW
    test_assemble.py          # NEW
    test_models_v2.py         # NEW
    test_evaluate_v2.py       # NEW
inference-py/app/model.py     # MODIFY: TwoStageDDIModel (two models + bio lookup)
inference-py/app/main.py      # (unchanged interface; only DDIModel construction changes)
```

Backend/mobile need **no changes** — the JSON response shape (`interactionProbability`,
`severity`, `probabilities`) is identical to iteration 1.

---

## Task 1: Biological overlap features

**Files:**
- Create: `ml/bio_features.py`
- Test: `ml/tests/test_bio_features.py`

- [ ] **Step 1: Write failing tests `ml/tests/test_bio_features.py`**

```python
# ml/tests/test_bio_features.py
import numpy as np
from ml.bio_features import (parse_entity_cell, bio_pair_features,
                             BIO_FEATURE_NAMES, empty_bio_sets)


def test_parse_entity_cell_splits_ids_and_handles_nan():
    assert parse_entity_cell("BE1 BE2 BE3") == {"BE1", "BE2", "BE3"}
    assert parse_entity_cell(float("nan")) == set()
    assert parse_entity_cell("") == set()


def test_bio_pair_features_symmetric_and_named():
    a = {"targets": {"T1", "T2"}, "enzymes": {"E1"}, "transporters": set(), "carriers": set()}
    b = {"targets": {"T2", "T3"}, "enzymes": {"E1"}, "transporters": set(), "carriers": set()}
    fa = bio_pair_features(a, b)
    fb = bio_pair_features(b, a)
    assert fa.shape == (len(BIO_FEATURE_NAMES),)
    assert np.allclose(fa, fb)                      # order-invariant


def test_bio_pair_features_overlap_values():
    a = {"targets": {"T1", "T2"}, "enzymes": set(), "transporters": set(), "carriers": set()}
    b = {"targets": {"T2"}, "enzymes": set(), "transporters": set(), "carriers": set()}
    feats = dict(zip(BIO_FEATURE_NAMES, bio_pair_features(a, b)))
    assert feats["targets_shared"] == 1.0          # T2
    assert abs(feats["targets_jaccard"] - 0.5) < 1e-6   # |{T2}| / |{T1,T2,T3?}| = 1/2


def test_empty_bio_sets_has_all_types():
    s = empty_bio_sets()
    assert set(s) == {"targets", "enzymes", "transporters", "carriers"}
    assert all(v == set() for v in s.values())
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_bio_features.py -v`
Expected: FAIL (`ModuleNotFoundError: ml.bio_features`).

- [ ] **Step 3: Write `ml/bio_features.py`**

```python
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
    return set(s.split()) if s else set()


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
```

- [ ] **Step 4: Run tests (pass)**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_bio_features.py -v`
Expected: 4 passed.

- [ ] **Step 5: Smoke-check real data loads**

Run: `/Users/zeyadzaher/grad-project/.venv/bin/python -c "from ml.bio_features import load_bio_sets; m=load_bio_sets('/Users/zeyadzaher/grad-project/merged_output.csv'); print('drugs:', len(m)); print('with enzymes:', sum(1 for v in m.values() if v['enzymes']))"`
Expected: `drugs: 8287`, `with enzymes: 1000`.

- [ ] **Step 6: Commit**

```bash
git add ml/bio_features.py ml/tests/test_bio_features.py
git commit -m "feat(ml): biological mechanistic-overlap features"
```

---

## Task 2: Upgraded structural features (count-Morgan + ~38 descriptors)

**Files:**
- Modify: `ml/features.py`
- Test: `ml/tests/test_features_v2.py`

- [ ] **Step 1: Write failing tests `ml/tests/test_features_v2.py`**

```python
# ml/tests/test_features_v2.py
import numpy as np
from ml.features import light_drug_vector, LIGHT_DESCRIPTORS, pair_features


def test_light_vector_shape_and_invalid_smiles():
    v = light_drug_vector("CC(=O)OC1=CC=CC=C1C(=O)O", fp_bits=512)
    assert v.shape == (512 + len(LIGHT_DESCRIPTORS),)
    assert np.all(np.isfinite(v))                 # no NaN/inf
    z = light_drug_vector("nonsense", fp_bits=512)
    assert z.shape == (512 + len(LIGHT_DESCRIPTORS),)
    assert np.all(z == 0)                          # invalid -> all zeros


def test_light_vector_is_count_based():
    # A molecule with repeated substructure should yield some fp entry > 1 (counts, not bits).
    v = light_drug_vector("CCCCCCCCCC", fp_bits=512)
    assert v[:512].max() >= 1.0


def test_pair_features_symmetric_on_light_vectors():
    a = light_drug_vector("CCO", fp_bits=256)
    b = light_drug_vector("CCN", fp_bits=256)
    assert np.allclose(pair_features(a, b), pair_features(b, a))
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_features_v2.py -v`
Expected: FAIL (`ImportError: cannot import name 'light_drug_vector'`).

- [ ] **Step 3: Append to `ml/features.py`** (keep existing `drug_vector`/`pair_features`)

Add these imports at the top if not present, then append the new code:

```python
# --- appended in Task 2: upgraded structural features ---
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED
from rdkit import Chem as _Chem

# Curated descriptor set (name, fn). All return a finite float for valid mols;
# NaN/inf are zeroed at the call site.
LIGHT_DESCRIPTORS = [
    ("MolWt", Descriptors.MolWt), ("ExactMolWt", Descriptors.ExactMolWt),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount), ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumHDonors", Descriptors.NumHDonors), ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumAromaticRings", rdMolDescriptors.CalcNumAromaticRings),
    ("NumAliphaticRings", rdMolDescriptors.CalcNumAliphaticRings),
    ("NumSaturatedRings", rdMolDescriptors.CalcNumSaturatedRings),
    ("RingCount", Descriptors.RingCount), ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
    ("TPSA", Descriptors.TPSA), ("MolLogP", Crippen.MolLogP), ("MolMR", Crippen.MolMR),
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
    ("NumHeteroatoms", rdMolDescriptors.CalcNumHeteroatoms),
    ("NOCount", Descriptors.NOCount), ("NHOHCount", Descriptors.NHOHCount),
    ("NumAromaticHeterocycles", rdMolDescriptors.CalcNumAromaticHeterocycles),
    ("NumAliphaticHeterocycles", rdMolDescriptors.CalcNumAliphaticHeterocycles),
    ("NumAromaticCarbocycles", rdMolDescriptors.CalcNumAromaticCarbocycles),
    ("NumAliphaticCarbocycles", rdMolDescriptors.CalcNumAliphaticCarbocycles),
    ("LabuteASA", Descriptors.LabuteASA), ("BalabanJ", Descriptors.BalabanJ),
    ("BertzCT", Descriptors.BertzCT), ("Chi0", Descriptors.Chi0), ("Chi1", Descriptors.Chi1),
    ("Chi0n", Descriptors.Chi0n), ("Chi1n", Descriptors.Chi1n),
    ("HallKierAlpha", Descriptors.HallKierAlpha), ("Kappa1", Descriptors.Kappa1),
    ("Kappa2", Descriptors.Kappa2), ("Kappa3", Descriptors.Kappa3),
    ("MaxPartialCharge", Descriptors.MaxPartialCharge),
    ("MinPartialCharge", Descriptors.MinPartialCharge),
    ("NumRadicalElectrons", Descriptors.NumRadicalElectrons),
    ("qed", QED.qed), ("FormalCharge", lambda m: _Chem.GetFormalCharge(m)),
]


def light_drug_vector(smiles: str, fp_bits: int = 1024, radius: int = 2) -> np.ndarray:
    """Count-based Morgan fingerprint + curated descriptors as float32; zeros if invalid."""
    mol = Chem.MolFromSmiles(str(smiles))
    fp = np.zeros((fp_bits,), dtype=np.float32)
    desc = np.zeros((len(LIGHT_DESCRIPTORS),), dtype=np.float32)
    if mol is not None:
        cfp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=fp_bits)  # counts
        for idx, cnt in cfp.GetNonzeroElements().items():
            fp[idx] = float(cnt)
        for i, (_, fn) in enumerate(LIGHT_DESCRIPTORS):
            try:
                val = float(fn(mol))
                desc[i] = val if np.isfinite(val) else 0.0
            except Exception:
                desc[i] = 0.0
    return np.concatenate([fp, desc]).astype(np.float32)
```

- [ ] **Step 4: Run tests (pass)**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_features_v2.py -v`
Expected: 3 passed. (`AllChem` and `np`/`Chem` are already imported at the top of features.py from iteration 1; if `AllChem` is missing, add `from rdkit.Chem import AllChem`.)

- [ ] **Step 5: Commit**

```bash
git add ml/features.py ml/tests/test_features_v2.py
git commit -m "feat(ml): count-Morgan + expanded descriptor structural features"
```

---

## Task 3: ChemBERTa embeddings (optional, lazy)

**Files:**
- Create: `ml/embeddings.py`
- Test: `ml/tests/test_features_v2.py` (append one guarded test)

- [ ] **Step 1: Write `ml/embeddings.py`**

```python
# ml/embeddings.py
"""Optional ChemBERTa molecular embeddings. Heavy deps imported lazily."""
from __future__ import annotations
import numpy as np

_MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
_cache = {}


def chemberta_available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _load():
    if "model" not in _cache:
        import torch
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
        mdl = AutoModel.from_pretrained(_MODEL_NAME).eval()
        _cache["tok"], _cache["model"], _cache["torch"] = tok, mdl, torch
    return _cache["tok"], _cache["model"], _cache["torch"]


def embed_smiles(smiles: str) -> np.ndarray:
    """Mean-pooled last-hidden-state embedding (float32). Zeros if invalid/unavailable."""
    if not chemberta_available():
        raise RuntimeError("transformers/torch not installed; cannot use chemberta config")
    tok, mdl, torch = _load()
    with torch.no_grad():
        enc = tok(str(smiles), return_tensors="pt", truncation=True, max_length=256)
        out = mdl(**enc).last_hidden_state  # (1, L, H)
        mask = enc["attention_mask"].unsqueeze(-1)  # (1, L, 1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled.squeeze(0).cpu().numpy().astype(np.float32)


def embedding_dim() -> int:
    _, mdl, _ = _load()
    return int(mdl.config.hidden_size)
```

- [ ] **Step 2: Append a guarded test to `ml/tests/test_features_v2.py`**

```python
import pytest
from ml.embeddings import chemberta_available, embed_smiles


@pytest.mark.skipif(not chemberta_available(), reason="transformers not installed")
def test_chemberta_embedding_shape():
    import numpy as np
    v = embed_smiles("CCO")
    assert v.ndim == 1 and v.shape[0] > 0 and np.all(np.isfinite(v))
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_features_v2.py -v`
Expected: 3 passed + 1 (passed if transformers present, else skipped). Either way, no failures.
Note: if you want the `chemberta` config to actually run later, install once: `/Users/zeyadzaher/grad-project/.venv/bin/python -m pip install transformers`. This is NOT required for the default `light` config.

- [ ] **Step 4: Commit**

```bash
git add ml/embeddings.py ml/tests/test_features_v2.py
git commit -m "feat(ml): optional ChemBERTa embeddings (lazy)"
```

---

## Task 4: Config + unified pair-matrix assembly

**Files:**
- Modify: `ml/config.yaml`
- Create: `ml/assemble.py`
- Test: `ml/tests/test_assemble.py`

- [ ] **Step 1: Add to `ml/config.yaml`** (append these keys; keep existing ones)

```yaml
features:
  fp_radius: 2
  fp_bits: 2048           # used by iteration-1 drug_vector (kept)
  light_fp_bits: 1024     # used by light_drug_vector
  structural: light       # light | chemberta
  use_bio: true

bio:
  merged_csv: /Users/zeyadzaher/grad-project/merged_output.csv

stage1:
  max_pairs: 120000       # subsample of the ~1.8M interaction edges (pos+neg combined target)
  neg_ratio: 1.0
```

- [ ] **Step 2: Write failing tests `ml/tests/test_assemble.py`**

```python
# ml/tests/test_assemble.py
import numpy as np
from ml.assemble import struct_dim, assemble_pair
from ml.features import LIGHT_DESCRIPTORS
from ml.bio_features import empty_bio_sets, N_BIO

NDESC = len(LIGHT_DESCRIPTORS)


def test_struct_dim_light():
    assert struct_dim({"features": {"structural": "light", "light_fp_bits": 256}}) == 256 + NDESC


def test_assemble_pair_symmetric_and_dim():
    cfg = {"features": {"structural": "light", "light_fp_bits": 128, "use_bio": True}}
    sa = np.ones(128 + NDESC, dtype=np.float32)
    sb = np.arange(128 + NDESC, dtype=np.float32)
    ba = empty_bio_sets(); bb = empty_bio_sets()
    ba["targets"] = {"T1"}; bb["targets"] = {"T1"}
    v1 = assemble_pair(sa, sb, ba, bb, use_bio=True)
    v2 = assemble_pair(sb, sa, bb, ba, use_bio=True)
    assert np.allclose(v1, v2)                          # symmetric
    assert v1.shape[0] == 3 * (128 + NDESC) + N_BIO     # struct pair block + bio block
```

- [ ] **Step 3: Run to confirm failure**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_assemble.py -v`
Expected: FAIL (`ModuleNotFoundError: ml.assemble`).

- [ ] **Step 4: Write `ml/assemble.py`**

```python
# ml/assemble.py
"""Unified pair-feature assembly shared by training, evaluation, and serving."""
from __future__ import annotations
import numpy as np
from ml.features import light_drug_vector, pair_features, LIGHT_DESCRIPTORS
from ml.bio_features import bio_pair_features, N_BIO


def struct_dim(cfg: dict) -> int:
    f = cfg["features"]
    if f["structural"] == "light":
        return int(f["light_fp_bits"]) + len(LIGHT_DESCRIPTORS)
    if f["structural"] == "chemberta":
        from ml.embeddings import embedding_dim
        return embedding_dim()
    raise ValueError(f"unknown structural config: {f['structural']}")


def drug_struct_vector(smiles: str, cfg: dict) -> np.ndarray:
    f = cfg["features"]
    if f["structural"] == "light":
        return light_drug_vector(smiles, fp_bits=int(f["light_fp_bits"]), radius=int(f["fp_radius"]))
    if f["structural"] == "chemberta":
        from ml.embeddings import embed_smiles
        try:
            return embed_smiles(smiles)
        except Exception:
            from ml.embeddings import embedding_dim
            return np.zeros((embedding_dim(),), dtype=np.float32)
    raise ValueError(f"unknown structural config: {f['structural']}")


def assemble_pair(struct_a, struct_b, bio_a, bio_b, use_bio: bool) -> np.ndarray:
    """struct_a/b: per-drug structural vectors; bio_a/b: bio-set dicts."""
    blocks = [pair_features(struct_a, struct_b)]
    if use_bio:
        blocks.append(bio_pair_features(bio_a, bio_b))
    return np.concatenate(blocks).astype(np.float32)


def pair_dim(cfg: dict) -> int:
    d = 3 * struct_dim(cfg)
    if cfg["features"].get("use_bio"):
        d += N_BIO
    return d
```

- [ ] **Step 5: Run tests (pass)**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_assemble.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add ml/config.yaml ml/assemble.py ml/tests/test_assemble.py
git commit -m "feat(ml): config flags + unified pair-feature assembly"
```

---

## Task 5: Stage-1 interaction-pair dataset

**Files:**
- Create: `ml/pipeline/p02b_interaction_pairs.py`
- Test: covered by stage run + reuse of `sample_negatives` (unit-tested in iteration 1)

Produces `ml/data/processed/interaction_pairs.parquet` with columns `drugA_id, drugB_id, y`
(y=1 interact, 0 none), subsampled to `stage1.max_pairs`.

- [ ] **Step 1: Write `ml/pipeline/p02b_interaction_pairs.py`**

```python
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
```

- [ ] **Step 2: Run the stage**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m ml.pipeline.p02b_interaction_pairs`
Expected: prints positive count (~600k+), subsample note, negatives, and `[done] wrote ~120000 interaction pairs (~0.50 positive)`.

- [ ] **Step 3: Commit**

```bash
git add ml/pipeline/p02b_interaction_pairs.py
git commit -m "feat(ml): subsampled binary interaction-pair dataset for stage 1"
```

---

## Task 6: Per-drug feature matrices + bio sets (stage 3b)

**Files:**
- Create: `ml/pipeline/p03b_features_v2.py`

Produces, for the active config: `ml/data/processed/struct_<config>.npz` (ids, X) and
`ml/data/processed/bio_sets.json` (drug_id -> {type: [ids]}).

- [ ] **Step 1: Write `ml/pipeline/p03b_features_v2.py`**

```python
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

    # Bio sets (config-independent; write once).
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
```

- [ ] **Step 2: Run the stage (light config)**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m ml.pipeline.p03b_features_v2`
Expected: `[done] struct (8287, 1062) (light) -> ...struct_light.npz` (1062 = 1024 fp + 38 desc; second dim = light_fp_bits + len(LIGHT_DESCRIPTORS)) and a bio_sets.json line.

- [ ] **Step 3: Commit**

```bash
git add ml/pipeline/p03b_features_v2.py
git commit -m "feat(ml): per-drug structural matrix + bio sets stage"
```

---

## Task 7: Models — shared MLP head

**Files:**
- Modify: `ml/models.py`
- Test: `ml/tests/test_models_v2.py`

- [ ] **Step 1: Write failing tests `ml/tests/test_models_v2.py`**

```python
# ml/tests/test_models_v2.py
import torch
from ml.models import MLPHead


def test_mlphead_binary_output():
    m = MLPHead(in_dim=20, n_out=1, hidden=16, dropout=0.1)
    out = m(torch.randn(4, 20))
    assert out.shape == (4, 1)


def test_mlphead_three_class_output():
    m = MLPHead(in_dim=20, n_out=3, hidden=16, dropout=0.1)
    out = m(torch.randn(5, 20))
    assert out.shape == (5, 3)
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_models_v2.py -v`
Expected: FAIL (`ImportError: cannot import name 'MLPHead'`).

- [ ] **Step 3: Append `MLPHead` to `ml/models.py`**

```python
# --- appended in Task 7 ---
class MLPHead(nn.Module):
    """Generic 2-hidden-layer MLP. n_out=1 for binary (BCE), n_out=3 for severity."""
    def __init__(self, in_dim: int, n_out: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)
```

(`nn` is already imported at the top of `ml/models.py` from iteration 1.)

- [ ] **Step 4: Run tests (pass)**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_models_v2.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add ml/models.py ml/tests/test_models_v2.py
git commit -m "feat(ml): shared MLPHead for binary + severity stages"
```

---

## Task 8: Train both stages

**Files:**
- Create: `ml/pipeline/p05b_train_stages.py`

Trains stage 1 (binary, on `interaction_pairs.parquet` restricted to train-split drugs) and
stage 2 (3-class severity, on DDInter positives in `splits.parquet`, train split). Both use the
active config's features. Saves `model_stage1_<cfg>.pt`, `model_stage2_<cfg>.pt`, curves.

**Key consistency rule:** stage 1's train/val/test partition MUST use the SAME drug-disjoint
groups as `splits.parquet` so S0/S1/S2 evaluation is coherent. We derive each interaction
pair's regime from the drug membership in `splits.parquet`.

- [ ] **Step 1: Write `ml/pipeline/p05b_train_stages.py`**

```python
# ml/pipeline/p05b_train_stages.py
"""Stage 5b: train binary interaction (stage 1) and 3-class severity (stage 2) models."""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from ml.config import load_config
from ml.common import set_seed
from ml.assemble import assemble_pair, drug_struct_vector, struct_dim, pair_dim
from ml.bio_features import empty_bio_sets, ENTITY_TYPES
from ml.models import MLPHead

SEV3 = ["Minor", "Moderate", "Major"]
SEV3_IDX = {c: i for i, c in enumerate(SEV3)}


def _load_struct(cfg):
    proc = cfg["paths"]["processed_dir"]
    tag = cfg["features"]["structural"]
    feat = np.load(os.path.join(proc, f"struct_{tag}.npz"), allow_pickle=True)
    ids = list(feat["ids"]); idx = {d: i for i, d in enumerate(ids)}
    return feat["X"].astype(np.float32), idx


def _load_bio(cfg):
    proc = cfg["paths"]["processed_dir"]
    raw = json.load(open(os.path.join(proc, "bio_sets.json")))
    return {d: {t: set(v.get(t, [])) for t in ENTITY_TYPES} for d, v in raw.items()}


def _drug_regime(cfg):
    """drug_id -> 'train'|'test' membership, from splits.parquet (drug-disjoint)."""
    proc = cfg["paths"]["processed_dir"]
    sp = pd.read_parquet(os.path.join(proc, "splits.parquet"))
    test_drugs = set(pd.unique(
        sp[sp.split.isin(["test_S1", "test_S2"])][["drugA_id", "drugB_id"]].values.ravel()))
    train_drugs = set(pd.unique(
        sp[sp.split.isin(["train", "val", "test_S0"])][["drugA_id", "drugB_id"]].values.ravel()))
    return train_drugs, test_drugs


def _matrix(pairs, X, idx, bio, cfg, use_bio):
    rows = []
    for a, b in zip(pairs["drugA_id"], pairs["drugB_id"]):
        rows.append(assemble_pair(X[idx[a]], X[idx[b]],
                                  bio.get(a, empty_bio_sets()), bio.get(b, empty_bio_sets()),
                                  use_bio=use_bio))
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def _train_loop(model, Xtr, ytr, Xva, yva, cfg, loss_fn, score_fn):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    bs = cfg["train"]["batch_size"]
    best, best_state, curve, bad = -1, None, [], 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]; opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            sc = score_fn(model, Xva, yva)
            tl = loss_fn(model(Xtr), ytr).item(); vl = loss_fn(model(Xva), yva).item()
        curve.append({"epoch": epoch, "train_loss": tl, "val_loss": vl, "val_score": sc})
        if sc > best:
            best, best_state, bad = sc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg["train"]["patience"]:
                break
    model.load_state_dict(best_state)
    return best, curve


def train_stage1(cfg, X, idx, bio, artifacts):
    set_seed(cfg["seed"])
    proc = cfg["paths"]["processed_dir"]
    pairs = pd.read_parquet(os.path.join(proc, "interaction_pairs.parquet"))
    train_drugs, test_drugs = _drug_regime(cfg)
    # Stage-1 train = pairs with BOTH drugs in train group; hold 10% as val.
    both_train = pairs[pairs.drugA_id.isin(train_drugs) & pairs.drugB_id.isin(train_drugs)].reset_index(drop=True)
    rng = np.random.default_rng(cfg["seed"]); u = rng.random(len(both_train))
    tr, va = both_train[u >= 0.1], both_train[u < 0.1]
    use_bio = cfg["features"]["use_bio"]
    Xtr = _matrix(tr, X, idx, bio, cfg, use_bio); ytr = torch.tensor(tr["y"].values, dtype=torch.float32).unsqueeze(1)
    Xva = _matrix(va, X, idx, bio, cfg, use_bio); yva = torch.tensor(va["y"].values, dtype=torch.float32).unsqueeze(1)
    model = MLPHead(in_dim=pair_dim(cfg), n_out=1, hidden=cfg["train"]["hidden"], dropout=cfg["train"]["dropout"])
    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    loss_fn = lambda logit, y: F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_w)
    def score(m, Xv, yv):
        p = torch.sigmoid(m(Xv)).numpy().ravel()
        return roc_auc_score(yv.numpy().ravel(), p)
    best, curve = _train_loop(model, Xtr, ytr, Xva, yva, cfg, loss_fn, score)
    tag = cfg["features"]["structural"] + ("_bio" if use_bio else "")
    torch.save({"model_state_dict": model.state_dict(), "in_dim": pair_dim(cfg), "hidden": cfg["train"]["hidden"]},
               os.path.join(artifacts, f"model_stage1_{tag}.pt"))
    json.dump(curve, open(os.path.join(artifacts, f"curve_stage1_{tag}.json"), "w"))
    print(f"[stage1:{tag}] best val AUC={best:.4f} over {len(curve)} epochs")


def train_stage2(cfg, X, idx, bio, artifacts):
    set_seed(cfg["seed"])
    proc = cfg["paths"]["processed_dir"]
    sp = pd.read_parquet(os.path.join(proc, "splits.parquet"))
    pos = sp[(sp.label != "None") & (sp.split.isin(["train", "val"]))].copy()
    pos["sy"] = pos["label"].map(SEV3_IDX)
    tr, va = pos[pos.split == "train"], pos[pos.split == "val"]
    use_bio = cfg["features"]["use_bio"]
    Xtr = _matrix(tr, X, idx, bio, cfg, use_bio); ytr = torch.tensor(tr["sy"].values, dtype=torch.long)
    Xva = _matrix(va, X, idx, bio, cfg, use_bio); yva = torch.tensor(va["sy"].values, dtype=torch.long)
    model = MLPHead(in_dim=pair_dim(cfg), n_out=3, hidden=cfg["train"]["hidden"], dropout=cfg["train"]["dropout"])
    counts = np.bincount(ytr.numpy(), minlength=3).astype(np.float32)
    w = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = lambda logit, y: F.cross_entropy(logit, y, weight=w)
    def score(m, Xv, yv):
        return f1_score(yv.numpy(), m(Xv).argmax(1).numpy(), average="macro")
    best, curve = _train_loop(model, Xtr, ytr, Xva, yva, cfg, loss_fn, score)
    tag = cfg["features"]["structural"] + ("_bio" if use_bio else "")
    torch.save({"model_state_dict": model.state_dict(), "in_dim": pair_dim(cfg), "hidden": cfg["train"]["hidden"]},
               os.path.join(artifacts, f"model_stage2_{tag}.pt"))
    json.dump(curve, open(os.path.join(artifacts, f"curve_stage2_{tag}.json"), "w"))
    print(f"[stage2:{tag}] best val macro-F1={best:.4f} over {len(curve)} epochs")


def main():
    cfg = load_config()
    artifacts = cfg["paths"]["artifacts_dir"]; os.makedirs(artifacts, exist_ok=True)
    X, idx = _load_struct(cfg); bio = _load_bio(cfg)
    train_stage1(cfg, X, idx, bio, artifacts)
    train_stage2(cfg, X, idx, bio, artifacts)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test (2 epochs)**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -c "import yaml; c=yaml.safe_load(open('ml/config.yaml')); c['train']['epochs']=2; c['train']['patience']=2; yaml.safe_dump(c,open('/tmp/v2_smoke.yaml','w')); import ml.config as cf; cf._DEFAULT='/tmp/v2_smoke.yaml'; from ml.pipeline import p05b_train_stages as t; t.main()"`
Expected: `[stage1:light_bio] best val AUC=...` and `[stage2:light_bio] best val macro-F1=...`, artifacts written.

- [ ] **Step 3: Full run**

Run: `/Users/zeyadzaher/grad-project/.venv/bin/python -m ml.pipeline.p05b_train_stages`
Expected: both stages train to completion; report the two scores. (Stage 1 on ~96k train pairs may take a few minutes on CPU — let it finish.)

- [ ] **Step 4: Commit**

```bash
git add ml/pipeline/p05b_train_stages.py
git commit -m "feat(ml): train binary interaction + 3-class severity stages"
```

---

## Task 9: Two-stage + end-to-end evaluation

**Files:**
- Create: `ml/pipeline/p06b_evaluate_v2.py`
- Test: `ml/tests/test_evaluate_v2.py`

Evaluates the active config across S0/S1/S2: stage-1 AUC, stage-2 macro-F1, and **end-to-end
4-class** (None via stage-1 threshold 0.5, severity via stage-2) macro-F1 + per-class + accuracy.
Writes `ml/artifacts/metrics_v2_<tag>.json`.

- [ ] **Step 1: Write failing test `ml/tests/test_evaluate_v2.py`**

```python
# ml/tests/test_evaluate_v2.py
import numpy as np
from ml.pipeline.p06b_evaluate_v2 import combine_four_class


def test_combine_four_class_thresholds_none():
    # p_interact low -> None dominates; severity probs spread over Minor/Mod/Major
    p_interact = np.array([0.2, 0.9])
    sev = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])  # (N,3) Minor/Mod/Major
    four = combine_four_class(p_interact, sev)
    assert four.shape == (2, 4)                          # None,Minor,Moderate,Major
    assert np.allclose(four.sum(1), 1.0)
    assert four[0].argmax() == 0                         # low interact -> None
    assert four[1].argmax() == 3                         # high interact + Major


def test_combine_four_class_rows_sum_to_one():
    four = combine_four_class(np.array([0.6]), np.array([[0.2, 0.5, 0.3]]))
    assert abs(four.sum() - 1.0) < 1e-6
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_evaluate_v2.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write `ml/pipeline/p06b_evaluate_v2.py`**

```python
# ml/pipeline/p06b_evaluate_v2.py
"""Stage 6b: evaluate the two-stage model (stage1 AUC, stage2 F1, end-to-end 4-class)."""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score
from ml.config import load_config
from ml.common import CLASSES, CLASS_TO_IDX
from ml.assemble import assemble_pair, pair_dim
from ml.bio_features import empty_bio_sets, ENTITY_TYPES
from ml.models import MLPHead
from ml.pipeline.p05b_train_stages import _load_struct, _load_bio, SEV3

SEV3_IDX = {c: i for i, c in enumerate(SEV3)}


def combine_four_class(p_interact: np.ndarray, sev_probs: np.ndarray) -> np.ndarray:
    """[None, Minor, Moderate, Major] = [1-p, p*minor, p*mod, p*major]."""
    p = np.asarray(p_interact).reshape(-1, 1)
    none = 1.0 - p
    return np.concatenate([none, p * sev_probs], axis=1)


def _matrix(rows, X, idx, bio, cfg, use_bio):
    out = [assemble_pair(X[idx[a]], X[idx[b]], bio.get(a, empty_bio_sets()),
                         bio.get(b, empty_bio_sets()), use_bio)
           for a, b in zip(rows.drugA_id, rows.drugB_id)]
    return torch.tensor(np.stack(out), dtype=torch.float32)


def main():
    cfg = load_config()
    proc, artifacts = cfg["paths"]["processed_dir"], cfg["paths"]["artifacts_dir"]
    use_bio = cfg["features"]["use_bio"]
    tag = cfg["features"]["structural"] + ("_bio" if use_bio else "")
    X, idx = _load_struct(cfg); bio = _load_bio(cfg)
    sp = pd.read_parquet(os.path.join(proc, "splits.parquet"))
    sp = sp[sp.drugA_id.isin(idx) & sp.drugB_id.isin(idx)].reset_index(drop=True)
    sp["y4"] = sp["label"].map(CLASS_TO_IDX)

    s1 = MLPHead(in_dim=pair_dim(cfg), n_out=1, hidden=cfg["train"]["hidden"])
    s1.load_state_dict(torch.load(os.path.join(artifacts, f"model_stage1_{tag}.pt"))["model_state_dict"]); s1.eval()
    s2 = MLPHead(in_dim=pair_dim(cfg), n_out=3, hidden=cfg["train"]["hidden"])
    s2.load_state_dict(torch.load(os.path.join(artifacts, f"model_stage2_{tag}.pt"))["model_state_dict"]); s2.eval()

    report = {"tag": tag, "regimes": {}}
    for regime in ["test_S0", "test_S1", "test_S2"]:
        rows = sp[sp.split == regime]
        if len(rows) == 0:
            continue
        M = _matrix(rows, X, idx, bio, cfg, use_bio)
        with torch.no_grad():
            p_int = torch.sigmoid(s1(M)).numpy().ravel()
            sev = F.softmax(s2(M), dim=1).numpy()
        y4 = rows["y4"].values
        # Stage-1 binary metrics (interact vs none)
        y_bin = (y4 > 0).astype(int)
        auc = roc_auc_score(y_bin, p_int) if 0 < y_bin.sum() < len(y_bin) else None
        # Stage-2 on true interactions only
        inter_mask = y4 > 0
        s2f1 = None
        if inter_mask.sum() > 0:
            ytrue_sev = np.array([SEV3_IDX[CLASSES[c]] for c in y4[inter_mask]])
            s2f1 = float(f1_score(ytrue_sev, sev[inter_mask].argmax(1), average="macro"))
        # End-to-end 4-class
        four = combine_four_class(p_int, sev)
        pred4 = four.argmax(1)
        per = {}
        from sklearn.metrics import precision_recall_fscore_support
        P, R, Fc, _ = precision_recall_fscore_support(y4, pred4, labels=[0, 1, 2, 3], average=None, zero_division=0)
        for i in range(4):
            per[CLASSES[i]] = {"precision": float(P[i]), "recall": float(R[i]), "f1": float(Fc[i])}
        report["regimes"][regime] = {
            "stage1_auc": float(auc) if auc is not None else None,
            "stage2_macro_f1": s2f1,
            "end2end_macro_f1": float(f1_score(y4, pred4, average="macro")),
            "end2end_accuracy": float(accuracy_score(y4, pred4)),
            "per_class": per,
            "confusion": confusion_matrix(y4, pred4, labels=[0, 1, 2, 3]).tolist(),
        }
    out = os.path.join(artifacts, f"metrics_v2_{tag}.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"[done] {tag}: " + " | ".join(
        f"{r}: e2e_F1={report['regimes'][r]['end2end_macro_f1']:.3f} "
        f"(S1auc={report['regimes'][r]['stage1_auc']}, S2f1={report['regimes'][r]['stage2_macro_f1']})"
        for r in report["regimes"]))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + stage**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_evaluate_v2.py -v`
Expected: 2 passed.

Run: `/Users/zeyadzaher/grad-project/.venv/bin/python -m ml.pipeline.p06b_evaluate_v2`
Expected: a per-regime line with end-to-end macro-F1 for `light_bio`, and `metrics_v2_light_bio.json` written. **Compare the S2 end2end_macro_f1 to iteration 1's 0.297**, and check whether Minor per-class F1 in S2 is now > 0.

- [ ] **Step 5: Commit**

```bash
git add ml/pipeline/p06b_evaluate_v2.py ml/tests/test_evaluate_v2.py
git commit -m "feat(ml): two-stage + end-to-end evaluation"
```

---

## Task 10: Ablation runner

**Files:**
- Create: `ml/run_ablation.py`

Runs the full v2 sub-pipeline for each feature config and writes a comparison table. Each config
re-runs features → train → evaluate by mutating an in-memory config and pointing `ml.config._DEFAULT`
at a temp file.

- [ ] **Step 1: Write `ml/run_ablation.py`**

```python
# ml/run_ablation.py
"""Sweep feature configs and emit an ablation table of end-to-end S2 macro-F1."""
from __future__ import annotations
import os, json, copy, yaml
import ml.config as mlconfig
from ml.config import load_config

# (structural, use_bio) configurations to compare.
CONFIGS = [
    ("light", False, "struct_light"),
    ("light", True, "struct_light+bio"),
    ("chemberta", True, "chemberta+bio"),
]


def _run_stage(modpath):
    import importlib
    importlib.import_module(modpath).main()


def main():
    base = load_config()
    artifacts = base["paths"]["artifacts_dir"]
    results = []
    for structural, use_bio, label in CONFIGS:
        cfg = copy.deepcopy(base)
        cfg["features"]["structural"] = structural
        cfg["features"]["use_bio"] = use_bio
        # ChemBERTa requires transformers; skip gracefully if unavailable.
        if structural == "chemberta":
            from ml.embeddings import chemberta_available
            if not chemberta_available():
                print(f"[skip] {label}: transformers not installed")
                results.append((label, None))
                continue
        tmp = f"/tmp/ablation_{structural}_{use_bio}.yaml"
        yaml.safe_dump(cfg, open(tmp, "w"))
        mlconfig._DEFAULT = tmp
        print(f"\n===== ablation: {label} =====")
        _run_stage("ml.pipeline.p03b_features_v2")
        _run_stage("ml.pipeline.p05b_train_stages")
        _run_stage("ml.pipeline.p06b_evaluate_v2")
        tag = structural + ("_bio" if use_bio else "")
        m = json.load(open(os.path.join(artifacts, f"metrics_v2_{tag}.json")))
        s2 = m["regimes"].get("test_S2", {})
        results.append((label, s2.get("end2end_macro_f1")))

    lines = ["# Ablation — end-to-end S2 macro-F1\n",
             "| config | S2 macro-F1 |", "|---|---|",
             "| iteration-1 baseline (single 4-class) | 0.297 |"]
    for label, f1 in results:
        lines.append(f"| {label} | {f1:.3f} |" if f1 is not None else f"| {label} | skipped |")
    open(os.path.join(artifacts, "ablation_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the ablation** (light configs always run; chemberta runs only if transformers installed)

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m ml.run_ablation`
Expected: a printed table comparing `struct_light`, `struct_light+bio`, (`chemberta+bio` or skipped) vs the iteration-1 baseline (0.297), and `ml/artifacts/ablation_table.md` written. Report the table.

- [ ] **Step 3: Commit**

```bash
git add ml/run_ablation.py
git commit -m "feat(ml): ablation runner across feature configs"
```

---

## Task 11: Export the two-stage model + bio lookup

**Files:**
- Create: `ml/pipeline/p07b_export_v2.py`
- Test: `ml/tests/test_evaluate_v2.py` (append a metadata test)

Exports the chosen config's stage-1 + stage-2 models, metadata, and a **canonical-SMILES → bio
sets** lookup (so serving can attach bio features for known drugs; unknown → zeros).

- [ ] **Step 1: Append a failing test to `ml/tests/test_evaluate_v2.py`**

```python
from ml.pipeline.p07b_export_v2 import build_v2_metadata


def test_v2_metadata_fields():
    md = build_v2_metadata(structural="light", use_bio=True, light_fp_bits=1024,
                           fp_radius=2, pair_dim=3000, hidden=512,
                           interaction_threshold=0.5)
    assert md["classes"] == ["None", "Minor", "Moderate", "Major"]
    assert md["severity_classes"] == ["Minor", "Moderate", "Major"]
    assert md["structural"] == "light" and md["use_bio"] is True
    assert md["interaction_threshold"] == 0.5 and md["pair_dim"] == 3000
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_evaluate_v2.py::test_v2_metadata_fields -v`
Expected: FAIL (module/function missing).

- [ ] **Step 3: Write `ml/pipeline/p07b_export_v2.py`**

```python
# ml/pipeline/p07b_export_v2.py
"""Stage 7b: export two-stage models + metadata + SMILES->bio lookup for serving."""
from __future__ import annotations
import os, json, shutil
import pandas as pd
from rdkit import Chem
import torch
from ml.config import load_config
from ml.common import CLASSES
from ml.assemble import pair_dim
from ml.bio_features import load_bio_sets, ENTITY_TYPES
from ml.pipeline.p05b_train_stages import SEV3


def build_v2_metadata(structural, use_bio, light_fp_bits, fp_radius, pair_dim, hidden,
                      interaction_threshold):
    return {
        "version": 2,
        "classes": CLASSES,
        "severity_classes": SEV3,
        "structural": structural,
        "use_bio": use_bio,
        "light_fp_bits": light_fp_bits,
        "fp_radius": fp_radius,
        "pair_dim": pair_dim,
        "hidden": hidden,
        "interaction_threshold": interaction_threshold,
    }


def _canonical(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m) if m is not None else None


def main():
    cfg = load_config()
    artifacts = cfg["paths"]["artifacts_dir"]
    raw = cfg["paths"]["raw_dir"]
    export = os.path.join(artifacts, "export_v2"); os.makedirs(export, exist_ok=True)
    use_bio = cfg["features"]["use_bio"]
    tag = cfg["features"]["structural"] + ("_bio" if use_bio else "")

    for stage in ["stage1", "stage2"]:
        shutil.copy(os.path.join(artifacts, f"model_{stage}_{tag}.pt"),
                    os.path.join(export, f"model_{stage}.pt"))

    md = build_v2_metadata(
        structural=cfg["features"]["structural"], use_bio=use_bio,
        light_fp_bits=int(cfg["features"]["light_fp_bits"]), fp_radius=int(cfg["features"]["fp_radius"]),
        pair_dim=pair_dim(cfg), hidden=int(cfg["train"]["hidden"]), interaction_threshold=0.5,
    )
    json.dump(md, open(os.path.join(export, "metadata.json"), "w"), indent=2)

    # canonical SMILES -> bio sets (lists) for known drugs.
    drugs = pd.read_csv(os.path.join(raw, "drugs.csv")).drop_duplicates("drug_id")
    bio = load_bio_sets(cfg["bio"]["merged_csv"])
    lookup = {}
    for did, smi in zip(drugs["drug_id"], drugs["smiles"].astype(str)):
        cs = _canonical(smi)
        if cs and did in bio:
            lookup[cs] = {t: sorted(bio[did][t]) for t in ENTITY_TYPES}
    json.dump(lookup, open(os.path.join(export, "bio_lookup.json"), "w"))
    print(f"[done] exported two-stage ({tag}) + metadata + bio_lookup ({len(lookup)} drugs) -> {export}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + stage**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_evaluate_v2.py -v`
Expected: all pass.

Run: `/Users/zeyadzaher/grad-project/.venv/bin/python -m ml.pipeline.p07b_export_v2`
Expected: `ml/artifacts/export_v2/{model_stage1.pt, model_stage2.pt, metadata.json, bio_lookup.json}` written.

- [ ] **Step 5: Commit**

```bash
git add ml/pipeline/p07b_export_v2.py ml/tests/test_evaluate_v2.py
git commit -m "feat(ml): export two-stage model + metadata + SMILES->bio lookup"
```

---

## Task 12: Two-stage serving in inference-py

**Files:**
- Modify: `inference-py/app/model.py`
- Modify: `inference-py/app/main.py`
- Copy: export artifacts → `inference-py/app/models/`
- Test: `inference-py/tests/test_model_v2.py`

- [ ] **Step 1: Copy artifacts into the service**

```bash
cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction
cp ml/artifacts/export_v2/model_stage1.pt inference-py/app/models/stage1.pt
cp ml/artifacts/export_v2/model_stage2.pt inference-py/app/models/stage2.pt
cp ml/artifacts/export_v2/metadata.json inference-py/app/models/severity_metadata.json
cp ml/artifacts/export_v2/bio_lookup.json inference-py/app/models/bio_lookup.json
```

- [ ] **Step 2: Write failing test `inference-py/tests/test_model_v2.py`**

```python
# inference-py/tests/test_model_v2.py
import os
import pytest
from app.model import TwoStageDDIModel

D = os.path.join(os.path.dirname(__file__), "..", "app", "models")
S1 = os.path.join(D, "stage1.pt"); S2 = os.path.join(D, "stage2.pt")
META = os.path.join(D, "severity_metadata.json"); BIO = os.path.join(D, "bio_lookup.json")


@pytest.mark.skipif(not os.path.exists(S1), reason="export not present")
def test_two_stage_predict_fields_and_consistency():
    m = TwoStageDDIModel(S1, S2, META, BIO, device="cpu")
    out = m.predict_with_severity("CC(=O)OC1=CC=CC=C1C(=O)O", "CCO")
    assert set(out["probabilities"]) == {"None", "Minor", "Moderate", "Major"}
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-4
    assert 0.0 <= out["interactionProbability"] <= 1.0
    if out["interactionProbability"] < 0.5:
        assert out["severity"] == "None"
    else:
        assert out["severity"] in {"Minor", "Moderate", "Major"}


def test_two_stage_symmetric():
    m = TwoStageDDIModel(S1, S2, META, BIO, device="cpu")
    a = m.predict_with_severity("CCO", "CCN")
    b = m.predict_with_severity("CCN", "CCO")
    assert abs(a["interactionProbability"] - b["interactionProbability"]) < 1e-5
```

- [ ] **Step 3: Run to confirm failure**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction/inference-py && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest tests/test_model_v2.py -v`
Expected: FAIL (`ImportError: TwoStageDDIModel`).

- [ ] **Step 4: Replace `inference-py/app/model.py`** with the two-stage version

```python
# inference-py/app/model.py
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors, QED

CLASSES = ["None", "Minor", "Moderate", "Major"]
SEV3 = ["Minor", "Moderate", "Major"]
ENTITY_TYPES = ["targets", "enzymes", "transporters", "carriers"]

# MUST match ml/features.py LIGHT_DESCRIPTORS order exactly.
_LIGHT_DESCRIPTORS = [
    Descriptors.MolWt, Descriptors.ExactMolWt, Descriptors.HeavyAtomCount, Descriptors.NumHAcceptors,
    Descriptors.NumHDonors, Descriptors.NumRotatableBonds, rdMolDescriptors.CalcNumAromaticRings,
    rdMolDescriptors.CalcNumAliphaticRings, rdMolDescriptors.CalcNumSaturatedRings, Descriptors.RingCount,
    rdMolDescriptors.CalcFractionCSP3, Descriptors.TPSA, Crippen.MolLogP, Crippen.MolMR,
    Descriptors.NumValenceElectrons, rdMolDescriptors.CalcNumHeteroatoms, Descriptors.NOCount,
    Descriptors.NHOHCount, rdMolDescriptors.CalcNumAromaticHeterocycles,
    rdMolDescriptors.CalcNumAliphaticHeterocycles, rdMolDescriptors.CalcNumAromaticCarbocycles,
    rdMolDescriptors.CalcNumAliphaticCarbocycles, Descriptors.LabuteASA, Descriptors.BalabanJ,
    Descriptors.BertzCT, Descriptors.Chi0, Descriptors.Chi1, Descriptors.Chi0n, Descriptors.Chi1n,
    Descriptors.HallKierAlpha, Descriptors.Kappa1, Descriptors.Kappa2, Descriptors.Kappa3,
    Descriptors.MaxPartialCharge, Descriptors.MinPartialCharge, Descriptors.NumRadicalElectrons,
    QED.qed, lambda m: Chem.GetFormalCharge(m),
]


class MLPHead(nn.Module):
    def __init__(self, in_dim, n_out, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def _light_vector(smiles, fp_bits, radius):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = np.zeros((fp_bits,), dtype=np.float32)
    cfp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=fp_bits)
    for idx, cnt in cfp.GetNonzeroElements().items():
        fp[idx] = float(cnt)
    desc = np.zeros((len(_LIGHT_DESCRIPTORS),), dtype=np.float32)
    for i, fn in enumerate(_LIGHT_DESCRIPTORS):
        try:
            v = float(fn(mol)); desc[i] = v if np.isfinite(v) else 0.0
        except Exception:
            desc[i] = 0.0
    return np.concatenate([fp, desc]).astype(np.float32)


def _struct_pair(va, vb):
    return np.concatenate([va + vb, np.abs(va - vb), va * vb]).astype(np.float32)


def _bio_pair(a, b):
    vals = []
    for t in ENTITY_TYPES:
        sa, sb = set(a.get(t, [])), set(b.get(t, []))
        inter = len(sa & sb); union = len(sa | sb)
        jacc = inter / union if union else 0.0
        vals += [float(inter), float(jacc), float(len(sa) + len(sb)), float(abs(len(sa) - len(sb)))]
    return np.array(vals, dtype=np.float32)


def _canonical(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m) if m is not None else None


class TwoStageDDIModel:
    def __init__(self, stage1_path, stage2_path, metadata_path, bio_lookup_path, device="cpu"):
        self.device = torch.device(device)
        self.meta = json.load(open(metadata_path))
        self.fp_bits = int(self.meta["light_fp_bits"])
        self.radius = int(self.meta["fp_radius"])
        self.use_bio = bool(self.meta["use_bio"])
        self.threshold = float(self.meta.get("interaction_threshold", 0.5))
        in_dim = int(self.meta["pair_dim"]); hidden = int(self.meta["hidden"])
        self.s1 = MLPHead(in_dim, 1, hidden); self.s1.load_state_dict(
            torch.load(stage1_path, map_location=self.device)["model_state_dict"]); self.s1.eval()
        self.s2 = MLPHead(in_dim, 3, hidden); self.s2.load_state_dict(
            torch.load(stage2_path, map_location=self.device)["model_state_dict"]); self.s2.eval()
        self.bio = json.load(open(bio_lookup_path)) if self.use_bio else {}

    def _bio_for(self, smiles):
        cs = _canonical(smiles)
        return self.bio.get(cs, {t: [] for t in ENTITY_TYPES}) if cs else {t: [] for t in ENTITY_TYPES}

    @torch.no_grad()
    def predict_with_severity(self, smiles_a, smiles_b):
        va = _light_vector(smiles_a, self.fp_bits, self.radius)
        vb = _light_vector(smiles_b, self.fp_bits, self.radius)
        blocks = [_struct_pair(va, vb)]
        if self.use_bio:
            blocks.append(_bio_pair(self._bio_for(smiles_a), self._bio_for(smiles_b)))
        feats = torch.tensor(np.concatenate(blocks), device=self.device).unsqueeze(0)
        p_int = float(torch.sigmoid(self.s1(feats)).item())
        sev = F.softmax(self.s2(feats), dim=1).squeeze(0).numpy()
        prob_map = {"None": 1.0 - p_int, "Minor": p_int * float(sev[0]),
                    "Moderate": p_int * float(sev[1]), "Major": p_int * float(sev[2])}
        if p_int < self.threshold:
            severity = "None"
        else:
            severity = SEV3[int(np.argmax(sev))]
        return {"interactionProbability": p_int, "severity": severity, "probabilities": prob_map}


# Backwards-compatible alias so existing imports keep working if any.
DDIModel = TwoStageDDIModel
```

- [ ] **Step 5: Update `inference-py/app/main.py`** construction

```python
# inference-py/app/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import TwoStageDDIModel

app = FastAPI(title="DDI Inference Service")


class PredictReq(BaseModel):
    smilesA: str
    smilesB: str


MODELS_DIR = os.getenv("MODELS_DIR", "/models")
DEVICE = os.getenv("DEVICE", "cpu")
ddi = None


@app.on_event("startup")
def startup():
    global ddi
    ddi = TwoStageDDIModel(
        os.path.join(MODELS_DIR, "stage1.pt"),
        os.path.join(MODELS_DIR, "stage2.pt"),
        os.path.join(MODELS_DIR, "severity_metadata.json"),
        os.path.join(MODELS_DIR, "bio_lookup.json"),
        device=DEVICE,
    )


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
def predict(req: PredictReq):
    try:
        return ddi.predict_with_severity(req.smilesA, req.smilesB)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="inference error")
```

- [ ] **Step 6: Update `docker-compose.yml` env** — replace `MODEL_PATH`/`META_PATH` for the
inference service with `MODELS_DIR=/models` (the volume mount `./inference-py/app/models:/models:ro`
already provides stage1.pt/stage2.pt/severity_metadata.json/bio_lookup.json). Keep `DEVICE=cpu`.

- [ ] **Step 7: Run serving tests**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction/inference-py && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest tests/ -v`
Expected: the new `test_model_v2.py` passes (2 tests). The old `test_model.py` referenced the
single-model `DDIModel(model_path, metadata_path)` signature — since `DDIModel` is now an alias
for `TwoStageDDIModel`, delete the obsolete `inference-py/tests/test_model.py` (it tests the
retired single-stage API). `git rm inference-py/tests/test_model.py`.

- [ ] **Step 8: Commit**

```bash
cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction
git rm inference-py/tests/test_model.py
git add inference-py/app/model.py inference-py/app/main.py inference-py/tests/test_model_v2.py docker-compose.yml
git add -f inference-py/app/models/stage1.pt inference-py/app/models/stage2.pt \
        inference-py/app/models/severity_metadata.json inference-py/app/models/bio_lookup.json
git rm --cached inference-py/app/models/severity_model.pt 2>/dev/null || true
git commit -m "feat(inference): two-stage serving with biological lookup"
```

---

## Task 13: End-to-end verification + results

**Files:** `ml/README.md` (modify), no code

- [ ] **Step 1: Full test suite**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests inference-py/tests -q`
Expected: all pass.

- [ ] **Step 2: Confirm the ablation table exists and read it**

Run: `cat ml/artifacts/ablation_table.md`
Expected: S2 macro-F1 rows for each config vs the 0.297 baseline.

- [ ] **Step 3: Docker stack smoke test**

```bash
docker compose up --build -d
sleep 25
curl -s -X POST localhost:8001/predict -H 'Content-Type: application/json' \
  -d '{"smilesA":"CC(=O)OC1=CC=CC=C1C(=O)O","smilesB":"OC(=O)C1=CC=CC=C1NC1=CC=CC=C1"}' | python3 -m json.tool
docker compose down
```
Expected: JSON with `interactionProbability`, `severity`, `probabilities` (4 keys summing to ~1).
(If host port 8080 is taken by an unrelated process, the inference service on 8001 is sufficient
for this smoke test.)

- [ ] **Step 4: Record results in `ml/README.md`**

Add an "Iteration 2 results" section with the ablation table, the best config's S0/S1/S2
end-to-end macro-F1 + accuracy vs iteration 1 (0.297 / 46%), and the Minor-class S2 F1 (did it
move off 0.00?). State honestly which levers helped and which did not.

```bash
git add ml/README.md
git commit -m "docs(ml): record iteration 2 ablation and results"
```

---

## Self-Review notes (addressed)

- **Spec §1 (two-stage):** Tasks 5/8 build the interaction dataset + train stage1 (binary on
  subsampled 1.8M edges) and stage2 (3-class on DDInter); Task 9 combines via `combine_four_class`
  and threshold 0.5; serving in Task 12 mirrors it. ✓
- **Spec §2 (bio features):** Task 1 parses entity sets + overlap features (DrugBank-ID join);
  Tasks 6/8/9/12 thread them through training, eval, and serving (with a SMILES lookup for the app). ✓
- **Spec §3 (structure):** Task 2 adds count-Morgan + ~38 descriptors (`light`); Task 3 adds the
  optional ChemBERTa block; the config switch in Task 4 selects between them; ablation Task 10
  compares them. ✓
- **Spec §5 (ablation/eval):** Task 9 reports stage1 AUC, stage2 F1, end-to-end 4-class per
  S0/S1/S2; Task 10 emits the ablation table vs the 0.297 baseline; Task 13 records it. ✓
- **Spec §6 (structure):** new focused modules (`bio_features`, `embeddings`, `assemble`) plus
  v2 pipeline stages; backend/mobile unchanged (identical JSON). ✓
- **Type consistency:** `pair_dim(cfg)`, `assemble_pair(...)`, `MLPHead(in_dim, n_out)`,
  `combine_four_class`, the `<structural>(+_bio)` artifact tag, and the descriptor order are used
  identically across `ml/assemble.py`, `p05b`, `p06b`, `p07b`, and `inference-py/app/model.py`. ✓
- **Serving feature parity:** `inference-py/app/model.py` `_light_vector`/`_struct_pair`/`_bio_pair`
  replicate `ml/features.light_drug_vector` + `pair_features` + `ml/bio_features.bio_pair_features`
  exactly (same descriptor list/order, same count-Morgan, same overlap feature order). Task 13
  Docker test verifies end-to-end. ✓
