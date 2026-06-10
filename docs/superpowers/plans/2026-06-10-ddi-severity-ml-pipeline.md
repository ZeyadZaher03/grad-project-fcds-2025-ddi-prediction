# DDI Severity ML Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, honestly-evaluated ML pipeline that predicts a real 4-class drug-pair severity (`None`/`Minor`/`Moderate`/`Major`) and serve the measured model in the app.

**Architecture:** A staged `ml/` pipeline inside the app repo: download DDInter severity labels → join to existing SMILES → build labeled pairs with hard negatives → drug-disjoint S0/S1/S2 splits → train an inductive fingerprint MLP and a GraphSAGE GNN head-to-head → evaluate with calibration → export the winner → rewire `inference-py` to serve it. Pure data-processing functions are unit-tested (TDD); training/eval scripts have smoke tests.

**Tech Stack:** Python 3.13, PyTorch, PyTorch Geometric, RDKit, scikit-learn, pandas, pyarrow, matplotlib, pytest. Reuses the existing working interpreter at `/Users/zeyadzaher/grad-project/.venv/bin/python` (already has torch 2.9 + torch_geometric 2.7 + rdkit).

**Conventions used by every task:**
- Repo root: `/Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction`
- Interpreter alias (use literally in every command): `PY=/Users/zeyadzaher/grad-project/.venv/bin/python`
- Run tests from repo root with `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction`
- Each pipeline stage is idempotent: reads inputs from disk, writes outputs to disk.

---

## File Structure

```
ml/
  __init__.py
  config.py                # typed config dataclass + load from config.yaml
  config.yaml              # paths, seeds, fp size, neg ratio, split fractions, hyperparams
  requirements.txt         # documented deps (env is reused, not reinstalled)
  README.md                # how to reproduce end-to-end
  common.py                # CLASSES constant, normalize_name, seed helpers
  features.py              # smiles->fingerprint+descriptors, symmetric pair features
  models.py                # PairMLP4 (deployable) + SageDDI (GNN)
  pipeline/
    __init__.py
    p01_download.py        # fetch DDInter CSVs + stage drugs.csv & edges
    p02_build_labels.py    # join + mask + negative sampling -> pairs.parquet
    p03_features.py        # precompute per-drug feature matrix
    p04_splits.py          # drug-disjoint S0/S1/S2 assignment
    p05_train.py           # train PairMLP4 and SageDDI
    p06_evaluate.py        # metrics + confusion + calibration + learning curves
    p07_export.py          # winner selection + artifact + metadata for inference-py
  run.py                   # orchestrates p01..p07
  data/
    raw/                   # downloaded DDInter + staged drugs.csv, edges
    processed/             # pairs.parquet, drug_features.npz, splits.parquet
  artifacts/               # trained models, metrics.json, report.md, plots, export/
  tests/
    __init__.py
    test_common.py
    test_features.py
    test_build_labels.py
    test_splits.py
    test_models.py
    test_evaluate.py
    test_export.py
inference-py/app/model.py  # MODIFY: load 4-class model + metadata, drop fake thresholds
inference-py/app/main.py   # MODIFY: response schema gains per-class probabilities
backend-go/inference.go    # MODIFY: pass through new fields
mobile-rn/App.js           # MODIFY: display real severity + class probabilities
```

**Class order is fixed everywhere** as `["None", "Minor", "Moderate", "Major"]` (index 0..3). Defined once in `ml/common.py` and read from metadata at serving time.

---

## Task 0: Scaffold the `ml/` package

**Files:**
- Create: `ml/__init__.py`, `ml/pipeline/__init__.py`, `ml/tests/__init__.py`
- Create: `ml/common.py`, `ml/config.py`, `ml/config.yaml`, `ml/requirements.txt`, `ml/README.md`
- Create: `ml/data/.gitkeep`, `ml/artifacts/.gitkeep`
- Test: `ml/tests/test_common.py`

- [ ] **Step 1: Create package directories and empty init files**

```bash
cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction
mkdir -p ml/pipeline ml/tests ml/data/raw ml/data/processed ml/artifacts
touch ml/__init__.py ml/pipeline/__init__.py ml/tests/__init__.py
touch ml/data/.gitkeep ml/artifacts/.gitkeep
```

- [ ] **Step 2: Write `ml/common.py`**

```python
# ml/common.py
"""Shared constants and helpers for the DDI severity pipeline."""
from __future__ import annotations
import re
import random
import numpy as np

# Fixed class order used EVERYWHERE (training, metrics, serving metadata).
CLASSES = ["None", "Minor", "Moderate", "Major"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
SEVERITY_CLASSES = ["Minor", "Moderate", "Major"]  # excludes "None"

# DDInter "Level" -> our class. "Unknown" maps to None (dropped from positives).
LEVEL_MAP = {"Minor": "Minor", "Moderate": "Moderate", "Major": "Major"}


def normalize_name(name: str) -> str:
    """Lowercase and strip everything except a-z0-9 for name-based joins."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
```

- [ ] **Step 3: Write the failing test `ml/tests/test_common.py`**

```python
# ml/tests/test_common.py
from ml.common import normalize_name, CLASSES, CLASS_TO_IDX, LEVEL_MAP


def test_normalize_name_strips_punct_and_case():
    assert normalize_name("Acetyl-Salicylic Acid") == "acetylsalicylicacid"
    assert normalize_name("Vitamin B12 (cobalamin)") == "vitaminb12cobalamin"


def test_class_order_is_fixed():
    assert CLASSES == ["None", "Minor", "Moderate", "Major"]
    assert CLASS_TO_IDX["None"] == 0 and CLASS_TO_IDX["Major"] == 3


def test_level_map_excludes_unknown():
    assert "Unknown" not in LEVEL_MAP
    assert LEVEL_MAP["Moderate"] == "Moderate"
```

- [ ] **Step 4: Run the test**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_common.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write `ml/config.yaml`**

```yaml
# ml/config.yaml
seed: 42

paths:
  drugs_csv: /Users/zeyadzaher/grad-project/data/drugs.csv
  drugbank_edges_csv: "/Users/zeyadzaher/grad-project/learn /drug_interaction_edges.csv"
  raw_dir: ml/data/raw
  processed_dir: ml/data/processed
  artifacts_dir: ml/artifacts

ddinter:
  base_url: https://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_
  categories: [A, B, D, H, L, P, R, V]

features:
  fp_radius: 2
  fp_bits: 2048
  descriptors: [MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA, NumRotatableBonds, NumAromaticRings, FractionCSP3]

negatives:
  ratio: 1.0           # number of None pairs = ratio * number of positive pairs
  hard_fraction: 0.5   # fraction of negatives drawn from structurally similar pairs

split:
  test_drug_frac: 0.20  # fraction of drugs held out entirely (feeds S1/S2)
  val_frac: 0.10        # fraction of train-train pairs used for validation
  s0_frac: 0.10         # fraction of train-train pairs held out as "both seen" test

train:
  epochs: 100
  patience: 10
  lr: 0.001
  weight_decay: 0.0001
  dropout: 0.3
  hidden: 512
  batch_size: 512
  gnn_hidden: 256
```

- [ ] **Step 6: Write `ml/config.py`**

```python
# ml/config.py
"""Load config.yaml into a plain dict with attribute-style access."""
from __future__ import annotations
import os
import yaml

_DEFAULT = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(path: str | None = None) -> dict:
    with open(path or _DEFAULT) as f:
        return yaml.safe_load(f)
```

- [ ] **Step 7: Write `ml/requirements.txt` and `ml/README.md`**

```text
# ml/requirements.txt — documents the env; reuse /Users/zeyadzaher/grad-project/.venv which already has these.
torch>=2.2
torch_geometric>=2.5
rdkit
scikit-learn
pandas
pyarrow
matplotlib
pyyaml
pytest
```

```markdown
# ml/ — DDI Severity Pipeline

Reproduces the severity model end-to-end.

## Environment
Uses the existing interpreter (already has torch + torch_geometric + rdkit):
`/Users/zeyadzaher/grad-project/.venv/bin/python`

## Run everything
```bash
cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction
/Users/zeyadzaher/grad-project/.venv/bin/python -m ml.run
```

Stages also run individually: `python -m ml.pipeline.p01_download`, etc.
Outputs land in `ml/data/processed/` and `ml/artifacts/`.
```

- [ ] **Step 8: Add `pyarrow` check and commit**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -c "import yaml, pyarrow, matplotlib; print('deps ok')"`
Expected: `deps ok`. If `pyarrow` or `matplotlib` is missing, install into the reused venv: `/Users/zeyadzaher/grad-project/.venv/bin/python -m pip install pyarrow matplotlib pyyaml`.

```bash
git add ml/
git commit -m "feat(ml): scaffold severity pipeline package + config"
```

---

## Task 1: Download DDInter labels and stage inputs

**Files:**
- Create: `ml/pipeline/p01_download.py`
- Test: covered by a smoke run (network); pure URL builder is unit-tested inline.

- [ ] **Step 1: Write the failing test for the URL builder in `ml/tests/test_build_labels.py`**

```python
# ml/tests/test_build_labels.py
from ml.pipeline.p01_download import ddinter_urls


def test_ddinter_urls_built_from_categories():
    cfg = {"ddinter": {"base_url": "https://x/code_", "categories": ["A", "P"]}}
    urls = ddinter_urls(cfg)
    assert urls == {
        "A": "https://x/code_A.csv",
        "P": "https://x/code_P.csv",
    }
```

- [ ] **Step 2: Run it to confirm failure**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_build_labels.py::test_ddinter_urls_built_from_categories -v`
Expected: FAIL (`ModuleNotFoundError: ml.pipeline.p01_download`).

- [ ] **Step 3: Write `ml/pipeline/p01_download.py`**

```python
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
```

- [ ] **Step 4: Run the unit test (passes) then run the stage**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_build_labels.py::test_ddinter_urls_built_from_categories -v`
Expected: PASS.

Run: `/Users/zeyadzaher/grad-project/.venv/bin/python -m ml.pipeline.p01_download`
Expected: 8 `[get ]`/`[skip]` lines, then `[done] DDInter rows downloaded: N ...` with N in the tens of thousands.

- [ ] **Step 5: Commit**

```bash
git add ml/pipeline/p01_download.py ml/tests/test_build_labels.py
git commit -m "feat(ml): download DDInter labels and stage inputs"
```

---

## Task 2: Build labeled pairs (join + mask + negatives)

**Files:**
- Modify: `ml/common.py` (already has helpers)
- Create: `ml/pipeline/p02_build_labels.py`
- Test: `ml/tests/test_build_labels.py` (extend)

This stage produces `ml/data/processed/pairs.parquet` with columns:
`drugA_id, drugB_id, label` (label is one of `None/Minor/Moderate/Major`). Pairs are stored
once in canonical order `drugA_id < drugB_id`; both orderings are added later at training time.

- [ ] **Step 1: Write failing tests for the core pure functions (append to `ml/tests/test_build_labels.py`)**

```python
import pandas as pd
from ml.pipeline.p02_build_labels import (
    canonical_pair, build_name_map, join_positives, sample_negatives,
)


def test_canonical_pair_orders_ids():
    assert canonical_pair("DB02", "DB01") == ("DB01", "DB02")
    assert canonical_pair("DB01", "DB02") == ("DB01", "DB02")


def test_build_name_map_normalizes():
    drugs = pd.DataFrame({"drug_id": ["DB1", "DB2"], "name": ["Aspirin", "Vitamin-C"]})
    m = build_name_map(drugs)
    assert m["aspirin"] == "DB1"
    assert m["vitaminc"] == "DB2"


def test_join_positives_keeps_only_mapped_pairs_and_drops_unknown():
    drugs = pd.DataFrame({"drug_id": ["DB1", "DB2"], "name": ["Aspirin", "Warfarin"]})
    dd = pd.DataFrame({
        "Drug_A": ["Aspirin", "Aspirin", "Ghostdrug"],
        "Drug_B": ["Warfarin", "Warfarin", "Warfarin"],
        "Level":  ["Major", "Unknown", "Major"],
    })
    pos = join_positives(dd, build_name_map(drugs))
    # Unknown dropped; unmapped "Ghostdrug" dropped; one Major pair remains.
    assert len(pos) == 1
    row = pos.iloc[0]
    assert (row.drugA_id, row.drugB_id, row.label) == ("DB1", "DB2", "Major")


def test_sample_negatives_excludes_masked_pairs():
    # 4 drugs, one known interacting pair masked out; request 2 negatives.
    drug_ids = ["DB1", "DB2", "DB3", "DB4"]
    masked = {("DB1", "DB2")}
    negs = sample_negatives(drug_ids, masked, n=2, hard_fraction=0.0,
                            features=None, seed=1)
    assert len(negs) == 2
    for a, b in negs:
        assert (a, b) not in masked and a < b
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_build_labels.py -v`
Expected: import errors / failures for the four new tests.

- [ ] **Step 3: Write `ml/pipeline/p02_build_labels.py`**

```python
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
```

> Note: hard negatives are wired through `features=None` here for speed and determinism in
> the smoke run; Task 3 produces the feature matrix, and Task 5's training reads it. If you
> want hard negatives active, run Task 3 first and pass its features dict in `main()` — left
> as `None` by default so this stage has no dependency on Task 3.

- [ ] **Step 4: Run the unit tests**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_build_labels.py -v`
Expected: all tests pass.

- [ ] **Step 5: Run the stage and check coverage**

Run: `/Users/zeyadzaher/grad-project/.venv/bin/python -m ml.pipeline.p02_build_labels`
Expected: prints positives count + label distribution (Moderate largest, Minor smallest), masked count, None count, and writes `pairs.parquet`. **Coverage gate:** if positives < ~5,000 the labeled set is too thin — stop and revisit (see spec §"Out of scope" note and the DDInter coverage flag).

- [ ] **Step 6: Commit**

```bash
git add ml/pipeline/p02_build_labels.py ml/tests/test_build_labels.py
git commit -m "feat(ml): build labeled pairs from DDInter with mask + negatives"
```

---

## Task 3: Per-drug feature matrix

**Files:**
- Create: `ml/features.py`, `ml/pipeline/p03_features.py`
- Test: `ml/tests/test_features.py`

- [ ] **Step 1: Write failing tests `ml/tests/test_features.py`**

```python
# ml/tests/test_features.py
import numpy as np
from ml.features import drug_vector, pair_features, FP_BITS_DEFAULT


def test_drug_vector_shape_and_invalid_smiles():
    v = drug_vector("CC(=O)OC1=CC=CC=C1C(=O)O", fp_bits=256)  # aspirin
    assert v.shape == (256 + 8,)              # fp + 8 descriptors
    z = drug_vector("not-a-smiles", fp_bits=256)
    assert z.shape == (256 + 8,)
    assert np.all(z[:256] == 0)               # invalid -> zero fingerprint


def test_pair_features_are_symmetric():
    a = drug_vector("CCO", fp_bits=256)
    b = drug_vector("CCN", fp_bits=256)
    assert np.allclose(pair_features(a, b), pair_features(b, a))
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_features.py -v`
Expected: FAIL (`ModuleNotFoundError: ml.features`).

- [ ] **Step 3: Write `ml/features.py`**

```python
# ml/features.py
"""SMILES -> per-drug feature vector and symmetric pair features."""
from __future__ import annotations
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Crippen, rdMolDescriptors

FP_BITS_DEFAULT = 2048

# Order MUST stay fixed; mirrored in config.yaml features.descriptors.
_DESCRIPTORS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Crippen.MolLogP),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("TPSA", Descriptors.TPSA),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumAromaticRings", rdMolDescriptors.CalcNumAromaticRings),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
]
N_DESC = len(_DESCRIPTORS)


def drug_vector(smiles: str, fp_bits: int = FP_BITS_DEFAULT, radius: int = 2) -> np.ndarray:
    """Return concatenated [Morgan fingerprint (fp_bits), descriptors (8)] as float32."""
    mol = Chem.MolFromSmiles(str(smiles))
    fp = np.zeros((fp_bits,), dtype=np.float32)
    desc = np.zeros((N_DESC,), dtype=np.float32)
    if mol is not None:
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_bits)
        DataStructs.ConvertToNumpyArray(bv, fp)
        for i, (_, fn) in enumerate(_DESCRIPTORS):
            try:
                desc[i] = float(fn(mol))
            except Exception:
                desc[i] = 0.0
    return np.concatenate([fp, desc]).astype(np.float32)


def pair_features(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
    """Symmetric pair features: order-invariant by construction."""
    summ = va + vb
    diff = np.abs(va - vb)
    prod = va * vb
    return np.concatenate([summ, diff, prod]).astype(np.float32)
```

- [ ] **Step 4: Run the tests (pass)**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_features.py -v`
Expected: 2 passed.

- [ ] **Step 5: Write `ml/pipeline/p03_features.py`**

```python
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
```

- [ ] **Step 6: Run the stage and commit**

Run: `/Users/zeyadzaher/grad-project/.venv/bin/python -m ml.pipeline.p03_features`
Expected: `[done] features (8287, 2056) for 8287 drugs -> ...drug_features.npz`.

```bash
git add ml/features.py ml/pipeline/p03_features.py ml/tests/test_features.py
git commit -m "feat(ml): per-drug fingerprint+descriptor features"
```

---

## Task 4: Drug-disjoint S0/S1/S2 splits

**Files:**
- Create: `ml/pipeline/p04_splits.py`
- Test: `ml/tests/test_splits.py`

Produces `ml/data/processed/splits.parquet` adding a `split` column to each pair:
one of `train`, `val`, `test_S0`, `test_S1`, `test_S2`.

- [ ] **Step 1: Write failing tests `ml/tests/test_splits.py`**

```python
# ml/tests/test_splits.py
import pandas as pd
from ml.pipeline.p04_splits import assign_regime, split_drugs


def test_split_drugs_disjoint():
    train, test = split_drugs(["D%d" % i for i in range(100)], test_frac=0.2, seed=1)
    assert len(test) == 20 and len(train) == 80
    assert set(train).isdisjoint(set(test))


def test_assign_regime_by_membership():
    test_drugs = {"D9", "D8"}
    assert assign_regime("D1", "D2", test_drugs) == "S0"   # both seen
    assert assign_regime("D1", "D9", test_drugs) == "S1"   # one unseen
    assert assign_regime("D8", "D9", test_drugs) == "S2"   # both unseen
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_splits.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write `ml/pipeline/p04_splits.py`**

```python
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
```

- [ ] **Step 4: Run tests + stage**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_splits.py -v`
Expected: 2 passed.

Run: `/Users/zeyadzaher/grad-project/.venv/bin/python -m ml.pipeline.p04_splits`
Expected: split counts table + `[ok] S2 is fully drug-disjoint from train/val`.

- [ ] **Step 5: Commit**

```bash
git add ml/pipeline/p04_splits.py ml/tests/test_splits.py
git commit -m "feat(ml): drug-disjoint S0/S1/S2 splits with leakage assertion"
```

---

## Task 5: Models + training

**Files:**
- Create: `ml/models.py`, `ml/pipeline/p05_train.py`
- Test: `ml/tests/test_models.py`

- [ ] **Step 1: Write failing tests `ml/tests/test_models.py`**

```python
# ml/tests/test_models.py
import torch
from ml.models import PairMLP4, SageDDI


def test_pairmlp_output_shape_is_4_classes():
    m = PairMLP4(in_dim=30, hidden=16, dropout=0.1)
    out = m(torch.randn(5, 30))
    assert out.shape == (5, 4)


def test_sage_output_shape_is_4_classes():
    m = SageDDI(in_channels=12, hidden=16)
    x = torch.randn(6, 12)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    pair_index = torch.tensor([[0, 2], [3, 5]], dtype=torch.long)  # 2 pairs
    out = m(x, edge_index, pair_index)
    assert out.shape == (2, 4)
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_models.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write `ml/models.py`**

```python
# ml/models.py
"""Two 4-class severity models: deployable MLP and an inductive GraphSAGE."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

N_CLASSES = 4


class PairMLP4(nn.Module):
    """Inductive: consumes symmetric pair features -> 4-class logits."""
    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


class SageDDI(nn.Module):
    """Inductive GraphSAGE encoder + symmetric pair decoder -> 4-class logits."""
    def __init__(self, in_channels: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, N_CLASSES),
        )

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return x

    def forward(self, x, edge_index, pair_index):
        z = self.encode(x, edge_index)
        a, b = pair_index
        # Symmetric combination so predict(A,B) == predict(B,A).
        h = torch.cat([z[a] + z[b], torch.abs(z[a] - z[b])], dim=-1)
        return self.head(h)
```

- [ ] **Step 4: Run model tests (pass)**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_models.py -v`
Expected: 2 passed.

> Note: `SageDDI.head` takes `hidden*2` because the symmetric decoder concatenates
> `(z_a+z_b)` and `|z_a-z_b|` (each width `hidden`).

- [ ] **Step 5: Write `ml/pipeline/p05_train.py`**

```python
# ml/pipeline/p05_train.py
"""Stage 5: train PairMLP4 and SageDDI on the same splits; save models + curves."""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from ml.config import load_config
from ml.common import set_seed, CLASS_TO_IDX, CLASSES
from ml.features import pair_features
from ml.models import PairMLP4, SageDDI


def _load(cfg):
    proc = cfg["paths"]["processed_dir"]
    splits = pd.read_parquet(os.path.join(proc, "splits.parquet"))
    feat = np.load(os.path.join(proc, "drug_features.npz"), allow_pickle=True)
    ids = list(feat["ids"])
    idx = {d: i for i, d in enumerate(ids)}
    X = feat["X"].astype(np.float32)
    # Drop pairs whose drugs lack features (should be none, but be safe).
    splits = splits[splits.drugA_id.isin(idx) & splits.drugB_id.isin(idx)].reset_index(drop=True)
    splits["y"] = splits["label"].map(CLASS_TO_IDX)
    return splits, X, idx


def _class_weights(y, n=4):
    counts = np.bincount(y, minlength=n).astype(np.float32)
    w = counts.sum() / (n * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def _mlp_matrix(splits, X, idx, mask):
    rows = splits[mask]
    feats = np.stack([pair_features(X[idx[a]], X[idx[b]])
                      for a, b in zip(rows.drugA_id, rows.drugB_id)])
    return torch.tensor(feats), torch.tensor(rows["y"].values, dtype=torch.long)


def train_mlp(cfg, splits, X, idx, artifacts):
    set_seed(cfg["seed"])
    Xtr, ytr = _mlp_matrix(splits, X, idx, splits.split == "train")
    Xva, yva = _mlp_matrix(splits, X, idx, splits.split == "val")
    model = PairMLP4(in_dim=Xtr.shape[1], hidden=cfg["train"]["hidden"],
                     dropout=cfg["train"]["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    w = _class_weights(ytr.numpy())
    bs = cfg["train"]["batch_size"]
    best_f1, best_state, curve, bad = -1, None, [], 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[b]), ytr[b], weight=w)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            tr_loss = F.cross_entropy(model(Xtr), ytr, weight=w).item()
            va_logits = model(Xva)
            va_loss = F.cross_entropy(va_logits, yva, weight=w).item()
            va_f1 = f1_score(yva.numpy(), va_logits.argmax(1).numpy(), average="macro")
        curve.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_macro_f1": va_f1})
        if va_f1 > best_f1:
            best_f1, best_state, bad = va_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg["train"]["patience"]:
                break
    model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict(), "in_dim": Xtr.shape[1],
                "kind": "mlp"}, os.path.join(artifacts, "model_mlp.pt"))
    json.dump(curve, open(os.path.join(artifacts, "curve_mlp.json"), "w"))
    print(f"[mlp] best val macro-F1={best_f1:.4f} over {len(curve)} epochs")


def train_gnn(cfg, splits, X, idx, artifacts):
    set_seed(cfg["seed"])
    x = torch.tensor(X)
    # Message-passing graph = TRAIN positive interactions only (no val/test leakage).
    tr_pos = splits[(splits.split == "train") & (splits.label != "None")]
    ei = torch.tensor([[idx[a] for a in tr_pos.drugA_id] + [idx[b] for b in tr_pos.drugB_id],
                       [idx[b] for b in tr_pos.drugB_id] + [idx[a] for a in tr_pos.drugA_id]],
                      dtype=torch.long)
    def pidx(mask):
        r = splits[mask]
        return (torch.tensor([[idx[a] for a in r.drugA_id], [idx[b] for b in r.drugB_id]],
                             dtype=torch.long),
                torch.tensor(r["y"].values, dtype=torch.long))
    ptr, ytr = pidx(splits.split == "train")
    pva, yva = pidx(splits.split == "val")
    model = SageDDI(in_channels=x.shape[1], hidden=cfg["train"]["gnn_hidden"],
                    dropout=cfg["train"]["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    w = _class_weights(ytr.numpy())
    best_f1, best_state, curve, bad = -1, None, [], 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(x, ei, ptr), ytr, weight=w)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va_logits = model(x, ei, pva)
            va_loss = F.cross_entropy(va_logits, yva, weight=w).item()
            va_f1 = f1_score(yva.numpy(), va_logits.argmax(1).numpy(), average="macro")
        curve.append({"epoch": epoch, "train_loss": loss.item(), "val_loss": va_loss, "val_macro_f1": va_f1})
        if va_f1 > best_f1:
            best_f1, best_state, bad = va_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg["train"]["patience"]:
                break
    model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict(), "in_channels": x.shape[1],
                "kind": "gnn"}, os.path.join(artifacts, "model_gnn.pt"))
    json.dump(curve, open(os.path.join(artifacts, "curve_gnn.json"), "w"))
    print(f"[gnn] best val macro-F1={best_f1:.4f} over {len(curve)} epochs")


def main():
    cfg = load_config()
    artifacts = cfg["paths"]["artifacts_dir"]
    os.makedirs(artifacts, exist_ok=True)
    splits, X, idx = _load(cfg)
    train_mlp(cfg, splits, X, idx, artifacts)
    train_gnn(cfg, splits, X, idx, artifacts)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke test (tiny epoch count) then full run**

Smoke run with reduced epochs to confirm the loop executes end-to-end:
Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -c "import yaml; c=yaml.safe_load(open('ml/config.yaml')); c['train']['epochs']=2; yaml.safe_dump(c, open('/tmp/ml_smoke.yaml','w')); from ml.pipeline import p05_train as t; from ml import config as cf; cf._DEFAULT='/tmp/ml_smoke.yaml'; t.main()"`
Expected: `[mlp] best val macro-F1=...` and `[gnn] best val macro-F1=...` print without error, and `ml/artifacts/model_mlp.pt`, `model_gnn.pt`, `curve_*.json` exist.

Full run:
Run: `/Users/zeyadzaher/grad-project/.venv/bin/python -m ml.pipeline.p05_train`
Expected: both models trained with their best val macro-F1 reported.

- [ ] **Step 7: Commit**

```bash
git add ml/models.py ml/pipeline/p05_train.py ml/tests/test_models.py
git commit -m "feat(ml): 4-class MLP + GraphSAGE training with class weights + early stop"
```

---

## Task 6: Evaluation, calibration, baselines, report

**Files:**
- Create: `ml/pipeline/p06_evaluate.py`
- Test: `ml/tests/test_evaluate.py`

- [ ] **Step 1: Write failing tests `ml/tests/test_evaluate.py`**

```python
# ml/tests/test_evaluate.py
import numpy as np
import torch
from ml.pipeline.p06_evaluate import fit_temperature, expected_calibration_error


def test_temperature_reduces_overconfidence():
    # Overconfident logits (scaled up) should get T > 1 to calibrate.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=400)
    base = np.zeros((400, 4)); base[np.arange(400), y] = 1.0
    logits = torch.tensor(base * 8.0, dtype=torch.float32)  # very confident
    labels = torch.tensor(y, dtype=torch.long)
    T = fit_temperature(logits, labels)
    assert T > 1.0


def test_ece_is_zero_for_perfect_calibration():
    probs = np.eye(4)[np.array([0, 1, 2, 3])]
    y = np.array([0, 1, 2, 3])
    assert expected_calibration_error(probs, y) < 1e-6
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_evaluate.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write `ml/pipeline/p06_evaluate.py`**

```python
# ml/pipeline/p06_evaluate.py
"""Stage 6: evaluate both models on S0/S1/S2 + baselines, calibrate, write report."""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score, average_precision_score)
from sklearn.linear_model import LogisticRegression
from ml.config import load_config
from ml.common import CLASSES, CLASS_TO_IDX
from ml.features import pair_features
from ml.models import PairMLP4, SageDDI


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Single temperature scaling on validation logits (Guo et al. 2017)."""
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().clamp(min=0.05).item())


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _metrics_block(y_true, probs):
    pred = probs.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2, 3],
                                                 average=None, zero_division=0)
    block = {
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "per_class": {CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                   "f1": float(f[i])} for i in range(4)},
        "confusion": confusion_matrix(y_true, pred, labels=[0, 1, 2, 3]).tolist(),
        "ece": expected_calibration_error(probs, y_true),
    }
    # One-vs-rest AUC/AUPRC where a class is present.
    auc, auprc = {}, {}
    for i in range(4):
        yi = (y_true == i).astype(int)
        if yi.sum() > 0 and yi.sum() < len(yi):
            auc[CLASSES[i]] = float(roc_auc_score(yi, probs[:, i]))
            auprc[CLASSES[i]] = float(average_precision_score(yi, probs[:, i]))
    block["ovr_auc"] = auc
    block["ovr_auprc"] = auprc
    return block


def main():
    cfg = load_config()
    proc, artifacts = cfg["paths"]["processed_dir"], cfg["paths"]["artifacts_dir"]
    splits = pd.read_parquet(os.path.join(proc, "splits.parquet"))
    feat = np.load(os.path.join(proc, "drug_features.npz"), allow_pickle=True)
    ids = list(feat["ids"]); idx = {d: i for i, d in enumerate(ids)}
    X = feat["X"].astype(np.float32)
    splits = splits[splits.drugA_id.isin(idx) & splits.drugB_id.isin(idx)].reset_index(drop=True)
    splits["y"] = splits["label"].map(CLASS_TO_IDX)

    def mlp_feats(rows):
        return torch.tensor(np.stack([pair_features(X[idx[a]], X[idx[b]])
                                      for a, b in zip(rows.drugA_id, rows.drugB_id)]))

    # --- MLP ---
    ck = torch.load(os.path.join(artifacts, "model_mlp.pt"), map_location="cpu")
    mlp = PairMLP4(in_dim=ck["in_dim"], hidden=cfg["train"]["hidden"],
                   dropout=cfg["train"]["dropout"])
    mlp.load_state_dict(ck["model_state_dict"]); mlp.eval()
    val = splits[splits.split == "val"]
    with torch.no_grad():
        T_mlp = fit_temperature(mlp(mlp_feats(val)),
                                torch.tensor(val["y"].values, dtype=torch.long))

    # --- GNN ---
    ckg = torch.load(os.path.join(artifacts, "model_gnn.pt"), map_location="cpu")
    gnn = SageDDI(in_channels=ckg["in_channels"], hidden=cfg["train"]["gnn_hidden"],
                  dropout=cfg["train"]["dropout"])
    gnn.load_state_dict(ckg["model_state_dict"]); gnn.eval()
    x = torch.tensor(X)
    tr_pos = splits[(splits.split == "train") & (splits.label != "None")]
    ei = torch.tensor([[idx[a] for a in tr_pos.drugA_id] + [idx[b] for b in tr_pos.drugB_id],
                       [idx[b] for b in tr_pos.drugB_id] + [idx[a] for a in tr_pos.drugA_id]],
                      dtype=torch.long)
    def gnn_pidx(rows):
        return torch.tensor([[idx[a] for a in rows.drugA_id], [idx[b] for b in rows.drugB_id]],
                            dtype=torch.long)
    with torch.no_grad():
        T_gnn = fit_temperature(gnn(x, ei, gnn_pidx(val)),
                                torch.tensor(val["y"].values, dtype=torch.long))

    # --- Logistic-regression baseline on pair features ---
    tr = splits[splits.split == "train"]
    lr = LogisticRegression(max_iter=200, class_weight="balanced", multi_class="multinomial")
    lr.fit(mlp_feats(tr).numpy(), tr["y"].values)

    report = {"temperatures": {"mlp": T_mlp, "gnn": T_gnn}, "models": {}}
    for name in ["mlp", "gnn", "logreg", "majority"]:
        report["models"][name] = {}
    majority_class = int(np.bincount(tr["y"].values, minlength=4).argmax())

    for regime in ["test_S0", "test_S1", "test_S2"]:
        rows = splits[splits.split == regime]
        if len(rows) == 0:
            continue
        y = rows["y"].values
        with torch.no_grad():
            probs_mlp = F.softmax(mlp(mlp_feats(rows)) / T_mlp, dim=1).numpy()
            probs_gnn = F.softmax(gnn(x, ei, gnn_pidx(rows)) / T_gnn, dim=1).numpy()
        probs_lr = lr.predict_proba(mlp_feats(rows).numpy())
        probs_maj = np.zeros((len(rows), 4)); probs_maj[:, majority_class] = 1.0
        report["models"]["mlp"][regime] = _metrics_block(y, probs_mlp)
        report["models"]["gnn"][regime] = _metrics_block(y, probs_gnn)
        report["models"]["logreg"][regime] = _metrics_block(y, probs_lr)
        report["models"]["majority"][regime] = _metrics_block(y, probs_maj)

    json.dump(report, open(os.path.join(artifacts, "metrics.json"), "w"), indent=2)

    # Learning-curve plots (over/underfit diagnostic).
    for kind in ["mlp", "gnn"]:
        curve = json.load(open(os.path.join(artifacts, f"curve_{kind}.json")))
        ep = [c["epoch"] for c in curve]
        plt.figure()
        plt.plot(ep, [c["train_loss"] for c in curve], label="train_loss")
        plt.plot(ep, [c["val_loss"] for c in curve], label="val_loss")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title(f"{kind} learning curve")
        plt.savefig(os.path.join(artifacts, f"curve_{kind}.png")); plt.close()

    # Human-readable summary.
    lines = ["# DDI Severity — Evaluation Report\n",
             f"Temperatures: MLP={T_mlp:.3f}, GNN={T_gnn:.3f}\n",
             "## Macro-F1 by model and split\n",
             "| model | S0 | S1 | S2 |", "|---|---|---|---|"]
    for name in ["mlp", "gnn", "logreg", "majority"]:
        def cell(r):
            b = report["models"][name].get(r)
            return f"{b['macro_f1']:.3f}" if b else "-"
        lines.append(f"| {name} | {cell('test_S0')} | {cell('test_S1')} | {cell('test_S2')} |")
    open(os.path.join(artifacts, "report.md"), "w").write("\n".join(lines) + "\n")
    print("[done] wrote metrics.json, report.md, curve_*.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + stage**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_evaluate.py -v`
Expected: 2 passed.

Run: `/Users/zeyadzaher/grad-project/.venv/bin/python -m ml.pipeline.p06_evaluate`
Expected: a macro-F1 table printed for mlp/gnn/logreg/majority across S0/S1/S2. **Sanity gate:** both neural models should beat `majority` and `logreg` on S1/S2; if not, revisit features/negatives before exporting.

- [ ] **Step 5: Commit**

```bash
git add ml/pipeline/p06_evaluate.py ml/tests/test_evaluate.py
git commit -m "feat(ml): evaluation with calibration, baselines, learning curves, report"
```

---

## Task 7: Export the winner for serving

**Files:**
- Create: `ml/pipeline/p07_export.py`
- Test: `ml/tests/test_export.py`

Exports to `ml/artifacts/export/`: `model.pt` + `metadata.json`. Per the spec winner rule,
the deployable winner is the model with best **S2 macro-F1 among models that can score an
arbitrary SMILES pair** — that is always the MLP unless you explicitly enable GNN serving.
This task exports the MLP (the deployable model) with the chosen temperature, and records
the GNN's S2 result in metadata for the thesis comparison.

- [ ] **Step 1: Write failing test `ml/tests/test_export.py`**

```python
# ml/tests/test_export.py
import json
from ml.pipeline.p07_export import build_metadata


def test_metadata_has_required_serving_fields():
    md = build_metadata(fp_bits=2048, radius=2,
                        descriptors=["MolWt"], temperature=1.7,
                        in_dim=6168, gnn_s2_macro_f1=0.41, mlp_s2_macro_f1=0.46)
    assert md["classes"] == ["None", "Minor", "Moderate", "Major"]
    assert md["fp_bits"] == 2048 and md["temperature"] == 1.7
    assert md["in_dim"] == 6168
    assert md["comparison"]["gnn_s2_macro_f1"] == 0.41
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_export.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write `ml/pipeline/p07_export.py`**

```python
# ml/pipeline/p07_export.py
"""Stage 7: export the deployable MLP + metadata for inference-py."""
from __future__ import annotations
import os, json, shutil
import torch
from ml.config import load_config
from ml.common import CLASSES


def build_metadata(fp_bits, radius, descriptors, temperature, in_dim,
                   gnn_s2_macro_f1, mlp_s2_macro_f1):
    return {
        "classes": CLASSES,
        "severity_classes": ["Minor", "Moderate", "Major"],
        "fp_bits": fp_bits,
        "fp_radius": radius,
        "descriptors": descriptors,
        "temperature": temperature,
        "in_dim": in_dim,
        "comparison": {"gnn_s2_macro_f1": gnn_s2_macro_f1,
                       "mlp_s2_macro_f1": mlp_s2_macro_f1},
    }


def main():
    cfg = load_config()
    artifacts = cfg["paths"]["artifacts_dir"]
    export = os.path.join(artifacts, "export")
    os.makedirs(export, exist_ok=True)

    metrics = json.load(open(os.path.join(artifacts, "metrics.json")))
    ck = torch.load(os.path.join(artifacts, "model_mlp.pt"), map_location="cpu")

    def s2(name):
        b = metrics["models"][name].get("test_S2")
        return float(b["macro_f1"]) if b else None

    md = build_metadata(
        fp_bits=cfg["features"]["fp_bits"], radius=cfg["features"]["fp_radius"],
        descriptors=cfg["features"]["descriptors"],
        temperature=metrics["temperatures"]["mlp"], in_dim=ck["in_dim"],
        gnn_s2_macro_f1=s2("gnn"), mlp_s2_macro_f1=s2("mlp"),
    )
    torch.save({"model_state_dict": ck["model_state_dict"], "in_dim": ck["in_dim"],
                "hidden": cfg["train"]["hidden"]}, os.path.join(export, "model.pt"))
    json.dump(md, open(os.path.join(export, "metadata.json"), "w"), indent=2)
    print(f"[done] exported MLP -> {export} (S2 macro-F1 mlp={md['comparison']['mlp_s2_macro_f1']}, "
          f"gnn={md['comparison']['gnn_s2_macro_f1']})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + stage**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests/test_export.py -v`
Expected: 1 passed.

Run: `/Users/zeyadzaher/grad-project/.venv/bin/python -m ml.pipeline.p07_export`
Expected: `ml/artifacts/export/model.pt` and `metadata.json` written.

- [ ] **Step 5: Commit**

```bash
git add ml/pipeline/p07_export.py ml/tests/test_export.py
git commit -m "feat(ml): export deployable model + serving metadata"
```

---

## Task 8: Orchestrator

**Files:**
- Create: `ml/run.py`

- [ ] **Step 1: Write `ml/run.py`**

```python
# ml/run.py
"""Run all pipeline stages in order."""
from ml.pipeline import (p01_download, p02_build_labels, p03_features,
                         p04_splits, p05_train, p06_evaluate, p07_export)


def main():
    for stage in [p01_download, p02_build_labels, p03_features,
                  p04_splits, p05_train, p06_evaluate, p07_export]:
        print(f"\n===== {stage.__name__} =====")
        stage.main()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the whole pipeline end-to-end**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && /Users/zeyadzaher/grad-project/.venv/bin/python -m ml.run`
Expected: stages run in sequence, finishing with the export line. `ml/artifacts/export/` is populated.

- [ ] **Step 3: Commit**

```bash
git add ml/run.py
git commit -m "feat(ml): end-to-end pipeline orchestrator"
```

---

## Task 9: Serve the measured model in `inference-py`

**Files:**
- Modify: `inference-py/app/model.py` (replace severity logic + loader)
- Modify: `inference-py/app/main.py` (response fields)
- Copy: `ml/artifacts/export/model.pt` + `metadata.json` → `inference-py/app/models/`
- Test: `inference-py/tests/test_model.py`

- [ ] **Step 1: Copy the exported artifact into the service**

```bash
cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction
cp ml/artifacts/export/model.pt inference-py/app/models/severity_model.pt
cp ml/artifacts/export/metadata.json inference-py/app/models/severity_metadata.json
```

- [ ] **Step 2: Write the failing test `inference-py/tests/test_model.py`**

```python
# inference-py/tests/test_model.py
import os
import pytest
from app.model import DDIModel

MODEL = os.path.join(os.path.dirname(__file__), "..", "app", "models", "severity_model.pt")
META = os.path.join(os.path.dirname(__file__), "..", "app", "models", "severity_metadata.json")


@pytest.mark.skipif(not os.path.exists(MODEL), reason="export model not present")
def test_predict_returns_calibrated_severity_fields():
    m = DDIModel(MODEL, META, device="cpu")
    out = m.predict_with_severity("CCO", "CCN")  # ethanol-like vs ethylamine-like
    assert set(out["probabilities"]) == {"None", "Minor", "Moderate", "Major"}
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-4
    assert out["severity"] in {"Minor", "Moderate", "Major"}
    assert 0.0 <= out["interactionProbability"] <= 1.0


def test_predict_is_symmetric():
    m = DDIModel(MODEL, META, device="cpu")
    a = m.predict_with_severity("CCO", "CCN")
    b = m.predict_with_severity("CCN", "CCO")
    assert abs(a["interactionProbability"] - b["interactionProbability"]) < 1e-5
```

- [ ] **Step 3: Run to confirm failure**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction/inference-py && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest tests/test_model.py -v`
Expected: FAIL — current `DDIModel.__init__` takes no metadata path and has no `probabilities` output.

- [ ] **Step 4: Rewrite `inference-py/app/model.py`**

Replace the entire file with:

```python
# inference-py/app/model.py
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Crippen, rdMolDescriptors

CLASSES = ["None", "Minor", "Moderate", "Major"]
SEVERITY_CLASSES = ["Minor", "Moderate", "Major"]

_DESCRIPTORS = [
    Descriptors.MolWt, Crippen.MolLogP, Descriptors.NumHDonors,
    Descriptors.NumHAcceptors, Descriptors.TPSA, Descriptors.NumRotatableBonds,
    rdMolDescriptors.CalcNumAromaticRings, rdMolDescriptors.CalcFractionCSP3,
]


class PairMLP4(nn.Module):
    def __init__(self, in_dim, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 4),
        )

    def forward(self, x):
        return self.net(x)


def _drug_vector(smiles, fp_bits, radius):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = np.zeros((fp_bits,), dtype=np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_bits)
    DataStructs.ConvertToNumpyArray(bv, fp)
    desc = np.zeros((len(_DESCRIPTORS),), dtype=np.float32)
    for i, fn in enumerate(_DESCRIPTORS):
        try:
            desc[i] = float(fn(mol))
        except Exception:
            desc[i] = 0.0
    return np.concatenate([fp, desc]).astype(np.float32)


def _pair_features(va, vb):
    return np.concatenate([va + vb, np.abs(va - vb), va * vb]).astype(np.float32)


class DDIModel:
    def __init__(self, model_path: str, metadata_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        with open(metadata_path) as f:
            self.meta = json.load(f)
        self.fp_bits = int(self.meta["fp_bits"])
        self.radius = int(self.meta["fp_radius"])
        self.temperature = float(self.meta.get("temperature", 1.0))
        ck = torch.load(model_path, map_location=self.device)
        self.model = PairMLP4(in_dim=int(ck["in_dim"]), hidden=int(ck.get("hidden", 512)))
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_with_severity(self, smiles_a: str, smiles_b: str) -> dict:
        va = _drug_vector(smiles_a, self.fp_bits, self.radius)
        vb = _drug_vector(smiles_b, self.fp_bits, self.radius)
        feats = torch.tensor(_pair_features(va, vb), device=self.device).unsqueeze(0)
        probs = F.softmax(self.model(feats) / self.temperature, dim=1).squeeze(0)
        prob_map = {CLASSES[i]: float(probs[i]) for i in range(4)}
        interaction = 1.0 - prob_map["None"]
        sev_idx = int(torch.tensor([probs[CLASSES.index(c)] for c in SEVERITY_CLASSES]).argmax())
        return {
            "interactionProbability": interaction,
            "severity": SEVERITY_CLASSES[sev_idx],
            "probabilities": prob_map,
        }
```

- [ ] **Step 5: Update `inference-py/app/main.py`**

Replace the file with:

```python
# inference-py/app/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import DDIModel

app = FastAPI(title="DDI Inference Service")


class PredictReq(BaseModel):
    smilesA: str
    smilesB: str


MODEL_PATH = os.getenv("MODEL_PATH", "/models/severity_model.pt")
META_PATH = os.getenv("META_PATH", "/models/severity_metadata.json")
DEVICE = os.getenv("DEVICE", "cpu")

ddi = None


@app.on_event("startup")
def startup():
    global ddi
    ddi = DDIModel(MODEL_PATH, META_PATH, device=DEVICE)


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

- [ ] **Step 6: Run the serving tests**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction/inference-py && mkdir -p tests && touch tests/__init__.py && /Users/zeyadzaher/grad-project/.venv/bin/python -m pytest tests/test_model.py -v`
Expected: 2 passed.

- [ ] **Step 7: Update Docker model paths**

Check `inference-py/Dockerfile` and `docker-compose.yml` for `MODEL_PATH=/models/best_model.pt`. Update the env to the new files and ensure both `severity_model.pt` and `severity_metadata.json` are copied to `/models`. Set:
```
MODEL_PATH=/models/severity_model.pt
META_PATH=/models/severity_metadata.json
```

- [ ] **Step 8: Commit**

```bash
cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction
git add inference-py/app/model.py inference-py/app/main.py inference-py/tests/ \
        inference-py/app/models/severity_model.pt inference-py/app/models/severity_metadata.json \
        inference-py/Dockerfile docker-compose.yml
git commit -m "feat(inference): serve calibrated 4-class severity model; drop fake thresholds"
```

---

## Task 10: Pass new fields through backend + mobile

**Files:**
- Modify: `backend-go/inference.go`
- Modify: `mobile-rn/App.js`

- [ ] **Step 1: Inspect the current response struct in `backend-go/inference.go`**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction && grep -n "probability\|severity\|Probability\|Severity\|struct" backend-go/inference.go`
Expected: shows the struct that decodes the inference response and the struct returned to mobile.

- [ ] **Step 2: Add the new fields to the inference-response and API-response structs**

In `backend-go/inference.go`, the struct decoding the Python response must include:
```go
type inferenceResp struct {
    InteractionProbability float64            `json:"interactionProbability"`
    Severity               string             `json:"severity"`
    Probabilities          map[string]float64 `json:"probabilities"`
}
```
And the struct returned from `/v1/ddi/predict` must carry these through (rename existing
`Probability` usage to `InteractionProbability`, keep `Severity`, add `Probabilities`).
Update the mapping code that builds the API response from `inferenceResp` accordingly.

- [ ] **Step 3: Build the backend to verify it compiles**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction/backend-go && go build ./...`
Expected: no errors.

- [ ] **Step 4: Update mobile to display real severity + class probabilities**

In `mobile-rn/App.js`, find where the prediction result renders `severity`/`probability`.
Use `interactionProbability` for the probability bar and `severity` for the label; optionally
render the per-class `probabilities` map as a small breakdown. (No logic change — display only.)

- [ ] **Step 5: Verify mobile bundles**

Run: `cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction/mobile-rn && npx expo export --platform web 2>&1 | tail -5`
Expected: bundle completes without referencing removed fields. (If `expo export` is heavy, a `grep -n "interactionProbability\|severity" App.js` confirming the new field names is acceptable.)

- [ ] **Step 6: Commit**

```bash
cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction
git add backend-go/inference.go mobile-rn/App.js
git commit -m "feat(api): pass calibrated severity + class probabilities to mobile"
```

---

## Task 11: End-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Full pipeline from scratch**

```bash
cd /Users/zeyadzaher/grad-project-fcds-2025-ddi-prediction
rm -rf ml/data/processed/* ml/artifacts/*
/Users/zeyadzaher/grad-project/.venv/bin/python -m ml.run
```
Expected: completes through export; `ml/artifacts/report.md` shows mlp/gnn beating baselines on S1/S2.

- [ ] **Step 2: Run the whole test suite**

```bash
/Users/zeyadzaher/grad-project/.venv/bin/python -m pytest ml/tests inference-py/tests -v
```
Expected: all green.

- [ ] **Step 3: Boot the stack and hit the API**

```bash
docker compose up --build -d
sleep 20
curl -s -X POST localhost:8080/v1/ddi/predict -H 'Content-Type: application/json' \
  -d '{"drugAName":"warfarin","drugBName":"aspirin"}' | python3 -m json.tool
docker compose down
```
Expected: JSON includes `interactionProbability`, `severity` (a real Minor/Moderate/Major from the model), and a `probabilities` map summing to ~1.0.

- [ ] **Step 4: Review the report and commit any final notes**

Open `ml/artifacts/report.md`. Confirm the **train→S2 gap** is reported via learning curves and the S2 macro-F1 beats `majority`/`logreg`. Record the final numbers in `ml/README.md` "Results" section.

```bash
git add ml/README.md
git commit -m "docs(ml): record final S0/S1/S2 results"
```

---

## Self-Review notes (addressed)

- **Spec §2 (data):** Tasks 1–2 download DDInter, join by normalized name, build the
  interaction mask from DrugBank edges, sample hard/random negatives → `pairs.parquet`. ✓
- **Spec §3 (anti-overfitting):** Task 4 produces drug-disjoint S0/S1/S2 with a leakage
  assertion; Task 6 emits learning curves + the train→S2 story. ✓
- **Spec §4 (both models + baselines):** Task 5 trains MLP + GNN with class weights/early
  stop; Task 6 adds majority + logreg baselines. ✓
- **Spec §5 (metrics + calibration):** Task 6 computes macro-F1, per-class P/R/F1, confusion,
  OvR AUC/AUPRC, ECE, temperature scaling, reliability via curves. ✓
- **Spec §6 (serving):** Task 9 rewires `inference-py` to the 4-class calibrated model and
  deletes the fake thresholds; Task 10 passes fields through backend + mobile. ✓
- **Spec §7 (structure/repro):** Tasks 0 + 8 give the staged, seeded, single-command pipeline. ✓
- **Winner rule:** Task 7 exports the deployable MLP and records the GNN comparison, matching
  the spec's "best S2 among servable models" rule. ✓
- **Type consistency:** `CLASSES` order, `pair_features` (sum/|diff|/prod), `in_dim`,
  `temperature`, and metadata keys are identical across `ml/features.py`, `ml/models.py`,
  `p05`–`p07`, and `inference-py/app/model.py`. ✓
```
