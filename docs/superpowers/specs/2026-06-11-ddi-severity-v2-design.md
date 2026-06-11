# DDI Severity — Iteration 2 Design

**Date:** 2026-06-11
**Status:** Approved design, pending implementation plan
**Author:** Zeyad Zaher (with Claude)
**Builds on:** `2026-06-10-ddi-severity-ml-pipeline-design.md` (iteration 1)

## Motivation

Iteration 1 produced a measurable, non-overfitting model, but with modest absolute numbers
on the honest cold-start (S2) regime:

- End-to-end severity accuracy ≈ 46%; macro-F1 = 0.297.
- **Minor class is unlearnable** (F1 = 0.00, AUC ≈ 0.5) — too rare (5% of positives) and it
  competes directly with the dominant None class in a single 4-class head.
- The model uses **structure only** (Morgan fingerprints + 8 descriptors) — it cannot see the
  *mechanism* of an interaction (shared metabolizing enzymes, shared targets).

This iteration attacks all four diagnosed levers.

## Decisions (locked)

| # | Lever | Decision |
| --- | --- | --- |
| 2+4 | Minor class + more data | **Two-stage model**: a binary interaction detector trained on the full ~1M DrugBank edges, then a 3-class severity classifier trained on DDInter's 51.7k interactions |
| 1 | Biological features | **Pairwise mechanistic-overlap features** from targets/enzymes/transporters/carriers |
| 3 | Structure representation | **Lightweight upgrade** (count-Morgan + ~50 RDKit descriptors) as default, **ChemBERTa embeddings as an optional A/B** behind a feature flag |
| — | Deliverable | An **ablation table** quantifying each lever's contribution, on the same drug-disjoint S0/S1/S2 protocol |

## Design

### 1. Two-stage model (backbone — levers #2 + #4)

**Stage 1 — Interaction detector (binary).**
- Positives: the full DrugBank interaction edge list (`drug_interaction_edges.csv`, ~1M pairs),
  filtered to drugs present in `drugs.csv`.
- Negatives: sampled non-interacting pairs (random + structurally-hard), excluded via the
  same interaction mask as iteration 1.
- Output: `P(interact)` (sigmoid).

**Stage 2 — Severity classifier (3-class: Minor / Moderate / Major).**
- Trained only on DDInter's 51.7k labeled interactions (no None class).
- Minor now competes only against Moderate/Major, with class-weighted loss, on the subset
  where it actually occurs.
- Output: softmax over {Minor, Moderate, Major}.

**Serving (revised inference-py):**
- `interactionProbability = P(interact)` from stage 1.
- If `interactionProbability < 0.5` → `severity = "None"`.
- Else `severity = argmax(stage-2)`.
- `probabilities` returned as the four-class distribution
  `{None: 1−P(interact), Minor: P(interact)·p_minor, Moderate: …, Major: …}`.

The iteration-1 single 4-class MLP (S2 macro-F1 = 0.297) is retained as a **reported baseline**.

### 2. Biological features (lever #1)

Source: `merged_output.csv`, joined to `drugs.csv` by **DrugBank ID** (verified 8,287/8,287
overlap). Relevant columns hold space-separated DrugBank entity IDs (`BE…`):
`targets` (5,177 drugs), `enzymes` (1,000), `transporters` (596), `carriers` (352).

Build `drug → {targets}, {enzymes}, {transporters}, {carriers}` sets. For each drug pair,
compute **order-invariant overlap features** per entity type:
- shared count `|A ∩ B|`
- Jaccard `|A ∩ B| / |A ∪ B|` (0 if both empty)
- each drug's entity count (added symmetrically: sum and abs-diff)

≈ 4 types × (shared + Jaccard) + count summaries ≈ 12–16 features. Missing data → zeros.
These features feed **both** stages.

Rationale: overlap features encode the actual interaction *mechanism* (e.g. two drugs sharing
CYP3A4) and generalize to unseen drugs because they describe overlap, not drug identity — the
right inductive bias for cold-start. Per-drug multi-hot identity vectors are deliberately
excluded (high-dimensional, memorization-prone on S2).

### 3. Structure representation (lever #3)

A feature-config switch selects the structural block:
- **`light` (default):** count-based Morgan fingerprint (`GetHashedMorganFingerprint`, counts
  not bits, 2048 dims) + an expanded curated RDKit descriptor set (~50: rings, formal charge,
  fraction sp3, TPSA, complexity, H-bond donors/acceptors, etc.).
- **`chemberta` (optional A/B):** mean-pooled `DeepChem/ChemBERTa-77M-MLM` embeddings
  (384-dim) per molecule, via HuggingFace `transformers`. Loaded lazily; only required when
  this config is selected so the core pipeline has no transformers dependency.

Pair-level structural features remain the symmetric `[vA+vB, |vA−vB|, vA·vB]` construction.

### 4. Final pair feature vector

`pair_features = [structural pair block] ++ [biological overlap block]`, symmetric by
construction. The same vector feeds stage 1 and stage 2 (each has its own trained head).

### 5. Evaluation (same protocol, extended to an ablation)

Same **drug-disjoint S0/S1/S2** splits (drug-level holdout; S2 = both unseen). Reported per
regime:
- **Stage 1:** ROC-AUC, PR-AUC, F1 for interaction detection.
- **Stage 2:** 3-class macro-F1 + per-class P/R/F1 on true interactions (does Minor move off 0?).
- **End-to-end 4-class macro-F1 + accuracy** — directly comparable to iteration 1's 0.297,
  plus the 4-class confusion matrix.
- Calibration (temperature + ECE) for both stages.

**Ablation table** — end-to-end S2 macro-F1 for each feature configuration, run as a sweep:

| config | structural | biological | embeddings |
| --- | --- | --- | --- |
| `struct_light` | light | — | — |
| `struct_light + bio` | light | overlap | — |
| `struct_chemberta + bio` | chemberta | overlap | chemberta |

Plus the iteration-1 baseline row. This is the centerpiece deliverable: it shows which lever
paid off and honestly flags any that did not.

### 6. Pipeline structure

Extends the existing `ml/` package (reuses config, splits, common):

```
ml/
  config.yaml              # + feature.structural: light|chemberta; + bio toggle; + stage hyperparams
  features.py              # MODIFY: light structural block (count-Morgan + ~50 descriptors)
  bio_features.py          # NEW: parse entity sets, pairwise overlap features
  embeddings.py            # NEW: ChemBERTa embedding (lazy, optional)
  models.py                # MODIFY/ADD: InteractionMLP (binary) + SeverityMLP (3-class)
  pipeline/
    p02b_interaction_pairs.py   # NEW: build the ~1M binary interaction dataset (+ negatives)
    p03_features.py             # MODIFY: structural + bio per-drug/pair features per config
    p05_train.py                # MODIFY: train stage 1 (binary) and stage 2 (severity)
    p06_evaluate.py             # MODIFY: two-stage + end-to-end + ablation table
    p07_export.py               # MODIFY: export both stage models + metadata
  run_ablation.py          # NEW: sweep feature configs, emit ablation table
inference-py/app/model.py  # MODIFY: load two models, two-stage serving
```

The two stages are independent, separately testable units with a shared feature interface.

## Out of scope (YAGNI)

- Per-drug multi-hot identity features (memorization risk on cold-start).
- Pathways column (empty in the data).
- Mechanism-of-action / interaction free-text NLP (separate, larger effort).
- GNN variants (iteration 1 showed they overfit cold-start; not revisited here).
- Retraining infrastructure / scheduling.

## Success criteria

1. One reproducible command regenerates the two-stage models + the ablation table.
2. The ablation table quantifies each lever's S2 contribution against the iteration-1 baseline,
   with any non-improving lever reported honestly.
3. **Minor-class S2 F1 is measurably above 0.00** (the headline failure of iteration 1), or, if
   it still cannot be learned, that is shown explicitly with the two-stage isolation.
4. End-to-end S2 macro-F1 and accuracy are reported and compared to iteration 1 (0.297 / 46%).
5. The app serves the two-stage model; serving stays symmetric and calibrated.
