# DDI Severity Prediction — ML Pipeline Redesign

**Date:** 2026-06-10
**Status:** Approved design, pending implementation plan
**Author:** Zeyad Zaher (with Claude)

## Problem

The deployed app predicts drug–drug interactions but has three fundamental flaws:

1. **No measurable model.** The app serves a fingerprint `PairMLP` (`inference-py`) that
   has **no metrics whatsoever**. The only recorded result — `test_auc 0.863` — belongs to
   a *different* model (the GAT in the separate `grad-project` folder) that the app never
   runs. "Are we overfitting?" is currently unanswerable.

2. **Data starvation + leakage.** The GAT was trained on `data/ddi.csv` (2,266 positive
   edges) — roughly 0.2% of the ~1M+ interactions available in
   `learn/drug_interaction_edges.csv`. It was evaluated with `RandomLinkSplit`, which hides
   *edges* but keeps both drugs (and their neighbors) in the graph — a leaky setup that
   inflates AUC. At inference, every candidate scores ~0.98 (`artifacts/infer_summary.json`),
   a classic sign of trivial negatives + leakage.

3. **Severity is fake.** No severity labels exist in any local dataset (`ddi.csv` and the
   large `labeled_*` files are all binary `label=1`). The app's `minor`/`moderate`/`major`
   output is just the interaction probability run through hardcoded thresholds
   (`inference-py/app/model.py:79-85`).

## Goal

Build one clean, reproducible, **honestly-evaluated** ML pipeline that predicts a real,
supervised **severity class** for any drug pair, and serve exactly the model we measure.

## Decisions (locked)

| Decision | Choice |
| --- | --- |
| Prediction target | **4-class**: `None` / `Minor` / `Moderate` / `Major` |
| Severity label source | **DDInter 2.0** (free, curated, ~0.24M labeled pairs, 1,833 drugs) |
| Model approach | **Both** an inductive fingerprint MLP *and* a GNN, head-to-head; deploy the winner |
| Pipeline location | Inside the app repo, new `ml/` directory |

## Design

### 1. Problem framing

A single softmax classifier over an unordered drug pair, 4 classes:
`None`, `Minor`, `Moderate`, `Major`.

This maps directly onto what the app already displays:
- `P(interaction) = 1 − P(None)` → the interaction probability
- `severity = argmax(Minor, Moderate, Major)` → a real, learned severity

The model must be **symmetric**: `predict(A, B) == predict(B, A)`. Enforced via symmetric
pair features (see §4) and by adding both orderings during training.

### 2. Data pipeline

Sources and roles:

| Source | File | Role |
| --- | --- | --- |
| Drugs + SMILES | `grad-project/data/drugs.csv` (8,287) | Node set; provides SMILES for features |
| Severity positives | DDInter 2.0 CSVs (ATC categories A,B,D,H,L,P,R,V) | `(drugA, drugB, severity)` labels |
| Interaction mask | `grad-project/learn/drug_interaction_edges.csv` (~1M) | Excludes known interactions from the `None` pool |

Steps:

1. **Download** DDInter category CSVs into `ml/data/raw/ddinter/`.
2. **Join** DDInter rows to `drugs.csv` by normalized drug name (lowercase, strip salts /
   punctuation). Records the join hit-rate; pairs where either drug fails to map are dropped.
   DDInter `Unknown` severity rows are dropped (kept out of all classes).
3. **Build the interaction mask**: the union of DDInter pairs and DrugBank
   `drug_interaction_edges.csv` pairs = "these definitely interact."
4. **Sample `None` negatives** from drug pairs absent from the mask:
   - a random share, plus
   - **hard negatives**: pairs of structurally similar drugs (high Tanimoto on Morgan FP)
     that are *not* in the mask, so the task isn't trivially separable.
   - Negative count controlled by config (default: balance so `None` ≈ largest severity
     class, not the full combinatorial space).
5. **Output** a single canonical `pairs.parquet`: `drugA_id, drugB_id, smilesA, smilesB,
   label` with both orderings.

Retire `ddi.csv`, `labeled_drug_interactions*.csv`, and `merged_labeled_interactions*.csv`
(redundant or explosion artifacts; the last two are 8GB each).

### 3. Evaluation protocol (the anti-overfitting core)

Splits are **drug-disjoint**, not edge-random. We partition the *drug set* into
train/val/test groups, then assign each pair to a regime by how many of its drugs are unseen:

- **S0** — both drugs seen in training (optimistic, reported for context)
- **S1** — exactly one drug unseen (realistic deployment)
- **S2** — both drugs unseen (true cold-start; the honest generalization number)

The **train→S2 performance gap is the overfitting measure.** Additionally:

- **Learning curves** (train vs val loss/macro-F1 across epochs) to read under/overfit directly.
- Fixed seed; splits saved to disk so every model is evaluated on identical pairs.
- No tuning on the test regimes — hyperparameters chosen on validation only.

### 4. Models (head-to-head)

All models output 4-class logits and are trained with class-weighted cross-entropy
(severity is imbalanced), dropout, weight decay, and early stopping on **val macro-F1**.

- **Model A — Inductive fingerprint MLP (deployable).**
  Per drug: Morgan fingerprint (radius 2, 2048 bits) + selected RDKit descriptors.
  Symmetric pair features: `[fpA+fpB, |fpA−fpB|, fpA·fpB, descA+descB, |descA−descB|]`.
  MLP head → 4 logits. Works on any SMILES → the app serves exactly this.

- **Model B — GNN (GraphSAGE, inductive).**
  Node features = same per-drug vectors; edges = DrugBank interaction graph. Inductive so
  unseen drugs can be embedded from features + neighborhood. Pair scored by concatenating
  the two node embeddings into a 4-class head.

- **Baselines** (prove we beat trivial): majority-class and logistic regression on pair
  fingerprints (reuse `grad-project/learn/logreg_ddi_model.pkl` approach).

### 5. Metrics & calibration

Reported for **every model × every split regime (S0/S1/S2)**:

- Macro-F1 (primary), per-class precision / recall / F1
- Confusion matrix
- One-vs-rest ROC-AUC and PR-AUC (AUPRC) per class
- **Calibration**: reliability diagram + Expected Calibration Error (ECE), then
  **temperature scaling** fit on validation (fixes the "everything scores 0.98" collapse)

A single `report.md` / `metrics.json` per run summarizes all of the above plus learning curves.

### 6. Serving

0. **Winner selection rule:** the deployed model is the one with the best **S2 (cold-start)
   macro-F1** *among models that can score an arbitrary SMILES pair at inference*. Model A
   always qualifies; Model B qualifies only if its inductive embedding of unseen drugs works
   in practice. If the GNN wins on metrics but cannot reliably serve novel pairs, Model A is
   deployed and the GNN result is reported as the research comparison.
1. The winning model is exported as one artifact (`model.pt`) + `metadata.json`
   (class order, feature config, temperature, fp size, descriptor list).
2. `inference-py/app/model.py`:
   - load the 4-class model + metadata,
   - return calibrated `probabilities` (per class), `severity` (argmax of severity classes),
     and `interactionProbability` (`1 − P(None)`),
   - **delete the hardcoded threshold block (`model.py:79-85`).**
3. `inference-py/app/main.py` response schema gains the per-class probabilities.
4. `backend-go` passes the new fields through; `mobile-rn` displays the real severity.
   (Backend/mobile changes are pass-through only — no logic changes.)

### 7. Pipeline structure & reproducibility

New `ml/` directory in the app repo, ordered idempotent stages, config-driven, seeded:

```
ml/
  config.yaml              # paths, seeds, fp size, neg ratio, split fractions, hyperparams
  data/raw/                # downloaded DDInter + copied drugs.csv, edges
  data/processed/          # pairs.parquet, splits/, features/
  pipeline/
    01_download.py         # fetch DDInter category CSVs
    02_build_labels.py     # join + mask + negative sampling -> pairs.parquet
    03_features.py         # SMILES -> fingerprints + descriptors
    04_splits.py           # drug-disjoint S0/S1/S2 splits
    05_train.py            # trains Model A and Model B from config
    06_evaluate.py         # metrics + confusion + calibration + learning curves
    07_export.py           # export winner artifact + metadata for inference-py
  artifacts/               # trained models, metrics.json, report.md, plots
  README.md                # how to reproduce end-to-end
```

Each stage reads from and writes to disk so it can be re-run independently. A top-level
`make all` (or `python -m ml.run`) runs the full chain.

## Out of scope (YAGNI)

- Interaction *type/mechanism* prediction (a possible future extension).
- Retraining infrastructure / scheduled pipelines.
- Changes to PubChem name→SMILES resolution in `backend-go`.
- Mobile UI redesign beyond surfacing the new severity fields.

## Success criteria

1. One reproducible command regenerates data → models → metrics from scratch.
2. Honest **S2 (cold-start)** macro-F1 is reported and clearly beats the majority/logreg
   baselines, with a quantified train→S2 gap and learning curves showing controlled
   over/underfit.
3. The app serves the exact model that was measured, returning calibrated severity — the
   fake threshold code is gone.
4. Every model × split × metric is captured in a single committed report for the thesis.
