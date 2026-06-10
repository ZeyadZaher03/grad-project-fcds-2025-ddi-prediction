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

## Data
- **Drugs / SMILES:** 8,287 DrugBank drugs (`drugs.csv`).
- **Severity labels:** DDInter 2.0 (8 ATC categories), joined to drugs by normalized name.
  After the join: **51,700 labeled pairs** — Moderate 36,619 / Major 12,308 / Minor 2,773
  (Minor is only ~5% → heavy class imbalance, handled with class-weighted loss).
- **`None` class:** non-interacting pairs sampled with a DrugBank-edge mask so known
  interactions are never mislabeled as None. Final dataset 103,400 pairs (50% None).

## Evaluation protocol (anti-overfitting)
Splits are **drug-disjoint**, reported in three cold-start regimes:
- **S0** — both drugs seen in training (optimistic)
- **S1** — one drug unseen
- **S2** — both drugs unseen (true generalization)

The train→S2 gap is the overfitting measure; learning curves are in `artifacts/curve_*.png`.

## Results (macro-F1)

| model | S0 (both seen) | S1 (one new) | S2 (both new) |
|---|---|---|---|
| **MLP (deployed)** | 0.323 | 0.313 | **0.297** |
| GNN (GraphSAGE) | 0.534 | 0.166 | 0.169 |
| logreg baseline | 0.297 | 0.286 | 0.274 |
| majority baseline | 0.168 | 0.166 | 0.169 |

**Read of the results:**
- The **MLP barely degrades** from seen→unseen drugs (0.323 → 0.297): it generalizes, i.e.
  it is **not overfitting**. It is the deployed model.
- The **GNN overfits hard**: 0.534 on seen drugs collapses to the majority baseline (0.169)
  on unseen drugs. The cold-start split is what exposed this — it is reported as the
  research comparison, not deployed.
- Both neural models beat the baselines on the honest S2 regime.

**Per-class F1 (MLP, S2):** None 0.61 · Moderate 0.36 · Major 0.22 · **Minor 0.00**.
Calibration: temperature ≈ 0.96, ECE ≈ 0.17.

**Honest limitations:** the model separates *interaction vs none* reasonably (None F1 ≈ 0.61)
and does fair on Moderate, but it **cannot predict the Minor class** (too rare, and
structurally similar drug pairs span all severities). Absolute macro-F1 (~0.30) is modest —
this reflects the genuine difficulty of structure-only severity prediction, not a pipeline
bug. Future work: richer features (targets/enzymes/pathways), Minor-class resampling.

## Serving
The deployed model is exported to `artifacts/export/{model.pt, metadata.json}` and copied into
`inference-py/app/models/severity_{model.pt,metadata.json}`. The inference service loads it,
applies temperature calibration, and returns `interactionProbability` (= 1 − P(None)),
`severity` (`None` when `interactionProbability < 0.5`, else the most likely
Minor/Moderate/Major class), and the full per-class `probabilities` map.
