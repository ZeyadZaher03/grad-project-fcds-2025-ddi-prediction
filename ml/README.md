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

## Serving (iteration 1 — superseded by iteration 2 below)
The iteration-1 model was a single calibrated 4-class MLP. It has been superseded by the
two-stage ChemBERTa model below, which is what the service now serves.

---

# Iteration 2 — two-stage + biological features + ChemBERTa

Iteration 1 plateaued at S2 macro-F1 0.297 and could not predict the Minor class at all.
Iteration 2 attacked four levers and measured each with an ablation on the **same
drug-disjoint S0/S1/S2 splits**.

## What changed
1. **Two-stage model.** Stage 1 = binary "do they interact?" trained on a 120k subsample of
   the full ~1.8M DrugBank interaction edges. Stage 2 = 3-class severity (Minor/Moderate/Major)
   trained on DDInter's labeled interactions only — so Minor competes only against the other
   severities, not the dominant None class.
2. **Biological mechanistic-overlap features.** Shared targets / enzymes / transporters /
   carriers (count + Jaccard) per drug pair, from DrugBank, joined by DrugBank ID.
3. **Structure representation.** Count-Morgan + ~38 RDKit descriptors (`light`), and
   **ChemBERTa-77M embeddings** (`chemberta`) as a head-to-head option.
4. **More data.** The 1.8M binary edges feed stage 1 (severity labels remain capped at DDInter).

## Ablation — end-to-end S2 macro-F1 (cold-start, both drugs unseen)

| config | S2 macro-F1 | S2 accuracy | Minor F1 | stage-1 AUC |
|---|---|---|---|---|
| iteration-1 baseline (single 4-class) | 0.297 | ~46% | 0.00 | — |
| struct_light (no bio) | 0.239 | 29% | 0.06 | 0.59 |
| struct_light + bio | 0.277 | 43% | 0.00 | 0.64 |
| **chemberta + bio (deployed)** | **0.485** | **69%** | **0.23** | **0.91** |

Per-class F1 for the deployed `chemberta+bio` model:

| regime | None | Minor | Moderate | Major | macro-F1 | accuracy |
|---|---|---|---|---|---|---|
| S0 (both seen) | 0.94 | 0.59 | 0.81 | 0.67 | 0.754 | 86% |
| S1 (one new) | 0.87 | 0.38 | 0.69 | 0.46 | 0.600 | 75% |
| S2 (both new) | 0.83 | 0.23 | 0.60 | 0.29 | 0.485 | 69% |

## What each lever actually did (honest)
- **ChemBERTa embeddings were the dominant win** — nearly doubled S2 macro-F1 (0.297 → 0.485)
  and lifted cold-start interaction-detection AUC from ~0.6 to **0.91**. (We had predicted this
  would be the *smallest* lever; the data said otherwise.)
- **The two-stage design + good features finally cracked Minor** — S2 Minor F1 went 0.00 → 0.23.
  Note it only worked *with* ChemBERTa; with the light features Minor stayed at 0.
- **Biological overlap features add a real, modest lift** — `struct_light` 0.239 → `+bio` 0.277
  (+0.04), consistent across regimes.
- **The lightweight structural upgrade alone underperformed** (0.239, below the 0.297 baseline) —
  cheap descriptors were not enough; the pretrained embeddings did the heavy lifting.
- We also tried lowering the stage-2 learning rate and adding label smoothing to rescue Minor on
  the light features; it did not help, confirming the limitation was the features, not tuning.

## Honest limitations (iteration 2)
- S2 accuracy is 69% and Minor F1 is 0.23 — much improved, but Minor is still the weakest class
  and the model is a decision-support signal, not a clinical authority.
- The S0→S2 gap (0.754 → 0.485) is real and expected; it is the honest cost of unseen drugs.
- The deployed model now requires the `transformers` library and a ChemBERTa download in the
  inference container (a heavier image), which is the price of the accuracy gain.

## Serving (iteration 2)
The deployed model is exported to `artifacts/export_v2/{model_stage1.pt, model_stage2.pt,
metadata.json, bio_lookup.json}` and copied into `inference-py/app/models/`. The service:
1. embeds each SMILES with ChemBERTa (same code path as training — verified to match),
2. attaches biological overlap features via the canonical-SMILES → entity-set `bio_lookup`
   (zeros for drugs not in DrugBank),
3. returns `interactionProbability` (stage-1 sigmoid), `severity` (`None` when
   `interactionProbability < 0.5`, else argmax of stage-2 over Minor/Moderate/Major), and the
   full per-class `probabilities`.

Reproduce the ablation end-to-end with `python -m ml.run_ablation`.
