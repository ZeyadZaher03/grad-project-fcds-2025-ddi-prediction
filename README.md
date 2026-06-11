# DDI Prediction — Drug–Drug Interaction Checker

Graduation project (FCDS 2025). A full-stack system that predicts the likelihood and
severity of an interaction between two drugs from their molecular structure.

Enter two drug names → the names are resolved to SMILES via PubChem → a PyTorch model
scores the pair → you get an interaction **probability** and a learned **severity** class
(`None` / `Minor` / `Moderate` / `Major`).

The severity model is trained and evaluated reproducibly in [`ml/`](ml/README.md) on
DDInter-labelled drug pairs, with drug-disjoint cold-start splits so the reported numbers
reflect generalization to unseen drugs. See that README for the dataset, evaluation
protocol, and results.

## Architecture

```
┌────────────────┐      ┌──────────────────┐      ┌────────────────────┐
│  mobile-rn     │      │   backend-go     │      │   inference-py     │
│  (Expo / RN)   │─────▶│   Go HTTP API    │─────▶│  FastAPI + PyTorch │
│  search & UI   │ HTTP │  :8080           │ HTTP │  :8001             │
└────────────────┘      └────────┬─────────┘      └────────────────────┘
                                 │
                                 ▼ name → SMILES
                          PubChem PUG REST API
```

| Component       | Stack                          | Port | Responsibility                                                        |
| --------------- | ------------------------------ | ---- | --------------------------------------------------------------------- |
| `inference-py`  | FastAPI, PyTorch, RDKit        | 8001 | Loads the trained model, turns SMILES into fingerprints, scores pairs |
| `backend-go`    | Go standard library            | 8080 | Resolves drug names → SMILES via PubChem, calls the inference service |
| `mobile-rn`     | React Native (Expo)            | —    | Drug search, two-drug selection, animated result screen               |

## Prerequisites

- **Docker** + **Docker Compose v2** — to run the backend stack (recommended path).
- **Node.js 18+** — only needed to run the mobile app.
- Outbound internet access — the Go backend calls the public PubChem API.

## Quick start (backend stack)

From the repository root:

```bash
docker compose up --build
```

This builds and starts both services. The first build downloads PyTorch and RDKit, so it
takes several minutes. The backend waits until the inference service reports healthy
before it starts.

Once up:

- Inference service: <http://localhost:8001>
- Backend API: <http://localhost:8080>

Stop the stack with `docker compose down`.

## API

### Backend (`:8080`)

**Search drug names** (PubChem autocomplete)

```bash
curl "http://localhost:8080/v1/drugs/search?q=aspir"
# { "query": "aspir", "results": ["Aspirin", "aspirin", ...] }
```

**Resolve a name to SMILES**

```bash
curl "http://localhost:8080/v1/resolve-smiles?name=aspirin"
# { "name": "aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O" }
```

**Predict an interaction** (names in, prediction out)

```bash
curl -X POST "http://localhost:8080/v1/ddi/predict" \
  -H 'Content-Type: application/json' \
  -d '{"drugAName":"warfarin","drugBName":"aspirin"}'
# {
#   "drugAName": "warfarin", "drugBName": "aspirin",
#   "smilesA": "...", "smilesB": "...",
#   "interactionProbability": 0.83, "severity": "Major", "model": "severity_model",
#   "probabilities": { "None": 0.17, "Minor": 0.06, "Moderate": 0.31, "Major": 0.46 }
# }
```

### Inference service (`:8001`)

**Health**

```bash
curl http://localhost:8001/health   # { "ok": true }
```

**Predict from SMILES directly**

```bash
curl -X POST "http://localhost:8001/predict" \
  -H 'Content-Type: application/json' \
  -d '{"smilesA":"CC(=O)OC1=CC=CC=C1C(=O)O","smilesB":"CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"}'
# {
#   "interactionProbability": 0.71, "severity": "Moderate",
#   "probabilities": { "None": 0.29, "Minor": 0.08, "Moderate": 0.40, "Major": 0.23 }
# }
```

`interactionProbability` is `1 − P(None)`; `severity` is `None` when an interaction is
unlikely (`interactionProbability < 0.5`) and otherwise the most likely
`Minor`/`Moderate`/`Major` class; `probabilities` is the full calibrated per-class
distribution.

## The model

The inference service serves a **two-stage** model (see [`ml/README.md`](ml/README.md) for the
full design and ablation). Stage 1 predicts whether the two drugs interact; stage 2 grades the
severity. Each molecule is embedded with **ChemBERTa** (`DeepChem/ChemBERTa-77M-MLM`), combined
into a symmetric pair vector `[vA + vB, |vA − vB|, vA * vB]` plus biological mechanistic-overlap
features (shared targets/enzymes/transporters/carriers). `interactionProbability` is stage 1's
score; `severity` is `None` below the 0.5 threshold, else the most likely stage-2 class.
Severity is **learned**, not thresholded from the probability.

The two-stage models, metadata, and a SMILES→biology lookup are produced by the `ml/` pipeline
and copied into `inference-py/app/models/`. Paths are set in `docker-compose.yml`:

| Env var      | Default    | Description                                                                     |
| ------------ | ---------- | ------------------------------------------------------------------------------- |
| `MODELS_DIR` | `/models`  | Directory holding `stage1.pt`, `stage2.pt`, `severity_metadata.json`, `bio_lookup.json` |
| `DEVICE`     | `cpu`      | Torch device (`cpu` / `cuda`)                                                   |

To retrain or change the model, see [`ml/README.md`](ml/README.md) and re-export. The inference
image installs `transformers` and pre-downloads ChemBERTa at build time.

The backend reads `INFERENCE_URL` (default `http://localhost:8001`) to locate the
inference service.

## Mobile app

The Expo client lives in `mobile-rn/`. See [`mobile-rn/README.md`](mobile-rn/README.md)
for details.

```bash
cd mobile-rn
npm install
npm run start
```

> The API host is set by the `API_BASE` constant near the top of `mobile-rn/App.js`.
> Point it at your backend (e.g. `http://localhost:8080`, or your machine's LAN IP when
> testing on a physical device).

## Repository layout

```
.
├── docker-compose.yml     # Runs backend + inference together
├── backend-go/            # Go API gateway (PubChem resolution + inference proxy)
├── inference-py/          # FastAPI + PyTorch model service
│   └── app/models/        # Trained checkpoints (.pt)
└── mobile-rn/             # React Native (Expo) client
```
