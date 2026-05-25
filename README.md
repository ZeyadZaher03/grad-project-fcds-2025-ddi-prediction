# DDI Prediction — Drug–Drug Interaction Checker

Graduation project (FCDS 2025). A full-stack system that predicts the likelihood and
severity of an interaction between two drugs from their molecular structure.

Enter two drug names → the names are resolved to SMILES via PubChem → a PyTorch model
scores the pair → you get an interaction **probability** and a **severity** label
(`minor` / `moderate` / `major`).

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
#   "probability": 0.997, "label": "major", "model": "best_model"
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
# { "probability": 0.998, "severity": "major" }
```

Severity thresholds: `probability >= 0.7` → `major`, `>= 0.4` → `moderate`, else `minor`.

## The model

The inference service loads a `PairMLP` checkpoint (`inference-py/app/models/`). Each
molecule is converted to a Morgan fingerprint (RDKit), and the pair feature vector is
`[fpA, fpB, |fpA − fpB|, fpA * fpB]`. The MLP outputs a single logit that is passed
through a sigmoid to produce the interaction probability.

The checkpoint stores `model_state_dict`, `fp_size`, and `in_dim`. Select which checkpoint
to load with the `MODEL_PATH` environment variable (set in `docker-compose.yml`):

- `ddi_pairmlp_scaffold_smiles_only.pt` — used by default in Compose.
- `best_model.pt` — alternative checkpoint.

| Env var      | Default                 | Description                          |
| ------------ | ----------------------- | ------------------------------------ |
| `MODEL_PATH` | `/models/best_model.pt` | Path to the checkpoint inside the container |
| `DEVICE`     | `cpu`                   | Torch device (`cpu` / `cuda`)        |

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
