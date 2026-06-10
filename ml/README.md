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
