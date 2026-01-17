# inference-py/app/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import DDIModel

app = FastAPI(title="DDI Inference Service")

class PredictReq(BaseModel):
    smilesA: str
    smilesB: str

MODEL_PATH = os.getenv("MODEL_PATH", "/models/ddi_pairmlp_scaffold_smiles_only.pt")
DEVICE = os.getenv("DEVICE", "cpu")

ddi = None

@app.on_event("startup")
def startup():
    global ddi
    ddi = DDIModel(MODEL_PATH, device=DEVICE)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(req: PredictReq):
    try:
        prob = ddi.predict_smiles_pair(req.smilesA, req.smilesB)
        return {"probability": prob}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="inference error")
