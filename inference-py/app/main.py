# inference-py/app/main.py
import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from .model import DDIModel

app = FastAPI(title="DDI Inference Service")


class PredictReq(BaseModel):
    smilesA: str
    smilesB: str

MODEL_PATH = os.getenv("MODEL_PATH", "/models/best_model.pt")
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
        result = ddi.predict_with_severity(req.smilesA, req.smilesB)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="inference error")
