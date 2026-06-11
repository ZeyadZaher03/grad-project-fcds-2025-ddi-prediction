# inference-py/app/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import TwoStageDDIModel

app = FastAPI(title="DDI Inference Service")


class PredictReq(BaseModel):
    smilesA: str
    smilesB: str


MODELS_DIR = os.getenv("MODELS_DIR", "/models")
DEVICE = os.getenv("DEVICE", "cpu")
ddi = None


@app.on_event("startup")
def startup():
    global ddi
    ddi = TwoStageDDIModel(
        os.path.join(MODELS_DIR, "stage1.pt"),
        os.path.join(MODELS_DIR, "stage2.pt"),
        os.path.join(MODELS_DIR, "severity_metadata.json"),
        os.path.join(MODELS_DIR, "bio_lookup.json"),
        device=DEVICE,
    )


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
