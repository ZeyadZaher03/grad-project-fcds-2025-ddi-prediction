# inference-py/app/model.py
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Crippen, rdMolDescriptors

CLASSES = ["None", "Minor", "Moderate", "Major"]
SEVERITY_CLASSES = ["Minor", "Moderate", "Major"]

_DESCRIPTORS = [
    Descriptors.MolWt, Crippen.MolLogP, Descriptors.NumHDonors,
    Descriptors.NumHAcceptors, Descriptors.TPSA, Descriptors.NumRotatableBonds,
    rdMolDescriptors.CalcNumAromaticRings, rdMolDescriptors.CalcFractionCSP3,
]


class PairMLP4(nn.Module):
    def __init__(self, in_dim, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 4),
        )

    def forward(self, x):
        return self.net(x)


def _drug_vector(smiles, fp_bits, radius):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = np.zeros((fp_bits,), dtype=np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_bits)
    DataStructs.ConvertToNumpyArray(bv, fp)
    desc = np.zeros((len(_DESCRIPTORS),), dtype=np.float32)
    for i, fn in enumerate(_DESCRIPTORS):
        try:
            desc[i] = float(fn(mol))
        except Exception:
            desc[i] = 0.0
    return np.concatenate([fp, desc]).astype(np.float32)


def _pair_features(va, vb):
    return np.concatenate([va + vb, np.abs(va - vb), va * vb]).astype(np.float32)


class DDIModel:
    def __init__(self, model_path: str, metadata_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        with open(metadata_path) as f:
            self.meta = json.load(f)
        self.fp_bits = int(self.meta["fp_bits"])
        self.radius = int(self.meta["fp_radius"])
        self.temperature = float(self.meta.get("temperature", 1.0))
        ck = torch.load(model_path, map_location=self.device)
        self.model = PairMLP4(in_dim=int(ck["in_dim"]), hidden=int(ck.get("hidden", 512)))
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_with_severity(self, smiles_a: str, smiles_b: str) -> dict:
        va = _drug_vector(smiles_a, self.fp_bits, self.radius)
        vb = _drug_vector(smiles_b, self.fp_bits, self.radius)
        feats = torch.tensor(_pair_features(va, vb), device=self.device).unsqueeze(0)
        probs = F.softmax(self.model(feats) / self.temperature, dim=1).squeeze(0)
        prob_map = {CLASSES[i]: float(probs[i]) for i in range(4)}
        interaction = 1.0 - prob_map["None"]
        # Keep severity consistent with the interaction decision: if an interaction is
        # unlikely (interactionProbability < 0.5, i.e. None is the majority outcome),
        # report "None"; otherwise report the most likely Minor/Moderate/Major class.
        if interaction < 0.5:
            severity = "None"
        else:
            sev_idx = int(torch.tensor([probs[CLASSES.index(c)]
                                        for c in SEVERITY_CLASSES]).argmax())
            severity = SEVERITY_CLASSES[sev_idx]
        return {
            "interactionProbability": interaction,
            "severity": severity,
            "probabilities": prob_map,
        }
