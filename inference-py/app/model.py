import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

class PairMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x).squeeze(-1)
        return x

def smiles_to_fp_np(smiles: str, n_bits: int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def build_pair_feats_from_two_fp(fpA: torch.Tensor, fpB: torch.Tensor) -> torch.Tensor:
    fpA = fpA.unsqueeze(0)
    fpB = fpB.unsqueeze(0)
    abs_diff = torch.abs(fpA - fpB)
    prod = fpA * fpB
    return torch.cat([fpA, fpB, abs_diff, prod], dim=1)


class DDIModel:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(model_path, map_location=self.device)
        if "model_state_dict" not in ckpt:
            raise ValueError("checkpoint missing model_state_dict")
        self.fp_size = int(ckpt.get("fp_size", 0))
        if self.fp_size <= 0:
            raise ValueError("checkpoint missing fp_size")
        in_dim = int(ckpt.get("in_dim", 0))
        if in_dim <= 0:
            raise ValueError("checkpoint missing in_dim")
        self.model = PairMLP(in_dim=in_dim)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _fp_tensor(self, smiles: str) -> torch.Tensor:
        arr = smiles_to_fp_np(smiles, n_bits=self.fp_size)
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict_smiles_pair(self, smiles_a: str, smiles_b: str) -> float:
        fp_a = self._fp_tensor(smiles_a)
        fp_b = self._fp_tensor(smiles_b)
        feats = build_pair_feats_from_two_fp(fp_a, fp_b)
        logits = self.model(feats)
        prob = torch.sigmoid(logits).item()
        return float(prob)
    
    @torch.no_grad()
    def predict_with_severity(self, smiles_a: str, smiles_b: str) -> dict:
        fp_a = self._fp_tensor(smiles_a)
        fp_b = self._fp_tensor(smiles_b)
        feats = build_pair_feats_from_two_fp(fp_a, fp_b)
        logits = self.model(feats)
        prob = torch.sigmoid(logits).item()
        
        # Determine severity based on model output
        if prob >= 0.7:
            severity = "major"
        elif prob >= 0.4:
            severity = "moderate"
        else:
            severity = "minor"
            
        return {
            "probability": float(prob),
            "severity": severity
        }