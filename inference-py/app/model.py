# inference-py/app/model.py
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors, QED

CLASSES = ["None", "Minor", "Moderate", "Major"]
SEV3 = ["Minor", "Moderate", "Major"]
ENTITY_TYPES = ["targets", "enzymes", "transporters", "carriers"]
_CHEMBERTA = "DeepChem/ChemBERTa-77M-MLM"

# Must match ml/features.py LIGHT_DESCRIPTORS order (used only if structural=='light').
_LIGHT_DESCRIPTORS = [
    Descriptors.MolWt, Descriptors.ExactMolWt, Descriptors.HeavyAtomCount, Descriptors.NumHAcceptors,
    Descriptors.NumHDonors, Descriptors.NumRotatableBonds, rdMolDescriptors.CalcNumAromaticRings,
    rdMolDescriptors.CalcNumAliphaticRings, rdMolDescriptors.CalcNumSaturatedRings, Descriptors.RingCount,
    rdMolDescriptors.CalcFractionCSP3, Descriptors.TPSA, Crippen.MolLogP, Crippen.MolMR,
    Descriptors.NumValenceElectrons, rdMolDescriptors.CalcNumHeteroatoms, Descriptors.NOCount,
    Descriptors.NHOHCount, rdMolDescriptors.CalcNumAromaticHeterocycles,
    rdMolDescriptors.CalcNumAliphaticHeterocycles, rdMolDescriptors.CalcNumAromaticCarbocycles,
    rdMolDescriptors.CalcNumAliphaticCarbocycles, Descriptors.LabuteASA, Descriptors.BalabanJ,
    Descriptors.BertzCT, Descriptors.Chi0, Descriptors.Chi1, Descriptors.Chi0n, Descriptors.Chi1n,
    Descriptors.HallKierAlpha, Descriptors.Kappa1, Descriptors.Kappa2, Descriptors.Kappa3,
    Descriptors.MaxPartialCharge, Descriptors.MinPartialCharge, Descriptors.NumRadicalElectrons,
    QED.qed, lambda m: Chem.GetFormalCharge(m),
]


class MLPHead(nn.Module):
    def __init__(self, in_dim, n_out, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


_emb_cache = {}


def _embed(smiles):
    if "model" not in _emb_cache:
        from transformers import AutoTokenizer, AutoModel
        _emb_cache["tok"] = AutoTokenizer.from_pretrained(_CHEMBERTA)
        _emb_cache["model"] = AutoModel.from_pretrained(_CHEMBERTA).eval()
    tok, mdl = _emb_cache["tok"], _emb_cache["model"]
    with torch.no_grad():
        enc = tok(str(smiles), return_tensors="pt", truncation=True, max_length=256)
        out = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled.squeeze(0).cpu().numpy().astype(np.float32)


def _light_vector(smiles, fp_bits, radius):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = np.zeros((fp_bits,), dtype=np.float32)
    cfp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=fp_bits)
    for idx, cnt in cfp.GetNonzeroElements().items():
        fp[idx] = float(cnt)
    desc = np.zeros((len(_LIGHT_DESCRIPTORS),), dtype=np.float32)
    for i, fn in enumerate(_LIGHT_DESCRIPTORS):
        try:
            v = float(fn(mol)); desc[i] = v if np.isfinite(v) else 0.0
        except Exception:
            desc[i] = 0.0
    return np.concatenate([fp, desc]).astype(np.float32)


def _struct_vector(smiles, meta):
    if Chem.MolFromSmiles(str(smiles)) is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if meta["structural"] == "chemberta":
        return _embed(smiles)
    return _light_vector(smiles, int(meta["light_fp_bits"]), int(meta["fp_radius"]))


def _struct_pair(va, vb):
    return np.concatenate([va + vb, np.abs(va - vb), va * vb]).astype(np.float32)


def _bio_pair(a, b):
    vals = []
    for t in ENTITY_TYPES:
        sa, sb = set(a.get(t, [])), set(b.get(t, []))
        inter = len(sa & sb); union = len(sa | sb)
        jacc = inter / union if union else 0.0
        vals += [float(inter), float(jacc), float(len(sa) + len(sb)), float(abs(len(sa) - len(sb)))]
    return np.array(vals, dtype=np.float32)


def _canonical(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m) if m is not None else None


class TwoStageDDIModel:
    def __init__(self, stage1_path, stage2_path, metadata_path, bio_lookup_path, device="cpu"):
        self.device = torch.device(device)
        self.meta = json.load(open(metadata_path))
        self.use_bio = bool(self.meta["use_bio"])
        self.threshold = float(self.meta.get("interaction_threshold", 0.5))
        in_dim = int(self.meta["pair_dim"]); hidden = int(self.meta["hidden"])
        self.s1 = MLPHead(in_dim, 1, hidden); self.s1.load_state_dict(
            torch.load(stage1_path, map_location=self.device)["model_state_dict"]); self.s1.eval()
        self.s2 = MLPHead(in_dim, 3, hidden); self.s2.load_state_dict(
            torch.load(stage2_path, map_location=self.device)["model_state_dict"]); self.s2.eval()
        self.bio = json.load(open(bio_lookup_path)) if self.use_bio else {}

    def _bio_for(self, smiles):
        cs = _canonical(smiles)
        empty = {t: [] for t in ENTITY_TYPES}
        return self.bio.get(cs, empty) if cs else empty

    @torch.no_grad()
    def predict_with_severity(self, smiles_a, smiles_b):
        va = _struct_vector(smiles_a, self.meta)
        vb = _struct_vector(smiles_b, self.meta)
        blocks = [_struct_pair(va, vb)]
        if self.use_bio:
            blocks.append(_bio_pair(self._bio_for(smiles_a), self._bio_for(smiles_b)))
        feats = torch.tensor(np.concatenate(blocks), device=self.device).unsqueeze(0)
        p_int = float(torch.sigmoid(self.s1(feats)).item())
        sev = F.softmax(self.s2(feats), dim=1).squeeze(0).numpy()
        prob_map = {"None": 1.0 - p_int, "Minor": p_int * float(sev[0]),
                    "Moderate": p_int * float(sev[1]), "Major": p_int * float(sev[2])}
        severity = "None" if p_int < self.threshold else SEV3[int(np.argmax(sev))]
        return {"interactionProbability": p_int, "severity": severity, "probabilities": prob_map}


DDIModel = TwoStageDDIModel
