# ml/features.py
"""SMILES -> per-drug feature vector and symmetric pair features."""
from __future__ import annotations
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Crippen, rdMolDescriptors

FP_BITS_DEFAULT = 2048

# Order MUST stay fixed; mirrored in config.yaml features.descriptors.
_DESCRIPTORS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Crippen.MolLogP),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("TPSA", Descriptors.TPSA),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumAromaticRings", rdMolDescriptors.CalcNumAromaticRings),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
]
N_DESC = len(_DESCRIPTORS)


def drug_vector(smiles: str, fp_bits: int = FP_BITS_DEFAULT, radius: int = 2) -> np.ndarray:
    """Return concatenated [Morgan fingerprint (fp_bits), descriptors (8)] as float32."""
    mol = Chem.MolFromSmiles(str(smiles))
    fp = np.zeros((fp_bits,), dtype=np.float32)
    desc = np.zeros((N_DESC,), dtype=np.float32)
    if mol is not None:
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_bits)
        DataStructs.ConvertToNumpyArray(bv, fp)
        for i, (_, fn) in enumerate(_DESCRIPTORS):
            try:
                desc[i] = float(fn(mol))
            except Exception:
                desc[i] = 0.0
    return np.concatenate([fp, desc]).astype(np.float32)


def pair_features(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
    """Symmetric pair features: order-invariant by construction."""
    summ = va + vb
    diff = np.abs(va - vb)
    prod = va * vb
    return np.concatenate([summ, diff, prod]).astype(np.float32)
