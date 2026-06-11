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


# --- appended in Task 2: upgraded structural features ---
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED
from rdkit import Chem as _Chem

# Curated descriptor set (name, fn). All return a finite float for valid mols;
# NaN/inf are zeroed at the call site.
LIGHT_DESCRIPTORS = [
    ("MolWt", Descriptors.MolWt), ("ExactMolWt", Descriptors.ExactMolWt),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount), ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumHDonors", Descriptors.NumHDonors), ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumAromaticRings", rdMolDescriptors.CalcNumAromaticRings),
    ("NumAliphaticRings", rdMolDescriptors.CalcNumAliphaticRings),
    ("NumSaturatedRings", rdMolDescriptors.CalcNumSaturatedRings),
    ("RingCount", Descriptors.RingCount), ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
    ("TPSA", Descriptors.TPSA), ("MolLogP", Crippen.MolLogP), ("MolMR", Crippen.MolMR),
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
    ("NumHeteroatoms", rdMolDescriptors.CalcNumHeteroatoms),
    ("NOCount", Descriptors.NOCount), ("NHOHCount", Descriptors.NHOHCount),
    ("NumAromaticHeterocycles", rdMolDescriptors.CalcNumAromaticHeterocycles),
    ("NumAliphaticHeterocycles", rdMolDescriptors.CalcNumAliphaticHeterocycles),
    ("NumAromaticCarbocycles", rdMolDescriptors.CalcNumAromaticCarbocycles),
    ("NumAliphaticCarbocycles", rdMolDescriptors.CalcNumAliphaticCarbocycles),
    ("LabuteASA", Descriptors.LabuteASA), ("BalabanJ", Descriptors.BalabanJ),
    ("BertzCT", Descriptors.BertzCT), ("Chi0", Descriptors.Chi0), ("Chi1", Descriptors.Chi1),
    ("Chi0n", Descriptors.Chi0n), ("Chi1n", Descriptors.Chi1n),
    ("HallKierAlpha", Descriptors.HallKierAlpha), ("Kappa1", Descriptors.Kappa1),
    ("Kappa2", Descriptors.Kappa2), ("Kappa3", Descriptors.Kappa3),
    ("MaxPartialCharge", Descriptors.MaxPartialCharge),
    ("MinPartialCharge", Descriptors.MinPartialCharge),
    ("NumRadicalElectrons", Descriptors.NumRadicalElectrons),
    ("qed", QED.qed), ("FormalCharge", lambda m: _Chem.GetFormalCharge(m)),
]


def light_drug_vector(smiles: str, fp_bits: int = 1024, radius: int = 2) -> np.ndarray:
    """Count-based Morgan fingerprint + curated descriptors as float32; zeros if invalid."""
    mol = Chem.MolFromSmiles(str(smiles))
    fp = np.zeros((fp_bits,), dtype=np.float32)
    desc = np.zeros((len(LIGHT_DESCRIPTORS),), dtype=np.float32)
    if mol is not None:
        cfp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=fp_bits)  # counts
        for idx, cnt in cfp.GetNonzeroElements().items():
            fp[idx] = float(cnt)
        for i, (_, fn) in enumerate(LIGHT_DESCRIPTORS):
            try:
                val = float(fn(mol))
                desc[i] = val if np.isfinite(val) else 0.0
            except Exception:
                desc[i] = 0.0
    return np.concatenate([fp, desc]).astype(np.float32)
