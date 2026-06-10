# ml/tests/test_features.py
import numpy as np
from ml.features import drug_vector, pair_features, FP_BITS_DEFAULT


def test_drug_vector_shape_and_invalid_smiles():
    v = drug_vector("CC(=O)OC1=CC=CC=C1C(=O)O", fp_bits=256)  # aspirin
    assert v.shape == (256 + 8,)              # fp + 8 descriptors
    z = drug_vector("not-a-smiles", fp_bits=256)
    assert z.shape == (256 + 8,)
    assert np.all(z[:256] == 0)               # invalid -> zero fingerprint


def test_pair_features_are_symmetric():
    a = drug_vector("CCO", fp_bits=256)
    b = drug_vector("CCN", fp_bits=256)
    assert np.allclose(pair_features(a, b), pair_features(b, a))
