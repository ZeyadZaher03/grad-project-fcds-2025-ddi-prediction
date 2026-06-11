# ml/tests/test_features_v2.py
import numpy as np
from ml.features import light_drug_vector, LIGHT_DESCRIPTORS, pair_features


def test_light_vector_shape_and_invalid_smiles():
    v = light_drug_vector("CC(=O)OC1=CC=CC=C1C(=O)O", fp_bits=512)
    assert v.shape == (512 + len(LIGHT_DESCRIPTORS),)
    assert np.all(np.isfinite(v))
    z = light_drug_vector("nonsense", fp_bits=512)
    assert z.shape == (512 + len(LIGHT_DESCRIPTORS),)
    assert np.all(z == 0)


def test_light_vector_is_count_based():
    v = light_drug_vector("CCCCCCCCCC", fp_bits=512)
    assert v[:512].max() >= 1.0


def test_pair_features_symmetric_on_light_vectors():
    a = light_drug_vector("CCO", fp_bits=256)
    b = light_drug_vector("CCN", fp_bits=256)
    assert np.allclose(pair_features(a, b), pair_features(b, a))
