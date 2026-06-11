# ml/tests/test_assemble.py
import numpy as np
from ml.assemble import struct_dim, assemble_pair
from ml.features import LIGHT_DESCRIPTORS
from ml.bio_features import empty_bio_sets, N_BIO

NDESC = len(LIGHT_DESCRIPTORS)


def test_struct_dim_light():
    assert struct_dim({"features": {"structural": "light", "light_fp_bits": 256}}) == 256 + NDESC


def test_assemble_pair_symmetric_and_dim():
    cfg = {"features": {"structural": "light", "light_fp_bits": 128, "use_bio": True}}
    sa = np.ones(128 + NDESC, dtype=np.float32)
    sb = np.arange(128 + NDESC, dtype=np.float32)
    ba = empty_bio_sets(); bb = empty_bio_sets()
    ba["targets"] = {"T1"}; bb["targets"] = {"T1"}
    v1 = assemble_pair(sa, sb, ba, bb, use_bio=True)
    v2 = assemble_pair(sb, sa, bb, ba, use_bio=True)
    assert np.allclose(v1, v2)
    assert v1.shape[0] == 3 * (128 + NDESC) + N_BIO
