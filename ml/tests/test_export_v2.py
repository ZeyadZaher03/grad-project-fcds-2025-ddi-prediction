# ml/tests/test_export_v2.py
from ml.pipeline.p07b_export_v2 import build_v2_metadata


def test_v2_metadata_fields():
    md = build_v2_metadata(structural="chemberta", use_bio=True, light_fp_bits=1024,
                           fp_radius=2, pair_dim=1168, hidden=512,
                           interaction_threshold=0.5)
    assert md["classes"] == ["None", "Minor", "Moderate", "Major"]
    assert md["severity_classes"] == ["Minor", "Moderate", "Major"]
    assert md["structural"] == "chemberta" and md["use_bio"] is True
    assert md["interaction_threshold"] == 0.5 and md["pair_dim"] == 1168
