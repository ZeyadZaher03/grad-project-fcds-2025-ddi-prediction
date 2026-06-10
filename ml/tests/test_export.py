# ml/tests/test_export.py
import json
from ml.pipeline.p07_export import build_metadata


def test_metadata_has_required_serving_fields():
    md = build_metadata(fp_bits=2048, radius=2,
                        descriptors=["MolWt"], temperature=1.7,
                        in_dim=6168, gnn_s2_macro_f1=0.41, mlp_s2_macro_f1=0.46)
    assert md["classes"] == ["None", "Minor", "Moderate", "Major"]
    assert md["fp_bits"] == 2048 and md["temperature"] == 1.7
    assert md["in_dim"] == 6168
    assert md["comparison"]["gnn_s2_macro_f1"] == 0.41
