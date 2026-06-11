# inference-py/tests/test_model_v2.py
import os
import pytest
from app.model import TwoStageDDIModel

D = os.path.join(os.path.dirname(__file__), "..", "app", "models")
S1 = os.path.join(D, "stage1.pt"); S2 = os.path.join(D, "stage2.pt")
META = os.path.join(D, "severity_metadata.json"); BIO = os.path.join(D, "bio_lookup.json")


@pytest.mark.skipif(not os.path.exists(S1), reason="export not present")
def test_two_stage_predict_fields_and_consistency():
    m = TwoStageDDIModel(S1, S2, META, BIO, device="cpu")
    out = m.predict_with_severity("CC(=O)OC1=CC=CC=C1C(=O)O", "CCO")
    assert set(out["probabilities"]) == {"None", "Minor", "Moderate", "Major"}
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-4
    assert 0.0 <= out["interactionProbability"] <= 1.0
    if out["interactionProbability"] < 0.5:
        assert out["severity"] == "None"
    else:
        assert out["severity"] in {"Minor", "Moderate", "Major"}


def test_two_stage_symmetric():
    m = TwoStageDDIModel(S1, S2, META, BIO, device="cpu")
    a = m.predict_with_severity("CCO", "CCN")
    b = m.predict_with_severity("CCN", "CCO")
    assert abs(a["interactionProbability"] - b["interactionProbability"]) < 1e-5
