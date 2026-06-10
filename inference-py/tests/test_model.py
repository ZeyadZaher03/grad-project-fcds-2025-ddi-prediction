# inference-py/tests/test_model.py
import os
import pytest
from app.model import DDIModel

MODEL = os.path.join(os.path.dirname(__file__), "..", "app", "models", "severity_model.pt")
META = os.path.join(os.path.dirname(__file__), "..", "app", "models", "severity_metadata.json")


@pytest.mark.skipif(not os.path.exists(MODEL), reason="export model not present")
def test_predict_returns_calibrated_severity_fields():
    m = DDIModel(MODEL, META, device="cpu")
    out = m.predict_with_severity("CCO", "CCN")  # ethanol-like vs ethylamine-like
    assert set(out["probabilities"]) == {"None", "Minor", "Moderate", "Major"}
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-4
    assert out["severity"] in {"None", "Minor", "Moderate", "Major"}
    assert 0.0 <= out["interactionProbability"] <= 1.0
    # severity must stay consistent with the interaction decision.
    if out["interactionProbability"] < 0.5:
        assert out["severity"] == "None"
    else:
        assert out["severity"] in {"Minor", "Moderate", "Major"}


def test_severity_none_when_interaction_unlikely():
    # A self-pair (identical molecule) should not be flagged as a severe interaction;
    # whatever the score, severity must agree with the interactionProbability gate.
    m = DDIModel(MODEL, META, device="cpu")
    out = m.predict_with_severity("CCO", "CCO")
    if out["interactionProbability"] < 0.5:
        assert out["severity"] == "None"
    else:
        assert out["severity"] in {"Minor", "Moderate", "Major"}


def test_predict_is_symmetric():
    m = DDIModel(MODEL, META, device="cpu")
    a = m.predict_with_severity("CCO", "CCN")
    b = m.predict_with_severity("CCN", "CCO")
    assert abs(a["interactionProbability"] - b["interactionProbability"]) < 1e-5
