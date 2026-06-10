# ml/tests/test_splits.py
import pandas as pd
from ml.pipeline.p04_splits import assign_regime, split_drugs


def test_split_drugs_disjoint():
    train, test = split_drugs(["D%d" % i for i in range(100)], test_frac=0.2, seed=1)
    assert len(test) == 20 and len(train) == 80
    assert set(train).isdisjoint(set(test))


def test_assign_regime_by_membership():
    test_drugs = {"D9", "D8"}
    assert assign_regime("D1", "D2", test_drugs) == "S0"   # both seen
    assert assign_regime("D1", "D9", test_drugs) == "S1"   # one unseen
    assert assign_regime("D8", "D9", test_drugs) == "S2"   # both unseen
