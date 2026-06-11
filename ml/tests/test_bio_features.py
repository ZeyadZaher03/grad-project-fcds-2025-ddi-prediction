# ml/tests/test_bio_features.py
import numpy as np
from ml.bio_features import (parse_entity_cell, bio_pair_features,
                             BIO_FEATURE_NAMES, empty_bio_sets)


def test_parse_entity_cell_splits_ids_and_handles_nan():
    assert parse_entity_cell("BE1 BE2 BE3") == {"BE1", "BE2", "BE3"}
    assert parse_entity_cell(float("nan")) == set()
    assert parse_entity_cell("") == set()


def test_bio_pair_features_symmetric_and_named():
    a = {"targets": {"T1", "T2"}, "enzymes": {"E1"}, "transporters": set(), "carriers": set()}
    b = {"targets": {"T2", "T3"}, "enzymes": {"E1"}, "transporters": set(), "carriers": set()}
    fa = bio_pair_features(a, b)
    fb = bio_pair_features(b, a)
    assert fa.shape == (len(BIO_FEATURE_NAMES),)
    assert np.allclose(fa, fb)


def test_bio_pair_features_overlap_values():
    a = {"targets": {"T1", "T2"}, "enzymes": set(), "transporters": set(), "carriers": set()}
    b = {"targets": {"T2"}, "enzymes": set(), "transporters": set(), "carriers": set()}
    feats = dict(zip(BIO_FEATURE_NAMES, bio_pair_features(a, b)))
    assert feats["targets_shared"] == 1.0
    assert abs(feats["targets_jaccard"] - 0.5) < 1e-6


def test_empty_bio_sets_has_all_types():
    s = empty_bio_sets()
    assert set(s) == {"targets", "enzymes", "transporters", "carriers"}
    assert all(v == set() for v in s.values())
