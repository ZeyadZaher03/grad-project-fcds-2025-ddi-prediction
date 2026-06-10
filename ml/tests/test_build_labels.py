# ml/tests/test_build_labels.py
from ml.pipeline.p01_download import ddinter_urls


def test_ddinter_urls_built_from_categories():
    cfg = {"ddinter": {"base_url": "https://x/code_", "categories": ["A", "P"]}}
    urls = ddinter_urls(cfg)
    assert urls == {
        "A": "https://x/code_A.csv",
        "P": "https://x/code_P.csv",
    }


import pandas as pd
from ml.pipeline.p02_build_labels import (
    canonical_pair, build_name_map, join_positives, sample_negatives,
)


def test_canonical_pair_orders_ids():
    assert canonical_pair("DB02", "DB01") == ("DB01", "DB02")
    assert canonical_pair("DB01", "DB02") == ("DB01", "DB02")


def test_build_name_map_normalizes():
    drugs = pd.DataFrame({"drug_id": ["DB1", "DB2"], "name": ["Aspirin", "Vitamin-C"]})
    m = build_name_map(drugs)
    assert m["aspirin"] == "DB1"
    assert m["vitaminc"] == "DB2"


def test_join_positives_keeps_only_mapped_pairs_and_drops_unknown():
    drugs = pd.DataFrame({"drug_id": ["DB1", "DB2"], "name": ["Aspirin", "Warfarin"]})
    dd = pd.DataFrame({
        "Drug_A": ["Aspirin", "Aspirin", "Ghostdrug"],
        "Drug_B": ["Warfarin", "Warfarin", "Warfarin"],
        "Level":  ["Major", "Unknown", "Major"],
    })
    pos = join_positives(dd, build_name_map(drugs))
    assert len(pos) == 1
    row = pos.iloc[0]
    assert (row.drugA_id, row.drugB_id, row.label) == ("DB1", "DB2", "Major")


def test_sample_negatives_excludes_masked_pairs():
    drug_ids = ["DB1", "DB2", "DB3", "DB4"]
    masked = {("DB1", "DB2")}
    negs = sample_negatives(drug_ids, masked, n=2, hard_fraction=0.0,
                            features=None, seed=1)
    assert len(negs) == 2
    for a, b in negs:
        assert (a, b) not in masked and a < b
