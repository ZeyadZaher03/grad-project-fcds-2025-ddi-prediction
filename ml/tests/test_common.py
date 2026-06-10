# ml/tests/test_common.py
from ml.common import normalize_name, CLASSES, CLASS_TO_IDX, LEVEL_MAP


def test_normalize_name_strips_punct_and_case():
    assert normalize_name("Acetyl-Salicylic Acid") == "acetylsalicylicacid"
    assert normalize_name("Vitamin B12 (cobalamin)") == "vitaminb12cobalamin"


def test_class_order_is_fixed():
    assert CLASSES == ["None", "Minor", "Moderate", "Major"]
    assert CLASS_TO_IDX["None"] == 0 and CLASS_TO_IDX["Major"] == 3


def test_level_map_excludes_unknown():
    assert "Unknown" not in LEVEL_MAP
    assert LEVEL_MAP["Moderate"] == "Moderate"
