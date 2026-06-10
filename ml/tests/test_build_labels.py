# ml/tests/test_build_labels.py
from ml.pipeline.p01_download import ddinter_urls


def test_ddinter_urls_built_from_categories():
    cfg = {"ddinter": {"base_url": "https://x/code_", "categories": ["A", "P"]}}
    urls = ddinter_urls(cfg)
    assert urls == {
        "A": "https://x/code_A.csv",
        "P": "https://x/code_P.csv",
    }
