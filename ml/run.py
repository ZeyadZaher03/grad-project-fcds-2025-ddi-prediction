# ml/run.py
"""Run all pipeline stages in order."""
from ml.pipeline import (p01_download, p02_build_labels, p03_features,
                         p04_splits, p05_train, p06_evaluate, p07_export)


def main():
    for stage in [p01_download, p02_build_labels, p03_features,
                  p04_splits, p05_train, p06_evaluate, p07_export]:
        print(f"\n===== {stage.__name__} =====")
        stage.main()


if __name__ == "__main__":
    main()
