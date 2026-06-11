# ml/run_ablation.py
"""Sweep feature configs and emit an ablation table of end-to-end S2 macro-F1."""
from __future__ import annotations
import os, json, copy, yaml
import ml.config as mlconfig
from ml.config import load_config

# (structural, use_bio, label) configurations to compare.
CONFIGS = [
    ("light", False, "struct_light"),
    ("light", True, "struct_light+bio"),
    ("chemberta", True, "chemberta+bio"),
]


def _run_stage(modpath):
    import importlib
    importlib.import_module(modpath).main()


def main():
    base = load_config()
    artifacts = base["paths"]["artifacts_dir"]
    results = []
    for structural, use_bio, label in CONFIGS:
        cfg = copy.deepcopy(base)
        cfg["features"]["structural"] = structural
        cfg["features"]["use_bio"] = use_bio
        if structural == "chemberta":
            from ml.embeddings import chemberta_available
            if not chemberta_available():
                print(f"[skip] {label}: transformers not installed")
                results.append((label, None))
                continue
        tmp = f"/tmp/ablation_{structural}_{use_bio}.yaml"
        yaml.safe_dump(cfg, open(tmp, "w"))
        mlconfig._DEFAULT = tmp
        print(f"\n===== ablation: {label} =====")
        _run_stage("ml.pipeline.p03b_features_v2")
        _run_stage("ml.pipeline.p05b_train_stages")
        _run_stage("ml.pipeline.p06b_evaluate_v2")
        tag = structural + ("_bio" if use_bio else "")
        m = json.load(open(os.path.join(artifacts, f"metrics_v2_{tag}.json")))
        s2 = m["regimes"].get("test_S2", {})
        results.append((label, s2.get("end2end_macro_f1")))

    lines = ["# Ablation - end-to-end S2 macro-F1\n",
             "| config | S2 macro-F1 |", "|---|---|",
             "| iteration-1 baseline (single 4-class) | 0.297 |"]
    for label, f1 in results:
        lines.append(f"| {label} | {f1:.3f} |" if f1 is not None else f"| {label} | skipped |")
    open(os.path.join(artifacts, "ablation_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
