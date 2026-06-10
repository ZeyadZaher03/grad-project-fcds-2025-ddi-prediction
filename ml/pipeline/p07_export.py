# ml/pipeline/p07_export.py
"""Stage 7: export the deployable MLP + metadata for inference-py."""
from __future__ import annotations
import os, json, shutil
import torch
from ml.config import load_config
from ml.common import CLASSES


def build_metadata(fp_bits, radius, descriptors, temperature, in_dim,
                   gnn_s2_macro_f1, mlp_s2_macro_f1):
    return {
        "classes": CLASSES,
        "severity_classes": ["Minor", "Moderate", "Major"],
        "fp_bits": fp_bits,
        "fp_radius": radius,
        "descriptors": descriptors,
        "temperature": temperature,
        "in_dim": in_dim,
        "comparison": {"gnn_s2_macro_f1": gnn_s2_macro_f1,
                       "mlp_s2_macro_f1": mlp_s2_macro_f1},
    }


def main():
    cfg = load_config()
    artifacts = cfg["paths"]["artifacts_dir"]
    export = os.path.join(artifacts, "export")
    os.makedirs(export, exist_ok=True)

    metrics = json.load(open(os.path.join(artifacts, "metrics.json")))
    ck = torch.load(os.path.join(artifacts, "model_mlp.pt"), map_location="cpu")

    def s2(name):
        b = metrics["models"][name].get("test_S2")
        return float(b["macro_f1"]) if b else None

    md = build_metadata(
        fp_bits=cfg["features"]["fp_bits"], radius=cfg["features"]["fp_radius"],
        descriptors=cfg["features"]["descriptors"],
        temperature=metrics["temperatures"]["mlp"], in_dim=ck["in_dim"],
        gnn_s2_macro_f1=s2("gnn"), mlp_s2_macro_f1=s2("mlp"),
    )
    torch.save({"model_state_dict": ck["model_state_dict"], "in_dim": ck["in_dim"],
                "hidden": cfg["train"]["hidden"]}, os.path.join(export, "model.pt"))
    json.dump(md, open(os.path.join(export, "metadata.json"), "w"), indent=2)
    print(f"[done] exported MLP -> {export} (S2 macro-F1 mlp={md['comparison']['mlp_s2_macro_f1']}, "
          f"gnn={md['comparison']['gnn_s2_macro_f1']})")


if __name__ == "__main__":
    main()
