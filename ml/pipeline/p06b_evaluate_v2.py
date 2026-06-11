# ml/pipeline/p06b_evaluate_v2.py
"""Stage 6b: evaluate the two-stage model (stage1 AUC, stage2 F1, end-to-end 4-class)."""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score
from ml.config import load_config
from ml.common import CLASSES, CLASS_TO_IDX
from ml.assemble import pair_dim
from ml.bio_features import empty_bio_sets, ENTITY_TYPES
from ml.models import MLPHead
from ml.pipeline.p05b_train_stages import _load_struct, _load_bio, SEV3

SEV3_IDX = {c: i for i, c in enumerate(SEV3)}


def combine_four_class(p_interact: np.ndarray, sev_probs: np.ndarray) -> np.ndarray:
    """[None, Minor, Moderate, Major] = [1-p, p*minor, p*mod, p*major]."""
    p = np.asarray(p_interact).reshape(-1, 1)
    none = 1.0 - p
    return np.concatenate([none, p * sev_probs], axis=1)


def _matrix(rows, X, idx, bio, cfg, use_bio):
    from ml.assemble import assemble_pair
    out = [assemble_pair(X[idx[a]], X[idx[b]], bio.get(a, empty_bio_sets()),
                         bio.get(b, empty_bio_sets()), use_bio)
           for a, b in zip(rows.drugA_id, rows.drugB_id)]
    return torch.tensor(np.stack(out), dtype=torch.float32)


def main():
    cfg = load_config()
    proc, artifacts = cfg["paths"]["processed_dir"], cfg["paths"]["artifacts_dir"]
    use_bio = cfg["features"]["use_bio"]
    tag = cfg["features"]["structural"] + ("_bio" if use_bio else "")
    X, idx = _load_struct(cfg); bio = _load_bio(cfg)
    sp = pd.read_parquet(os.path.join(proc, "splits.parquet"))
    sp = sp[sp.drugA_id.isin(idx) & sp.drugB_id.isin(idx)].reset_index(drop=True)
    sp["y4"] = sp["label"].map(CLASS_TO_IDX)

    s1 = MLPHead(in_dim=pair_dim(cfg), n_out=1, hidden=cfg["train"]["hidden"])
    s1.load_state_dict(torch.load(os.path.join(artifacts, f"model_stage1_{tag}.pt"))["model_state_dict"]); s1.eval()
    s2 = MLPHead(in_dim=pair_dim(cfg), n_out=3, hidden=cfg["train"]["hidden"])
    s2.load_state_dict(torch.load(os.path.join(artifacts, f"model_stage2_{tag}.pt"))["model_state_dict"]); s2.eval()

    from sklearn.metrics import precision_recall_fscore_support
    report = {"tag": tag, "regimes": {}}
    for regime in ["test_S0", "test_S1", "test_S2"]:
        rows = sp[sp.split == regime]
        if len(rows) == 0:
            continue
        M = _matrix(rows, X, idx, bio, cfg, use_bio)
        with torch.no_grad():
            p_int = torch.sigmoid(s1(M)).numpy().ravel()
            sev = F.softmax(s2(M), dim=1).numpy()
        y4 = rows["y4"].values
        y_bin = (y4 > 0).astype(int)
        auc = roc_auc_score(y_bin, p_int) if 0 < y_bin.sum() < len(y_bin) else None
        inter_mask = y4 > 0
        s2f1 = None
        if inter_mask.sum() > 0:
            ytrue_sev = np.array([SEV3_IDX[CLASSES[c]] for c in y4[inter_mask]])
            s2f1 = float(f1_score(ytrue_sev, sev[inter_mask].argmax(1), average="macro"))
        four = combine_four_class(p_int, sev)
        pred4 = four.argmax(1)
        per = {}
        P, R, Fc, _ = precision_recall_fscore_support(y4, pred4, labels=[0, 1, 2, 3], average=None, zero_division=0)
        for i in range(4):
            per[CLASSES[i]] = {"precision": float(P[i]), "recall": float(R[i]), "f1": float(Fc[i])}
        report["regimes"][regime] = {
            "stage1_auc": float(auc) if auc is not None else None,
            "stage2_macro_f1": s2f1,
            "end2end_macro_f1": float(f1_score(y4, pred4, average="macro")),
            "end2end_accuracy": float(accuracy_score(y4, pred4)),
            "per_class": per,
            "confusion": confusion_matrix(y4, pred4, labels=[0, 1, 2, 3]).tolist(),
        }
    out = os.path.join(artifacts, f"metrics_v2_{tag}.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"[done] {tag}: " + " | ".join(
        f"{r}: e2e_F1={report['regimes'][r]['end2end_macro_f1']:.3f} "
        f"(S1auc={report['regimes'][r]['stage1_auc']}, S2f1={report['regimes'][r]['stage2_macro_f1']})"
        for r in report["regimes"]))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
