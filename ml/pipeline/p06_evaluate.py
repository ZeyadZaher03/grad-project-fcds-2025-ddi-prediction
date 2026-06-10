# ml/pipeline/p06_evaluate.py
"""Stage 6: evaluate both models on S0/S1/S2 + baselines, calibrate, write report."""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score, average_precision_score)
from sklearn.linear_model import LogisticRegression
from ml.config import load_config
from ml.common import CLASSES, CLASS_TO_IDX
from ml.features import pair_features
from ml.models import PairMLP4, SageDDI


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Single temperature scaling on validation logits (Guo et al. 2017)."""
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().clamp(min=0.05).item())


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _metrics_block(y_true, probs):
    pred = probs.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2, 3],
                                                 average=None, zero_division=0)
    block = {
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "per_class": {CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                   "f1": float(f[i])} for i in range(4)},
        "confusion": confusion_matrix(y_true, pred, labels=[0, 1, 2, 3]).tolist(),
        "ece": expected_calibration_error(probs, y_true),
    }
    auc, auprc = {}, {}
    for i in range(4):
        yi = (y_true == i).astype(int)
        if yi.sum() > 0 and yi.sum() < len(yi):
            auc[CLASSES[i]] = float(roc_auc_score(yi, probs[:, i]))
            auprc[CLASSES[i]] = float(average_precision_score(yi, probs[:, i]))
    block["ovr_auc"] = auc
    block["ovr_auprc"] = auprc
    return block


def main():
    cfg = load_config()
    proc, artifacts = cfg["paths"]["processed_dir"], cfg["paths"]["artifacts_dir"]
    splits = pd.read_parquet(os.path.join(proc, "splits.parquet"))
    feat = np.load(os.path.join(proc, "drug_features.npz"), allow_pickle=True)
    ids = list(feat["ids"]); idx = {d: i for i, d in enumerate(ids)}
    X = feat["X"].astype(np.float32)
    splits = splits[splits.drugA_id.isin(idx) & splits.drugB_id.isin(idx)].reset_index(drop=True)
    splits["y"] = splits["label"].map(CLASS_TO_IDX)

    def mlp_feats(rows):
        return torch.tensor(np.stack([pair_features(X[idx[a]], X[idx[b]])
                                      for a, b in zip(rows.drugA_id, rows.drugB_id)]))

    # --- MLP ---
    ck = torch.load(os.path.join(artifacts, "model_mlp.pt"), map_location="cpu")
    mlp = PairMLP4(in_dim=ck["in_dim"], hidden=cfg["train"]["hidden"],
                   dropout=cfg["train"]["dropout"])
    mlp.load_state_dict(ck["model_state_dict"]); mlp.eval()
    val = splits[splits.split == "val"]
    with torch.no_grad():
        T_mlp = fit_temperature(mlp(mlp_feats(val)),
                                torch.tensor(val["y"].values, dtype=torch.long))

    # --- GNN ---
    ckg = torch.load(os.path.join(artifacts, "model_gnn.pt"), map_location="cpu")
    gnn = SageDDI(in_channels=ckg["in_channels"], hidden=cfg["train"]["gnn_hidden"],
                  dropout=cfg["train"]["dropout"])
    gnn.load_state_dict(ckg["model_state_dict"]); gnn.eval()
    x = torch.tensor(X, dtype=torch.float32)
    tr_pos = splits[(splits.split == "train") & (splits.label != "None")]
    ei = torch.tensor([[idx[a] for a in tr_pos.drugA_id] + [idx[b] for b in tr_pos.drugB_id],
                       [idx[b] for b in tr_pos.drugB_id] + [idx[a] for a in tr_pos.drugA_id]],
                      dtype=torch.long)
    def gnn_pidx(rows):
        return torch.tensor([[idx[a] for a in rows.drugA_id], [idx[b] for b in rows.drugB_id]],
                            dtype=torch.long)
    with torch.no_grad():
        T_gnn = fit_temperature(gnn(x, ei, gnn_pidx(val)),
                                torch.tensor(val["y"].values, dtype=torch.long))

    # --- Logistic-regression baseline on pair features ---
    tr = splits[splits.split == "train"]
    lr = LogisticRegression(max_iter=200, class_weight="balanced", multi_class="multinomial")
    lr.fit(mlp_feats(tr).numpy(), tr["y"].values)
    # predict_proba columns are ordered by lr.classes_; downstream code indexes [:, 0..3].
    assert list(lr.classes_) == [0, 1, 2, 3], f"unexpected logreg class order: {lr.classes_}"

    report = {"temperatures": {"mlp": T_mlp, "gnn": T_gnn}, "models": {}}
    for name in ["mlp", "gnn", "logreg", "majority"]:
        report["models"][name] = {}
    majority_class = int(np.bincount(tr["y"].values, minlength=4).argmax())

    for regime in ["test_S0", "test_S1", "test_S2"]:
        rows = splits[splits.split == regime]
        if len(rows) == 0:
            continue
        y = rows["y"].values
        with torch.no_grad():
            probs_mlp = F.softmax(mlp(mlp_feats(rows)) / T_mlp, dim=1).numpy()
            probs_gnn = F.softmax(gnn(x, ei, gnn_pidx(rows)) / T_gnn, dim=1).numpy()
        probs_lr = lr.predict_proba(mlp_feats(rows).numpy())
        probs_maj = np.zeros((len(rows), 4)); probs_maj[:, majority_class] = 1.0
        report["models"]["mlp"][regime] = _metrics_block(y, probs_mlp)
        report["models"]["gnn"][regime] = _metrics_block(y, probs_gnn)
        report["models"]["logreg"][regime] = _metrics_block(y, probs_lr)
        report["models"]["majority"][regime] = _metrics_block(y, probs_maj)

    json.dump(report, open(os.path.join(artifacts, "metrics.json"), "w"), indent=2)

    for kind in ["mlp", "gnn"]:
        curve = json.load(open(os.path.join(artifacts, f"curve_{kind}.json")))
        ep = [c["epoch"] for c in curve]
        plt.figure()
        plt.plot(ep, [c["train_loss"] for c in curve], label="train_loss")
        plt.plot(ep, [c["val_loss"] for c in curve], label="val_loss")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title(f"{kind} learning curve")
        plt.savefig(os.path.join(artifacts, f"curve_{kind}.png")); plt.close()

    lines = ["# DDI Severity — Evaluation Report\n",
             f"Temperatures: MLP={T_mlp:.3f}, GNN={T_gnn:.3f}\n",
             "## Macro-F1 by model and split\n",
             "| model | S0 | S1 | S2 |", "|---|---|---|---|"]
    for name in ["mlp", "gnn", "logreg", "majority"]:
        def cell(r):
            b = report["models"][name].get(r)
            return f"{b['macro_f1']:.3f}" if b else "-"
        lines.append(f"| {name} | {cell('test_S0')} | {cell('test_S1')} | {cell('test_S2')} |")
    open(os.path.join(artifacts, "report.md"), "w").write("\n".join(lines) + "\n")
    print("[done] wrote metrics.json, report.md, curve_*.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
