# ml/pipeline/p05_train.py
"""Stage 5: train PairMLP4 and SageDDI on the same splits; save models + curves."""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from ml.config import load_config
from ml.common import set_seed, CLASS_TO_IDX, CLASSES
from ml.features import pair_features
from ml.models import PairMLP4, SageDDI


def _load(cfg):
    proc = cfg["paths"]["processed_dir"]
    splits = pd.read_parquet(os.path.join(proc, "splits.parquet"))
    feat = np.load(os.path.join(proc, "drug_features.npz"), allow_pickle=True)
    ids = list(feat["ids"])
    idx = {d: i for i, d in enumerate(ids)}
    X = feat["X"].astype(np.float32)
    splits = splits[splits.drugA_id.isin(idx) & splits.drugB_id.isin(idx)].reset_index(drop=True)
    splits["y"] = splits["label"].map(CLASS_TO_IDX)
    return splits, X, idx


def _class_weights(y, n=4):
    counts = np.bincount(y, minlength=n).astype(np.float32)
    w = counts.sum() / (n * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def _mlp_matrix(splits, X, idx, mask):
    rows = splits[mask]
    feats = np.stack([pair_features(X[idx[a]], X[idx[b]])
                      for a, b in zip(rows.drugA_id, rows.drugB_id)])
    return torch.tensor(feats), torch.tensor(rows["y"].values, dtype=torch.long)


def train_mlp(cfg, splits, X, idx, artifacts):
    set_seed(cfg["seed"])
    Xtr, ytr = _mlp_matrix(splits, X, idx, splits.split == "train")
    Xva, yva = _mlp_matrix(splits, X, idx, splits.split == "val")
    model = PairMLP4(in_dim=Xtr.shape[1], hidden=cfg["train"]["hidden"],
                     dropout=cfg["train"]["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    w = _class_weights(ytr.numpy())
    bs = cfg["train"]["batch_size"]
    best_f1, best_state, curve, bad = -1, None, [], 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[b]), ytr[b], weight=w)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            tr_loss = F.cross_entropy(model(Xtr), ytr, weight=w).item()
            va_logits = model(Xva)
            va_loss = F.cross_entropy(va_logits, yva, weight=w).item()
            va_f1 = f1_score(yva.numpy(), va_logits.argmax(1).numpy(), average="macro")
        curve.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_macro_f1": va_f1})
        if va_f1 > best_f1:
            best_f1, best_state, bad = va_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg["train"]["patience"]:
                break
    model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict(), "in_dim": Xtr.shape[1],
                "kind": "mlp"}, os.path.join(artifacts, "model_mlp.pt"))
    json.dump(curve, open(os.path.join(artifacts, "curve_mlp.json"), "w"))
    print(f"[mlp] best val macro-F1={best_f1:.4f} over {len(curve)} epochs")


def train_gnn(cfg, splits, X, idx, artifacts):
    set_seed(cfg["seed"])
    x = torch.tensor(X, dtype=torch.float32)
    tr_pos = splits[(splits.split == "train") & (splits.label != "None")]
    ei = torch.tensor([[idx[a] for a in tr_pos.drugA_id] + [idx[b] for b in tr_pos.drugB_id],
                       [idx[b] for b in tr_pos.drugB_id] + [idx[a] for a in tr_pos.drugA_id]],
                      dtype=torch.long)
    def pidx(mask):
        r = splits[mask]
        return (torch.tensor([[idx[a] for a in r.drugA_id], [idx[b] for b in r.drugB_id]],
                             dtype=torch.long),
                torch.tensor(r["y"].values, dtype=torch.long))
    ptr, ytr = pidx(splits.split == "train")
    pva, yva = pidx(splits.split == "val")
    model = SageDDI(in_channels=x.shape[1], hidden=cfg["train"]["gnn_hidden"],
                    dropout=cfg["train"]["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    w = _class_weights(ytr.numpy())
    best_f1, best_state, curve, bad = -1, None, [], 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(x, ei, ptr), ytr, weight=w)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va_logits = model(x, ei, pva)
            va_loss = F.cross_entropy(va_logits, yva, weight=w).item()
            va_f1 = f1_score(yva.numpy(), va_logits.argmax(1).numpy(), average="macro")
        curve.append({"epoch": epoch, "train_loss": loss.item(), "val_loss": va_loss, "val_macro_f1": va_f1})
        if va_f1 > best_f1:
            best_f1, best_state, bad = va_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg["train"]["patience"]:
                break
    model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict(), "in_channels": x.shape[1],
                "hidden": cfg["train"]["gnn_hidden"], "kind": "gnn"},
               os.path.join(artifacts, "model_gnn.pt"))
    json.dump(curve, open(os.path.join(artifacts, "curve_gnn.json"), "w"))
    print(f"[gnn] best val macro-F1={best_f1:.4f} over {len(curve)} epochs")


def main():
    cfg = load_config()
    artifacts = cfg["paths"]["artifacts_dir"]
    os.makedirs(artifacts, exist_ok=True)
    splits, X, idx = _load(cfg)
    train_mlp(cfg, splits, X, idx, artifacts)
    train_gnn(cfg, splits, X, idx, artifacts)


if __name__ == "__main__":
    main()
