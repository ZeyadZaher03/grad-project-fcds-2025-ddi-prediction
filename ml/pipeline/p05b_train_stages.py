# ml/pipeline/p05b_train_stages.py
"""Stage 5b: train binary interaction (stage 1) and 3-class severity (stage 2) models."""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from ml.config import load_config
from ml.common import set_seed
from ml.assemble import assemble_pair, drug_struct_vector, struct_dim, pair_dim
from ml.bio_features import empty_bio_sets, ENTITY_TYPES
from ml.models import MLPHead

SEV3 = ["Minor", "Moderate", "Major"]
SEV3_IDX = {c: i for i, c in enumerate(SEV3)}


def _load_struct(cfg):
    proc = cfg["paths"]["processed_dir"]
    tag = cfg["features"]["structural"]
    feat = np.load(os.path.join(proc, f"struct_{tag}.npz"), allow_pickle=True)
    ids = list(feat["ids"]); idx = {d: i for i, d in enumerate(ids)}
    return feat["X"].astype(np.float32), idx


def _load_bio(cfg):
    proc = cfg["paths"]["processed_dir"]
    raw = json.load(open(os.path.join(proc, "bio_sets.json")))
    return {d: {t: set(v.get(t, [])) for t in ENTITY_TYPES} for d, v in raw.items()}


def _drug_regime(cfg):
    proc = cfg["paths"]["processed_dir"]
    sp = pd.read_parquet(os.path.join(proc, "splits.parquet"))
    test_drugs = set(pd.unique(
        sp[sp.split.isin(["test_S1", "test_S2"])][["drugA_id", "drugB_id"]].values.ravel()))
    train_drugs = set(pd.unique(
        sp[sp.split.isin(["train", "val", "test_S0"])][["drugA_id", "drugB_id"]].values.ravel()))
    return train_drugs, test_drugs


def _matrix(pairs, X, idx, bio, cfg, use_bio):
    rows = []
    for a, b in zip(pairs["drugA_id"], pairs["drugB_id"]):
        rows.append(assemble_pair(X[idx[a]], X[idx[b]],
                                  bio.get(a, empty_bio_sets()), bio.get(b, empty_bio_sets()),
                                  use_bio=use_bio))
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def _train_loop(model, Xtr, ytr, Xva, yva, cfg, loss_fn, score_fn):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    bs = cfg["train"]["batch_size"]
    best, best_state, curve, bad = -1, None, [], 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]; opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            sc = score_fn(model, Xva, yva)
            tl = loss_fn(model(Xtr), ytr).item(); vl = loss_fn(model(Xva), yva).item()
        curve.append({"epoch": epoch, "train_loss": tl, "val_loss": vl, "val_score": sc})
        if sc > best:
            best, best_state, bad = sc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg["train"]["patience"]:
                break
    model.load_state_dict(best_state)
    return best, curve


def train_stage1(cfg, X, idx, bio, artifacts):
    set_seed(cfg["seed"])
    proc = cfg["paths"]["processed_dir"]
    pairs = pd.read_parquet(os.path.join(proc, "interaction_pairs.parquet"))
    train_drugs, test_drugs = _drug_regime(cfg)
    both_train = pairs[pairs.drugA_id.isin(train_drugs) & pairs.drugB_id.isin(train_drugs)].reset_index(drop=True)
    rng = np.random.default_rng(cfg["seed"]); u = rng.random(len(both_train))
    tr, va = both_train[u >= 0.1], both_train[u < 0.1]
    use_bio = cfg["features"]["use_bio"]
    Xtr = _matrix(tr, X, idx, bio, cfg, use_bio); ytr = torch.tensor(tr["y"].values, dtype=torch.float32).unsqueeze(1)
    Xva = _matrix(va, X, idx, bio, cfg, use_bio); yva = torch.tensor(va["y"].values, dtype=torch.float32).unsqueeze(1)
    model = MLPHead(in_dim=pair_dim(cfg), n_out=1, hidden=cfg["train"]["hidden"], dropout=cfg["train"]["dropout"])
    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    loss_fn = lambda logit, y: F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_w)
    def score(m, Xv, yv):
        p = torch.sigmoid(m(Xv)).numpy().ravel()
        return roc_auc_score(yv.numpy().ravel(), p)
    best, curve = _train_loop(model, Xtr, ytr, Xva, yva, cfg, loss_fn, score)
    tag = cfg["features"]["structural"] + ("_bio" if use_bio else "")
    torch.save({"model_state_dict": model.state_dict(), "in_dim": pair_dim(cfg), "hidden": cfg["train"]["hidden"]},
               os.path.join(artifacts, f"model_stage1_{tag}.pt"))
    json.dump(curve, open(os.path.join(artifacts, f"curve_stage1_{tag}.json"), "w"))
    print(f"[stage1:{tag}] best val AUC={best:.4f} over {len(curve)} epochs")


def train_stage2(cfg, X, idx, bio, artifacts):
    set_seed(cfg["seed"])
    proc = cfg["paths"]["processed_dir"]
    sp = pd.read_parquet(os.path.join(proc, "splits.parquet"))
    pos = sp[(sp.label != "None") & (sp.split.isin(["train", "val"]))].copy()
    pos["sy"] = pos["label"].map(SEV3_IDX)
    tr, va = pos[pos.split == "train"], pos[pos.split == "val"]
    use_bio = cfg["features"]["use_bio"]
    Xtr = _matrix(tr, X, idx, bio, cfg, use_bio); ytr = torch.tensor(tr["sy"].values, dtype=torch.long)
    Xva = _matrix(va, X, idx, bio, cfg, use_bio); yva = torch.tensor(va["sy"].values, dtype=torch.long)
    model = MLPHead(in_dim=pair_dim(cfg), n_out=3, hidden=cfg["train"]["hidden"], dropout=cfg["train"]["dropout"])
    counts = np.bincount(ytr.numpy(), minlength=3).astype(np.float32)
    w = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = lambda logit, y: F.cross_entropy(logit, y, weight=w)
    def score(m, Xv, yv):
        return f1_score(yv.numpy(), m(Xv).argmax(1).numpy(), average="macro")
    best, curve = _train_loop(model, Xtr, ytr, Xva, yva, cfg, loss_fn, score)
    tag = cfg["features"]["structural"] + ("_bio" if use_bio else "")
    torch.save({"model_state_dict": model.state_dict(), "in_dim": pair_dim(cfg), "hidden": cfg["train"]["hidden"]},
               os.path.join(artifacts, f"model_stage2_{tag}.pt"))
    json.dump(curve, open(os.path.join(artifacts, f"curve_stage2_{tag}.json"), "w"))
    print(f"[stage2:{tag}] best val macro-F1={best:.4f} over {len(curve)} epochs")


def main():
    cfg = load_config()
    artifacts = cfg["paths"]["artifacts_dir"]; os.makedirs(artifacts, exist_ok=True)
    X, idx = _load_struct(cfg); bio = _load_bio(cfg)
    train_stage1(cfg, X, idx, bio, artifacts)
    train_stage2(cfg, X, idx, bio, artifacts)


if __name__ == "__main__":
    main()
