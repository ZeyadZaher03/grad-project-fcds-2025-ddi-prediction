# ml/assemble.py
"""Unified pair-feature assembly shared by training, evaluation, and serving."""
from __future__ import annotations
import numpy as np
from ml.features import light_drug_vector, pair_features, LIGHT_DESCRIPTORS
from ml.bio_features import bio_pair_features, N_BIO


def struct_dim(cfg: dict) -> int:
    f = cfg["features"]
    if f["structural"] == "light":
        return int(f["light_fp_bits"]) + len(LIGHT_DESCRIPTORS)
    if f["structural"] == "chemberta":
        from ml.embeddings import embedding_dim
        return embedding_dim()
    raise ValueError(f"unknown structural config: {f['structural']}")


def drug_struct_vector(smiles: str, cfg: dict) -> np.ndarray:
    f = cfg["features"]
    if f["structural"] == "light":
        return light_drug_vector(smiles, fp_bits=int(f["light_fp_bits"]), radius=int(f["fp_radius"]))
    if f["structural"] == "chemberta":
        from ml.embeddings import embed_smiles, embedding_dim
        try:
            return embed_smiles(smiles)
        except Exception:
            return np.zeros((embedding_dim(),), dtype=np.float32)
    raise ValueError(f"unknown structural config: {f['structural']}")


def assemble_pair(struct_a, struct_b, bio_a, bio_b, use_bio: bool) -> np.ndarray:
    """struct_a/b: per-drug structural vectors; bio_a/b: bio-set dicts."""
    blocks = [pair_features(struct_a, struct_b)]
    if use_bio:
        blocks.append(bio_pair_features(bio_a, bio_b))
    return np.concatenate(blocks).astype(np.float32)


def pair_dim(cfg: dict) -> int:
    d = 3 * struct_dim(cfg)
    if cfg["features"].get("use_bio"):
        d += N_BIO
    return d
