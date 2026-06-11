# ml/embeddings.py
"""Optional ChemBERTa molecular embeddings. Heavy deps imported lazily."""
from __future__ import annotations
import numpy as np

_MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
_cache = {}


def chemberta_available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _load():
    if "model" not in _cache:
        import torch
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
        mdl = AutoModel.from_pretrained(_MODEL_NAME).eval()
        _cache["tok"], _cache["model"], _cache["torch"] = tok, mdl, torch
    return _cache["tok"], _cache["model"], _cache["torch"]


def embed_smiles(smiles: str) -> np.ndarray:
    """Mean-pooled last-hidden-state embedding (float32). Zeros if invalid/unavailable."""
    if not chemberta_available():
        raise RuntimeError("transformers/torch not installed; cannot use chemberta config")
    tok, mdl, torch = _load()
    with torch.no_grad():
        enc = tok(str(smiles), return_tensors="pt", truncation=True, max_length=256)
        out = mdl(**enc).last_hidden_state  # (1, L, H)
        mask = enc["attention_mask"].unsqueeze(-1)  # (1, L, 1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled.squeeze(0).cpu().numpy().astype(np.float32)


def embedding_dim() -> int:
    _, mdl, _ = _load()
    return int(mdl.config.hidden_size)
