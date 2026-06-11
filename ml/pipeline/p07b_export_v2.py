# ml/pipeline/p07b_export_v2.py
"""Stage 7b: export two-stage models + metadata + SMILES->bio lookup for serving."""
from __future__ import annotations
import os, json, shutil
import pandas as pd
from rdkit import Chem
import torch
from ml.config import load_config
from ml.common import CLASSES
from ml.assemble import pair_dim
from ml.bio_features import load_bio_sets, ENTITY_TYPES
from ml.pipeline.p05b_train_stages import SEV3


def build_v2_metadata(structural, use_bio, light_fp_bits, fp_radius, pair_dim, hidden,
                      interaction_threshold):
    return {
        "version": 2,
        "classes": CLASSES,
        "severity_classes": SEV3,
        "structural": structural,
        "use_bio": use_bio,
        "light_fp_bits": light_fp_bits,
        "fp_radius": fp_radius,
        "pair_dim": pair_dim,
        "hidden": hidden,
        "interaction_threshold": interaction_threshold,
    }


def _canonical(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m) if m is not None else None


def main():
    cfg = load_config()
    artifacts = cfg["paths"]["artifacts_dir"]
    raw = cfg["paths"]["raw_dir"]
    export = os.path.join(artifacts, "export_v2"); os.makedirs(export, exist_ok=True)
    use_bio = cfg["features"]["use_bio"]
    tag = cfg["features"]["structural"] + ("_bio" if use_bio else "")

    for stage in ["stage1", "stage2"]:
        shutil.copy(os.path.join(artifacts, f"model_{stage}_{tag}.pt"),
                    os.path.join(export, f"model_{stage}.pt"))

    md = build_v2_metadata(
        structural=cfg["features"]["structural"], use_bio=use_bio,
        light_fp_bits=int(cfg["features"]["light_fp_bits"]), fp_radius=int(cfg["features"]["fp_radius"]),
        pair_dim=pair_dim(cfg), hidden=int(cfg["train"]["hidden"]), interaction_threshold=0.5,
    )
    json.dump(md, open(os.path.join(export, "metadata.json"), "w"), indent=2)

    drugs = pd.read_csv(os.path.join(raw, "drugs.csv")).drop_duplicates("drug_id")
    bio = load_bio_sets(cfg["bio"]["merged_csv"])
    lookup = {}
    for did, smi in zip(drugs["drug_id"], drugs["smiles"].astype(str)):
        cs = _canonical(smi)
        if cs and did in bio:
            lookup[cs] = {t: sorted(bio[did][t]) for t in ENTITY_TYPES}
    json.dump(lookup, open(os.path.join(export, "bio_lookup.json"), "w"))
    print(f"[done] exported two-stage ({tag}) + metadata + bio_lookup ({len(lookup)} drugs) -> {export}")


if __name__ == "__main__":
    main()
