"""run_glm.py

End‑to‑end driver that reads parameters from a YAML configuration file,
builds the long‑format within‑between selectivity table, fits a GLM, and
writes all artefacts to disk.

Default configuration file: ``configs/glm/default.yaml``

YAML keys
---------
root_dir                 : path produced by ``run_damage`` (string)
model_variant            : sub‑folder with the model results (string)
formula                  : statsmodels‑compatible formula (string)
include_activation_layer : whether to treat recording layer as a fixed factor (bool)
outfile_prefix           : prefix for files under ``glm_results`` (string)

The script writes
  data/derived/selectivity_long.parquet
  glm_results/<prefix>_summary.txt
  glm_results/<prefix>_model.pkl
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml
import utils


def main(cfg_file):

    # ------------------------------------------------------------------
    # 1. read YAML config
    # ------------------------------------------------------------------
    cfg = yaml.safe_load(open(cfg_file))

    root_dir: Path = Path(cfg.get("root_dir", "data/haupt_stim_activ/damaged"))
    model_variant: str = cfg.get("model_variant", "cornet_rt")
    formula: str = cfg.get(
        "formula",
        "observed_difference ~ damage_level * damage_type * damage_layer * category",
    )
    include_activation_layer: bool = bool(cfg.get("include_activation_layer", False))
    outfile_prefix: str = cfg.get("outfile_prefix", "glm_run")

    full_root = root_dir / model_variant
    if not full_root.exists():
        sys.exit(f"Data directory '{full_root}' does not exist.")

    repo_root = Path(__file__).resolve().parent
    glm_dir = repo_root / "glm_results"
    data_dir = repo_root / "data" / "derived"
    glm_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. build long DataFrame
    # ------------------------------------------------------------------
    print("Building long‑format selectivity table …")
    df_long = utils.build_selectivity_dataframe(
        str(full_root), include_activation_layer=include_activation_layer
    )

    long_path = data_dir / f"selectivity_long.parquet"
    df_long.to_parquet(long_path)
    print(f"Saved long table → {long_path.relative_to(repo_root)}  ({len(df_long)} rows)")

    # ------------------------------------------------------------------
    # 3. fit GLM
    # ------------------------------------------------------------------
    print("Fitting GLM …")
    glm_res = utils.fit_selectivity_glm(df_long, formula=formula)

    # ------------------------------------------------------------------
    # 4. save outputs
    # ------------------------------------------------------------------
    summary_path = glm_dir / f"{outfile_prefix}_summary.txt"
    with open(summary_path, "w") as fh:
        fh.write(glm_res.summary().as_text())

    model_path = glm_dir / f"{outfile_prefix}_model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(glm_res, fh)

    print("GLM results written:")
    print(f"  {summary_path.relative_to(repo_root)}")
    print(f"  {model_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main(sys.argv[1])
