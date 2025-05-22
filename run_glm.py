"""
run_glm.py

End-to-end driver that

  1. reads a YAML config (see example in configs/glm/)
  2. builds a long/tidy table via utils.build_long_df
  3. fits the requested GLM
  4. saves the table + summary + pickled model

Compatible with the *new* config keys:

root_dir
model_variants:               # list of str OR list of {name, take}
merge_bias_into_base: bool
use_bias_factor      : bool
dependent: {kind, metric, …}
glm: {formula, cov_type}
outfile_prefix
"""

from __future__ import annotations
import sys, pickle, re
from pathlib import Path
import yaml, pandas as pd, statsmodels.formula.api as smf
import utils


# ---------------------------------------------------------------------
def clean_formula(formula: str, drop_bias: bool) -> str:
    """Remove include_bias terms if the column is not present."""
    if drop_bias:
        # crude but safe: remove any token that contains 'include_bias'
        # and tidy multiple '+' or '~' that may result.
        formula = re.sub(r"\binclude_bias\b", "", formula)
        formula = re.sub(r"\+\s*\+", "+", formula)
        formula = formula.replace("~ +", "~").replace("+ ~", "~")
        formula = re.sub(r"\s+", " ", formula).strip()
    return formula


# ---------------------------------------------------------------------
def main(cfg_file: str):
    cfg = yaml.safe_load(open(cfg_file))

    root_dir   = cfg.get("root_dir", "data/haupt_stim_activ/damaged")
    variants   = cfg["model_variants"]               # required
    use_bias   = bool(cfg.get("use_bias_factor", True))
    merge_bias = bool(cfg.get("merge_bias_into_base", True))
    dep_cfg    = cfg["dependent"]
    glm_cfg    = cfg["glm"]
    prefix     = cfg.get("outfile_prefix", "glm_run")

    # 1) sanity-check that every requested folder exists
    for entry in variants:
        name_on_disk = entry if isinstance(entry, str) else entry["name"]
        full = Path(root_dir) / name_on_disk
        if not full.exists():
            sys.exit(f"❌  Data folder '{full}' does not exist.")

    # 2) build long DataFrame
    print("→ Building long/tidy table …")
    df = utils.build_long_df(
        root_dir, variants, dep_cfg,
        merge_bias_into_base = merge_bias,
        use_bias_factor      = use_bias
    )

    repo = Path(__file__).resolve().parent
    data_dir = repo / "data" / "derived"
    data_dir.mkdir(parents=True, exist_ok=True)
    kind_tag = dep_cfg["kind"]
    long_path = data_dir / f"{prefix}_{kind_tag}_long.parquet"
    df.to_parquet(long_path)
    print(f"   saved {len(df):,} rows → {long_path.relative_to(repo)}")

    # 3) fit GLM
    formula = clean_formula(glm_cfg["formula"], drop_bias = not use_bias)
    cov     = glm_cfg.get("cov_type", "HC3")
    print(f"→ Fitting OLS   [{cov}]")
    model = smf.ols(formula, data=df).fit(cov_type=cov)

    # 4) save artefacts
    glm_dir = repo / "glm_results"; glm_dir.mkdir(exist_ok=True)
    summary_path = glm_dir / f"{prefix}_summary.txt"
    model_path   = glm_dir / f"{prefix}_model.pkl"

    with open(summary_path, "w") as fh:
        fh.write(model.summary().as_text())
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)

    print("✓ Done:")
    print("  ▶", summary_path.relative_to(repo))
    print("  ▶", model_path.relative_to(repo))


# ---------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python run_glm.py <config.yaml>")
    main(sys.argv[1])
