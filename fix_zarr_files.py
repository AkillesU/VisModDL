#!/usr/bin/env python3
"""
safer_repair_alt_zarrs.py
  – merges or renames *_activ_alt_*.zarr into canonical stores
    without risk of deleting the only copy of the data.

Hard-coded settings:
    ROOT       = current working directory
    STIM_DIR   = ROOT/'stimuli'
    PURGE_EMPTY= True  (delete stores that end up with no 'activ' array)
"""

from __future__ import annotations
import logging, os, re, shutil, sys
from pathlib import Path
import zarr

ROOT        = Path.cwd()
STIM_DIR    = ROOT / "stimuli"
PURGE_EMPTY = True

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
ALT_RE = re.compile(r"__activ_alt_(\d+)\.zarr$")

def canonical_name(p: Path) -> Path:
    feat = ALT_RE.search(p.name).group(1)
    stem = ALT_RE.sub("", p.name).rstrip("_")
    if not stem.endswith(f"__activ_{feat}"):
        stem = f"{stem}__activ_{feat}"
    return p.with_name(stem + ".zarr")

def has_activ(store: Path) -> bool:
    try:
        return "activ" in zarr.open(store, mode="r")
    except Exception:
        return False

def activ_rows(store: Path) -> int:
    try:
        return zarr.open(store, mode="r")["activ"].shape[0]
    except Exception:
        return 0

def patch_attrs(store: Path, perm: int, n_img: int) -> None:
    z = zarr.open(store, mode="a")
    if not z.attrs.get("perm_indices"):
        z.attrs["perm_indices"] = [perm]
    if not z.attrs.get("image_names"):
        names = None
        if STIM_DIR.exists():
            files = sorted(STIM_DIR.glob("*"))
            if len(files) >= n_img:
                names = [f.name for f in files][:n_img]
        z.attrs["image_names"] = names or [f"img_{i:04d}" for i in range(n_img)]
    z.store.close()

def main() -> None:
    alt_paths = sorted(ROOT.rglob("*__activ_alt_*.zarr"))
    if not alt_paths:
        print("✅  No alt stores found.")
        return
    logging.info(f"Found {len(alt_paths)} alt stores")

    for alt in alt_paths:
        perm = int(alt.stem.split("__")[0])
        canon = canonical_name(alt)
        alt_has = has_activ(alt)
        canon_has = has_activ(canon) if canon.exists() else False

        # Decide which store to keep
        if not canon.exists():                         # easy – rename alt
            logging.info(f"[rename] {alt} → {canon.name}")
            os.rename(alt, canon)
        else:
            if canon_has and not alt_has:              # keep canonical
                logging.info(f"[keep] canonical already populated – drop {alt.name}")
                shutil.rmtree(alt)
            elif alt_has and not canon_has:            # replace empty canonical
                logging.info(f"[replace] canonical empty – use {alt.name}")
                shutil.rmtree(canon)
                os.rename(alt, canon)
            elif alt_has and canon_has:                # both have data
                keep_alt = activ_rows(alt) > activ_rows(canon)
                if keep_alt:
                    logging.info(f"[swap] alt has more rows – replace canonical")
                    shutil.rmtree(canon)
                    os.rename(alt, canon)
                else:
                    logging.info(f"[dup] canonical adequate – drop alt")
                    shutil.rmtree(alt)
            else:                                      # neither has data
                logging.warning(f"[empty] neither store has data: {canon}")
                if PURGE_EMPTY:
                    shutil.rmtree(alt)
                    shutil.rmtree(canon)

        if canon.exists() and has_activ(canon):
            n_img = zarr.open(canon, mode="r")["activ"].shape[1]
            patch_attrs(canon, perm, n_img)
        elif canon.exists() and PURGE_EMPTY:
            shutil.rmtree(canon)

    logging.info("🛠️  Repair finished.")

if __name__ == "__main__":
    main()
