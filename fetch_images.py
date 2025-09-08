#!/usr/bin/env python3
"""
Robust Unsplash-API image downloader
—————————————————————————————————————
• Needs   : pip install requests pillow matplotlib tqdm python-dotenv
• Key     : set UNSPLASH_ACCESS_KEY env-var (free at https://unsplash.com/developers)
• Licence : Unsplash photographs are free to use, but you must provide attribution
            if you publish or redistribute them — see https://unsplash.com/license
"""
from __future__ import annotations
import os, io, time, pathlib, random, math
import requests, urllib3
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv

# ───────────────────────── Config ──────────────────────────
ROOT_DIR    = pathlib.Path("categ_images_2")
PER_CLASS   = 10          # images per category
BATCH       = 10           # <=30 for /photos/random
SIZE        = "small"    # Unsplash sizes: raw, full, regular, small, thumb
PAUSE       = 0.2          # polite delay between requests
SHOW_FIRST  = 5            # how many to preview

CATEGORIES  = {
    #"places" : "landscape house",
    #"faces"  : "face",
    "fruit": "fruit",
    "vehicle": "vehicle",
    "furniture": "furniture",
    "stationery": "stationery",
    "sports": "sports item",
    "instrument": "instrument",
    "clothes": "clothes",
    #"animals": "animal wildlife"
}

# ─────────────────────── Helpers ───────────────────────────
def make_session() -> requests.Session:
    retry = urllib3.Retry(
        total=5, backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("https://", adapter)
    key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not key:
        raise RuntimeError("Set the UNSPLASH_ACCESS_KEY environment variable.")
    s.headers.update({
        "Authorization": f"Client-ID {key}",
        "Accept-Version": "v1",
        "User-Agent": "python-image-downloader/1.0"
    })
    return s

def download_one(url: str, out_path: pathlib.Path, session: requests.Session) -> bool:
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="JPEG", quality=92)
        return True
    except Exception as exc:
        print(f"⚠️  {exc} -> {url}")
        return False

# ───────────────────────── Main ────────────────────────────
if __name__ == "__main__":
    load_dotenv()                      # allow .env file
    session = make_session()
    samples: dict[str, list[pathlib.Path]] = {c: [] for c in CATEGORIES}

    for cat, query in CATEGORIES.items():
        target_dir = ROOT_DIR / cat
        successes  = 0
        needed     = PER_CLASS
        print(f"\n⏬ Downloading {PER_CLASS} “{cat}” images…")
        # Work in batches to stay within Unsplash limits (≤30)
        batches = math.ceil(needed / BATCH)
        for _ in tqdm(range(batches), unit="batch"):
            count = min(BATCH, needed - successes)
            if count <= 0: break
            params = {"query": query, "count": count}
            try:
                r = session.get("https://api.unsplash.com/photos/random",
                                params=params, timeout=20)
                if r.status_code == 403:
                    raise RuntimeError("Unsplash API quota exhausted. Wait or upgrade.")
                r.raise_for_status()
                data = r.json()
                # random endpoint returns list when count>1
                if isinstance(data, dict):
                    data = [data]
                for item in data:
                    img_url = item["urls"][SIZE]
                    file_id = item["id"]
                    out_path = target_dir / f"{file_id}.jpg"
                    if download_one(img_url, out_path, session):
                        if len(samples[cat]) < SHOW_FIRST:
                            samples[cat].append(out_path)
                        successes += 1
                time.sleep(PAUSE)
            except requests.exceptions.RetryError:
                print("⏳ Too many retries. Backing off 30 s…")
                time.sleep(30)
            except Exception as e:
                print(f"⚠️  Batch failed: {e}")
        print(f"✅  Saved {successes}/{PER_CLASS} files to {target_dir}")

    # ─────────────── Quick visual sanity-check ──────────────
    print("\n🖼  Showing first 5 images from each category…")
    rows, cols = len(CATEGORIES), SHOW_FIRST
    fig, axes = plt.subplots(rows, cols, figsize=(16, 2.8*rows))
    for r, (cat, paths) in enumerate(samples.items()):
        for c in range(cols):
            ax = axes[r, c] if rows > 1 else axes[c]
            ax.axis("off")
            if c==0:
                ax.set_ylabel(cat, rotation=0, labelpad=50, fontsize=12)
            if c < len(paths):
                ax.imshow(Image.open(paths[c]))
    plt.tight_layout()
    plt.show()
