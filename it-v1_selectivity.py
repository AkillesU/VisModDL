#!/usr/bin/env python3
"""
it2v1_contribution.py  ·  patched
──────────────────────────────────
Handles *both* IT conv layers (conv_input & conv1) and works even when
Conv2d modules no longer store `.output` after forward().
"""

from __future__ import annotations
import yaml, pathlib, sys, math, torch, torch.nn as nn
import torchvision.transforms as T
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# ─────────── helpers ────────────
def load_model(mcfg, device):
    import cornet
    ctor = {"cornet_rt": cornet.cornet_rt,
            "cornet_s":  cornet.cornet_s,
            "cornet_z":  cornet.cornet_z}[mcfg["name"].lower()]
    kwargs = {"pretrained": True}
    if mcfg["name"].lower() == "cornet_rt":
        kwargs["times"] = mcfg.get("time_steps", 5)
    return ctor(**kwargs).to(device).eval()

def build_transform():
    return T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

def iter_imgs(folder):
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in {".jpg",".jpeg",".png"}:
            yield Image.open(p).convert("RGB")

# receptive-field centre in input pixels for stride-4, 56×56 V1 map
def rf_px(fy, fx, stride=4):          # CORnet-RT uses stride-4 at V1 output
    return ((fy+0.5)*stride-0.5, (fx+0.5)*stride-0.5)

# ─────────── main ────────────────
def main(cfg_path):
    cfg   = yaml.safe_load(open(cfg_path, "r"))
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) top-N selective IT filters
    cat = cfg["category"].lower()
    # Read selectivity file
    sel = pd.read_csv(pathlib.Path(cfg["selectivity_csv_dir"])
                      / f"{cat}_unit_selectivity.csv")
    # Select IT rows
    it_rows = sel[sel.layer_name.str.contains(r"module\.IT\.conv_input|module\.IT\.conv1")]
    # after you’ve filtered `it_rows` to only conv_input & conv1 …
    total_it = len(it_rows)
    k        = max(1, int(cfg["top_frac"] * total_it))    # e.g. cfg["top_frac"] = 0.1 for top 10%
    topN     = it_rows.nlargest(k, "scaled_activation")
    top_filters = [(r.layer_name, int(r.channel)) for _, r in topN.iterrows()]
    print(f"Chosen {len(top_filters)} IT filters ({cat})")

    # 2) model + hooks ---------------------------------------------------
    model = load_model(cfg["model"], device)

    v1 = dict(model.named_modules())["module.V1.conv_input"]
    it_conv_in  = dict(model.named_modules())["module.IT.conv_input"]
    it_conv1    = dict(model.named_modules())["module.IT.conv1"]
    it_layers   = {"module.IT.conv_input": it_conv_in,
                   "module.IT.conv1":      it_conv1}

    # capture forward outputs
    layer_out = {}
    def save(name):
        def _hook(_, __, out): layer_out[name] = out
        return _hook
    v1.register_forward_hook(save("V1"))
    for name, mod in it_layers.items():
        mod.register_forward_hook(save(name))

    # 3) build small image batch
    tfm = build_transform()
    imgs = [tfm(img) for img in iter_imgs(pathlib.Path(cfg["category_images"]))][:20]
    x = torch.stack(imgs).to(device)

    # 4) forward + custom loss ------------------------------------------
    model.zero_grad()
    _ = model(x)                                    # fills layer_out[…]

    target = 0.0
    for lname, mod in it_layers.items():
        if lname not in layer_out: continue         # safety
        mask = torch.zeros_like(layer_out[lname])
        for (lay, ch) in top_filters:
            if lay == lname: mask[:, ch] = 1.0
        if mask.sum(): target = target + (layer_out[lname] * mask).sum() / x.size(0)


    target.backward()                               # populate v1.weight.grad

    # 5) contribution score per V1 filter -------------------------------
    g = v1.weight.grad.abs()
    g = g.cpu()
    contrib = g.view(v1.out_channels, -1).mean(1).numpy()
    contrib /= x.size(0)    # divide by number of images


    # 6) eccentricity px for each V1 filter -----------------------------
    fmap = layer_out["V1"].mean(0).detach().cpu()      # (C,56,56)

    # compute per-channel Y- and X-centers
    ys = (fmap * torch.arange(56).view(1, -1, 1)).sum((1, 2)) \
         / fmap.sum((1, 2))
    xs = (fmap * torch.arange(56).view(1, 1, -1)).sum((1, 2)) \
         / fmap.sum((1, 2))

    # make them plain Python floats
    ys = ys.numpy().tolist()
    xs = xs.numpy().tolist()

    stride = 4
    center = 224/2 - 0.5   # = 111.5

    ecc = []
    for y, x in zip(ys, xs):
        cy, cx = rf_px(y, x, stride)
        ecc.append(math.hypot(cy - center, cx - center))

    # 7) outputs ---------------------------------------------------------
    outdir = pathlib.Path(cfg["output_dir"]); outdir.mkdir(exist_ok=True)
    pd.DataFrame({
        "filter": np.arange(v1.out_channels),
        "contribution": contrib,
        "eccentricity_px": ecc
    }).to_csv(outdir/f"{cat}_V1_contributions.csv", index=False)

    pfi = float((contrib*np.array(ecc)).sum() / (contrib.sum()*max(ecc)))
    print(f"Peripheral-Foveal index: {pfi:.3f}")

    # heat-map
    fmap_contrib = (g.sum((1,2,3))[:, None, None] * fmap).sum(0)
    heat = torch.nn.functional.interpolate(
        fmap_contrib[None,None], size=224, mode="bilinear",
        align_corners=False)[0,0].numpy()
    heat = (heat-heat.min())/(heat.ptp()+1e-9)

    plt.figure(figsize=(4,4)); plt.imshow(heat, cmap="hot"); plt.axis("off")
    plt.title(f"V1 → IT ({cat}) - PF index = {pfi:.3f} \n {len(topN)} filters ({100*cfg['top_frac']} %)"); plt.tight_layout()
    plt.savefig(outdir/f"{cat}_V1_heatmap_{len(topN)}.png", dpi=300)
    plt.show()
    print("Saved CSV and heat-map to", outdir)

if __name__ == "__main__":
    if len(sys.argv)!=2:
        sys.exit("usage: python it2v1_contribution.py <config.yaml>")
    main(sys.argv[1])
