#!/usr/bin/env python3
"""
it2v1_contribution_units.py  ·  updated

Selects top-fraction of IT *units* (not filters), backprops their activations to V1,
computes per-filter contribution scores, and relates them to V1 eccentricities.
"""

import sys
import yaml
import pathlib
import math
import torch
import torch.nn as nn
import torchvision.transforms as T
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# ─────────── helpers ────────────
def load_model(mcfg, device):
    import cornet
    ctor = {
        "cornet_rt": cornet.cornet_rt,
        "cornet_s": cornet.cornet_s,
        "cornet_z": cornet.cornet_z
    }[mcfg["name"].lower()]
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

# receptive-field centre for stride-4, 56×56 V1 map
def rf_px(fy, fx, stride=4):
    return ((fy+0.5)*stride - 0.5, (fx+0.5)*stride - 0.5)

# ─────────── main ────────────────
def main(cfg_path):
    cfg    = yaml.safe_load(open(cfg_path, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) select top-fraction of IT units
    cat = cfg["category"].lower()
    # load per-unit selectivity CSV (unit_id format: layer:channel:y:x)
    sel_file = pathlib.Path(cfg["selectivity_csv_dir"]) / f"{cat}_unit_selectivity_all_units.csv"
    sel = pd.read_csv(sel_file)
    # keep only IT layers
    it_pattern = r"module\.IT\.conv_input|module\.IT\.conv1"
    it_rows = sel[sel.layer_name.str.contains(it_pattern, regex=True)]
    total_units = len(it_rows)
    k = max(1, int(cfg.get("top_frac", 0.1) * total_units))
    topN = it_rows.nlargest(k, "scaled_activation")
    # parse unit_id to (layer, c, y, x)
    top_units = []
    for _, r in topN.iterrows():
        lay = r.layer_name
        # unit_id is "layer:channel:y:x"
        _, c, y, x = r.unit_id.split(":")
        top_units.append((lay, int(c), int(y), int(x)))
    print(f"Chosen {len(top_units)} IT units ({100*cfg.get('top_frac',0.1):.1f}% of {total_units})")

    # 2) load model & register hooks
    model = load_model(cfg["model"], device)
    v1 = dict(model.named_modules())["module.V1.conv_input"]
    it_conv_in = dict(model.named_modules())["module.IT.conv_input"]
    it_conv1   = dict(model.named_modules())["module.IT.conv1"]
    it_layers  = {"module.IT.conv_input": it_conv_in,
                  "module.IT.conv1":      it_conv1}

    layer_out = {}
    def save(name):
        def _hook(_, __, out):
            layer_out[name] = out
        return _hook

    v1.register_forward_hook(save("V1"))
    for name, mod in it_layers.items():
        mod.register_forward_hook(save(name))

    # 3) load images
    tfm = build_transform()
    imgs = [tfm(img) for img in iter_imgs(pathlib.Path(cfg["category_images"]))][:20]
    x = torch.stack(imgs).to(device)

    # 4) forward + build loss from top units
    model.zero_grad()
    _ = model(x)

    target = torch.tensor(0., device=device)
    for lname in it_layers:
        out = layer_out.get(lname)
        if out is None:
            continue
        # out shape: [B, C, H, W]
        mask = torch.zeros_like(out)
        for lay, c, y, x_ in top_units:
            if lay == lname:
                mask[:, c, y, x_] = 1.0
        if mask.sum().item() > 0:
            # average over images
            target = target + (out * mask).sum() / x.size(0)

    target.backward()

    # 5) per-V1 filter contribution
    g = v1.weight.grad.abs().cpu()
    contrib = g.view(v1.out_channels, -1).mean(1).numpy()
    contrib /= x.size(0)

    # 6) eccentricities
    fmap = layer_out["V1"].mean(0).cpu()  # [C,56,56]
    ys = (fmap * torch.arange(fmap.shape[1]).view(1,-1,1)).sum((1,2)) / fmap.sum((1,2))
    xs = (fmap * torch.arange(fmap.shape[2]).view(1,1,-1)).sum((1,2)) / fmap.sum((1,2))
    ys, xs = ys.tolist(), xs.tolist()
    center = 224/2 - 0.5
    ecc = [math.hypot(*rf_px(y,x, stride=4), center, center) for y,x in zip(ys,xs)]

    # 7) save results
    outdir = pathlib.Path(cfg["output_dir"])
    outdir.mkdir(exist_ok=True)
    pd.DataFrame({
        "filter": np.arange(v1.out_channels),
        "contribution": contrib,
        "eccentricity_px": ecc
    }).to_csv(outdir/f"{cat}_V1_contributions.csv", index=False)

    pfi = ((contrib * np.array(ecc)).sum() / (contrib.sum() * max(ecc))).item()
    print(f"Peripheral-Foveal index: {pfi:.3f}")

    # heatmap
    fmap_contrib = (g.sum((1,2,3))[:,None,None] * fmap).sum(0)
    heat = torch.nn.functional.interpolate(
        fmap_contrib[None,None], size=224, mode="bilinear", align_corners=False
    )[0,0].detach().numpy()
    heat = (heat - heat.min()) / (heat.ptp() + 1e-9)

    plt.figure(figsize=(4,4))
    plt.imshow(heat, cmap="hot"); plt.axis("off")
    plt.title(
        f"V1 → IT ({cat}) - PF index = {pfi:.3f}\n{len(top_units)} units ({100*cfg.get('top_frac',0.1):.1f}%)"
    )
    plt.tight_layout()
    plt.savefig(outdir/f"{cat}_V1_heatmap_{len(top_units)}.png", dpi=300)
    plt.show()
    print("Saved CSV and heat-map to", outdir)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python it2v1_contribution_units.py <config.yaml>")
    main(sys.argv[1])
