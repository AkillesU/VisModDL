#!/usr/bin/env python3
"""
it2v1_input_contrib.py  ·  V1-to-input contributon with PFI

For a top-fraction of IT units: 
1. Compute their activation-based loss.
2. Backpropagate to V1 feature-map outputs to get per-unit gradients.
3. Weight each V1 activation by its gradient (|grad * activation|) → per-unit importance.
4. Project each V1 spatial unit back to input-pixel coordinates via receptive-field centroids.
5. Splat per-unit importance into a 224×224 input-space importance map, normalize.
6. Compute Peripheral–Foveal Index of that map and plot a heatmap.
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

# compute receptive-field center in input pixels from feature-map coords
def rf_px(fy, fx, stride, padding, kernel):
    # center = (fy * stride + (kernel-1)/2 - padding)
    cy = fy * stride + (kernel-1)/2 - padding
    cx = fx * stride + (kernel-1)/2 - padding
    return cy, cx

# ─────────── main ────────────────
def main(cfg_path):
    # load config
    cfg = yaml.safe_load(open(cfg_path,'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) select IT top units
    cat = cfg["category"].lower()
    sel = pd.read_csv(pathlib.Path(cfg["selectivity_csv_dir"]) / f"{cat}_unit_selectivity_all_units.csv")
    it_pat = r"module\.IT\.conv_input|module\.IT\.conv1"
    it_rows = sel[sel.layer_name.str.contains(it_pat, regex=True)]
    k = max(1, int(cfg.get("top_frac",0.1) * len(it_rows)))
    topN = it_rows.nlargest(k, "scaled_activation")
    top_units = []
    for _,r in topN.iterrows():
        lay, c, y, x = r.layer_name, *map(int, r.unit_id.split(':')[1:])
        top_units.append((lay, c, y, x))
    print(f"Selected {len(top_units)} IT units.")

    # 2) load model and hooks
    model = load_model(cfg["model"], device)
    # identify V1 conv_input module
    v1 = dict(model.named_modules())["module.V1.conv_input"]
    # IT layers
    it_v1_names = ["module.IT.conv_input","module.IT.conv1", "module.V1.conv_input"]
    layer_out = {}
    layer_grad = {}
    # forward hook
    def save_out(name):
        def hook(_mod, _in, out):
            layer_out[name] = out        # keep it “live” for grad hooks
        return hook
    # backward hook for gradients on outputs
    def save_grad(name):
        def hook(grad): layer_grad[name]=grad
        return hook

    # register IT hooks
    for name,mod in model.named_modules():
        if name in it_v1_names:
            mod.register_forward_hook(save_out(name))
    # Instead of returning the handle, just register it and return None:
    def v1_hook(m, inp, out):
        # register a backward‐gradient hook on the actual tensor
        out.register_hook(save_grad("module.V1.conv_input"))
        # DO NOT return anything

    v1.register_forward_hook(v1_hook)

    # define target: sum activation of top IT units
    def target_fn():
        # now val is part of the graph from the start
        val = torch.tensor(0., device=device, requires_grad=True)
        for name,c,y,x in top_units:
            act = layer_out[name]
            val = val + act[:,c,y,x].sum()/act.size(0)
        return val

    # 4) process images, accumulate feature-map importance
    tfm  = build_transform()
    imgs = list(iter_imgs(pathlib.Path(cfg["category_images"])))[:cfg.get("max_images",20)]

    feat_imp = None  # will be [Hf,Wf]
    for img in tqdm(imgs, desc="Images"):
        x = tfm(img).unsqueeze(0).to(device)
        layer_out.clear(); layer_grad.clear()
        model.zero_grad(); _ = model(x)
        loss = target_fn(); loss.backward()
        v1_out  = layer_out["module.V1.conv_input"][0].detach().cpu()       # [C,Hf,Wf]
        v1_gr   = layer_grad["module.V1.conv_input"][0].cpu()              # [C,Hf,Wf]
        imp_cwh = (v1_out.abs() * v1_gr.abs()).sum(0).numpy()  # sum over C → [Hf,Wf]
        feat_imp = imp_cwh if feat_imp is None else feat_imp + imp_cwh

    # average & normalize feature-map importance
    feat_imp /= len(imgs)
    feat_imp = (feat_imp - feat_imp.min()) / (feat_imp.ptp()+1e-9)
    Hf, Wf = feat_imp.shape

    # 5) compute per-location eccentricities (in pixels)
    # create full grid of feature-map coordinates
    ys, xs = np.indices((Hf, Wf))  # each of shape [Hf, Wf]
    # map to pixel centers via rf_px (vectorized)
    cy_mat, cx_mat = rf_px(ys, xs, v1.stride[0], v1.padding[0], v1.kernel_size[0])
    centers = ((224-1)/2.0, (224-1)/2.0)
    dist = np.hypot(cy_mat-centers[0], cx_mat-centers[1])
    dist_norm = dist / dist.max()

    # 6) central–peripheral index (higher = more central)
    central_weight = 1.0 - dist_norm
    cpi = (feat_imp * central_weight).sum() / feat_imp.sum() #central–peripheral index (higher = more central)
    central_weight = 1.0 - dist_norm
    cpi = (feat_imp * central_weight).sum() / feat_imp.sum()
    print(f"Feature-map Central–Peripheral index = {cpi:.3f}")

    # 7) save & plot heatmap in feature-map space
    outdir = pathlib.Path(cfg.get("output_dir","results")); outdir.mkdir(exist_ok=True)
    plt.figure(figsize=(5,5))
    plt.imshow(feat_imp, cmap="hot", interpolation='nearest')
    plt.title(f"V1 Feature-Map Heatmap: CPI={cpi:.3f}")
    plt.axis('off')
    plt.savefig(outdir/f"{cat}_v1_featuremap_heatmap.png", dpi=300)
    plt.show()

    # CSV export
    df = pd.DataFrame({
        'fy': ys.flatten(), 'fx': xs.flatten(),
        'importance': feat_imp.flatten(), 'dist_norm': dist_norm.flatten()
    })
    df.to_csv(outdir/f"{cat}_v1_featuremap_importance.csv", index=False)
    print("Saved feature-map heatmap and CSV to", outdir)

if __name__=="__main__":
    if len(sys.argv)!=2:
        sys.exit("usage: python it2v1_featuremap_contrib.py <config.yaml>")
    main(sys.argv[1])
