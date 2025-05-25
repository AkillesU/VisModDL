#!/usr/bin/env python3
"""
unit_categ_selectivity.py

Same YAML as before.  Now correctly computes leave-one-out scaling across
the four categories for each unit.
"""

import sys, pathlib, yaml
import torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import pandas as pd
from tqdm import tqdm

def load_model(cfg, device):
    src, name, wts = cfg.get("source","torchvision"), cfg["name"], cfg.get("weights","pretrained")
    if src == "cornet":
        import cornet
        ctor = {"cornet_rt": cornet.cornet_rt,
                "cornet_s": cornet.cornet_s,
                "cornet_z": cornet.cornet_z}[name.lower()]
        model = ctor(pretrained=(wts=="pretrained"),
                     **({"times":cfg.get("time_steps")} if name=="cornet_rt" else {}))
    elif src == "timm":
        import timm
        model = timm.create_model(name, pretrained=(wts=="pretrained"))
    else:  # torchvision or pytorch_hub
        if src=="pytorch_hub":
            model = torch.hub.load(cfg["repo"], name,
                                   weights=None if wts!="pretrained" else wts)
        else:
            import torchvision.models as tvm
            ctor = getattr(tvm, name)
            model = ctor(weights="IMAGENET1K_V1" if wts=="pretrained" else None)
    return model.to(device).eval()

def build_transform():
    return T.Compose([T.Resize(256), T.CenterCrop(224),
                      T.ToTensor(),
                      T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def iter_imgs(folder):
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in {".jpg",".jpeg",".png"}:
            yield Image.open(p).convert("RGB")

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path,"r",encoding="utf-8-sig"))
    device = torch.device((cfg["device"] if cfg["device"]!="auto"
                           else ("cuda" if torch.cuda.is_available() else "cpu")))
    model = load_model(cfg["model"], device)
    transform = build_transform()

    # 1) Discover conv layers and register hooks
    conv_layers = [n for n,m in model.named_modules() if isinstance(m, nn.Conv2d)]
    act_sums = {layer: None for layer in conv_layers}

    def make_hook(layer):
        def hook(_mod, _in, out):
            vec = out.detach().mean((0,2,3)).cpu()  # (C,)
            act_sums[layer] = act_sums[layer] + vec
        return hook

    for layer in conv_layers:
        mod = dict(model.named_modules())[layer]
        act_sums[layer] = torch.zeros(mod.out_channels)
        mod.register_forward_hook(make_hook(layer))

    # 2) Run each category and store raw means
    root = pathlib.Path(cfg["data_root"])
    categories = sorted([p.name for p in root.iterdir() if p.is_dir()])
    raw_means = {
        layer: {ch: {} for ch in range(act_sums[layer].shape[0])}
        for layer in conv_layers
    }

    batch = cfg.get("batch_size",4)
    for cat in categories:
        # zero sums
        for layer in conv_layers:
            act_sums[layer].zero_()
        n = 0
        buf = []
        for img in tqdm(iter_imgs(root/cat), total=20, desc=cat):
            buf.append(transform(img))
            if len(buf)==batch:
                with torch.no_grad(): model(torch.stack(buf).to(device))
                n += len(buf); buf.clear()
        if buf:
            with torch.no_grad(): model(torch.stack(buf).to(device))
            n += len(buf)

        # record per-filter mean
        for layer in conv_layers:
            mv = (act_sums[layer]/n).numpy()
            for ch,val in enumerate(mv):
                raw_means[layer][ch][cat] = float(val)

    # 3) Now compute leave-one-out scaling and write CSVs
    out = pathlib.Path(cfg.get("out_dir","results"))
    out.mkdir(exist_ok=True)
    for cat in categories:
        rows=[]
        for layer in conv_layers:
            for ch, d in raw_means[layer].items():
                vals = [d[c] for c in categories]
                total = sum(vals)
                N = len(vals)
                baseline = (total - d[cat]) / (N-1) or 1e-9
                scaled = d[cat] / baseline
                rows.append((layer, ch, f"{layer}:{ch}", d[cat], scaled))
        df = pd.DataFrame(rows, columns=[
            "layer_name","channel","unit_id","mean_activation","scaled_activation"
        ])
        df.to_csv(out/f"{cat}_unit_selectivity.csv", index=False)
        print(f"Wrote {cat}_unit_selectivity.csv ({len(df)} rows)")

if __name__=="__main__":
    if len(sys.argv)!=2:
        sys.exit("Usage: python unit_categ_selectivity.py <config.yaml>")
    main(sys.argv[1])
