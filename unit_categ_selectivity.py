#!/usr/bin/env python3
"""
unit_categ_selectivity_all_units.py

Treats each (channel, y, x) activation in every Conv2d layer as its own “unit,”
computing leave-one-out scaled selectivity per spatial unit rather than averaging
across the spatial map of a filter.
"""

import sys
import pathlib
import yaml
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import pandas as pd
from tqdm import tqdm


def load_model(cfg, device):
    src, name, wts = cfg.get("source", "torchvision"), cfg["name"], cfg.get("weights", "pretrained")
    if src == "cornet":
        import cornet
        ctor = {
            "cornet_rt": cornet.cornet_rt,
            "cornet_s": cornet.cornet_s,
            "cornet_z": cornet.cornet_z,
        }[name.lower()]
        model = ctor(
            pretrained=(wts == "pretrained"),
            **({"times": cfg.get("time_steps")} if name == "cornet_rt" else {}),
        )
    elif src == "timm":
        import timm
        model = timm.create_model(name, pretrained=(wts == "pretrained"))
    else:
        if src == "pytorch_hub":
            model = torch.hub.load(cfg["repo"], name,
                                   weights=None if wts != "pretrained" else wts)
        else:
            import torchvision.models as tvm
            ctor = getattr(tvm, name)
            model = ctor(weights="IMAGENET1K_V1" if wts == "pretrained" else None)
    return model.to(device).eval()


def build_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def iter_imgs(folder):
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            yield Image.open(p).convert("RGB")


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8-sig"))
    device = torch.device(
        cfg.get("device") if cfg.get("device") != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(cfg["model"], device)
    transform = build_transform()

    # 1) Find all Conv2d layers and hook to accumulate per-(C,H,W)
    conv_layers = [name for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    act_sums = {}  # layer -> tensor [C, H, W]

    def make_hook(layer_name):
        def hook(module, inp, out):
            # out: [batch, C, H, W]
            summed = out.detach().cpu().sum(dim=0)  # [C, H, W]
            if layer_name not in act_sums:
                act_sums[layer_name] = summed.clone()
            else:
                act_sums[layer_name] += summed
        return hook

    modules = dict(model.named_modules())
    for layer in conv_layers:
        modules[layer].register_forward_hook(make_hook(layer))

    # 2) Process each category, accumulate activations
    root = pathlib.Path(cfg["data_root"])
    categories = sorted([p.name for p in root.iterdir() if p.is_dir()])
    batch_size = cfg.get("batch_size", 4)

    # store raw means: layer -> {(c,y,x): {cat: mean_act}}
    raw_means = {layer: {} for layer in conv_layers}

    for cat in categories:
        # reset sums
        act_sums.clear()
        n_imgs = 0
        buffer = []
        for img in tqdm(iter_imgs(root / cat), desc=f"Cat={cat}"):
            buffer.append(transform(img))
            if len(buffer) == batch_size:
                with torch.no_grad():
                    model(torch.stack(buffer).to(device))
                n_imgs += len(buffer)
                buffer.clear()
        if buffer:
            with torch.no_grad():
                model(torch.stack(buffer).to(device))
            n_imgs += len(buffer)

        # compute mean per spatial unit
        for layer, sum_map in act_sums.items():
            mean_map = sum_map / max(n_imgs, 1)
            C, H, W = mean_map.shape
            for c in range(C):
                for y in range(H):
                    for x in range(W):
                        key = f"{layer}:{c}:{y}:{x}"
                        raw_means[layer].setdefault(key, {})[cat] = float(mean_map[c, y, x].item())

    # 3) Compute leave-one-out scaled selectivity & save
    out_dir = pathlib.Path(cfg.get("out_dir", "results"))
    out_dir.mkdir(exist_ok=True)

    for cat in categories:
        rows = []
        for layer in conv_layers:
            for uid, d in raw_means[layer].items():
                vals = [d.get(c, 0.0) for c in categories]
                total = sum(vals)
                N = len(vals)
                baseline = (total - d[cat]) / (N - 1) if N > 1 else 1e-9
                scaled = d[cat] / baseline if baseline else float('inf')
                rows.append((layer, uid, d[cat], scaled))
        df = pd.DataFrame(rows, columns=[
            "layer_name", "unit_id", "mean_activation", "scaled_activation"
        ])
        df.to_csv(out_dir / f"{cat}_unit_selectivity_all_units.csv", index=False)
        print(f"Wrote {cat}_unit_selectivity_all_units.csv ({len(df)} rows)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python unit_categ_selectivity_all_units.py <config.yaml>")
    main(sys.argv[1])
