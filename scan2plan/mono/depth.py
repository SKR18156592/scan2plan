"""Metric monocular depth with Depth Anything V2 (Metric-Indoor). Run as its
own process (torch and pycolmap ship incompatible OpenMP runtimes).

    python -m scan2plan.mono.depth <images_dir> <out_dir> [--model small|base]

Writes one float16 .npy depth map (metres) per image at 256x192 for 4:3
input, the same grid as the LiDAR depth so the downstream code is shared.
Weights are downloaded from Hugging Face on first use and cached; set
HF_HUB_OFFLINE=1 after `scripts/fetch_weights.sh` to run without network.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
}


def load(model="small"):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    name = MODELS[model]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(name)
    net = AutoModelForDepthEstimation.from_pretrained(name).to(dev).eval()
    return proc, net, dev


@torch.no_grad()
def predict(img, proc, net, dev, out_size=(256, 192)):
    inputs = proc(images=img, return_tensors="pt").to(dev)
    d = net(**inputs).predicted_depth  # (1, h, w), metres
    d = torch.nn.functional.interpolate(d[:, None], size=out_size[::-1], mode="bilinear", align_corners=False)
    return d[0, 0].float().cpu().numpy()


def run(images, out, model="small", rot=0):
    """rot: clockwise quarter-turns that make the images upright. The network
    is run on the upright image and its depth is turned back, so depth maps
    stay aligned with the original pixels (and SfM intrinsics)."""
    images, out = Path(images), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    proc, net, dev = load(model)
    paths = sorted(p for p in images.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".heic"))
    for i, p in enumerate(paths):
        rel = p.relative_to(images).with_suffix(".npy")
        dst = out / rel
        if dst.exists():
            continue
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        w, h = img.size
        up = np.rot90(np.asarray(img), -rot)
        uh, uw = up.shape[:2]
        size = (256, 192) if uw >= uh else (192, 256)
        d = predict(Image.fromarray(np.ascontiguousarray(up)), proc, net, dev, size)
        d = np.rot90(d, rot)
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, d.astype(np.float16))
        if i % 25 == 0:
            print(f"{i}/{len(paths)} {rel} median {np.median(d):.2f} m", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("out")
    ap.add_argument("--model", default="small", choices=list(MODELS))
    ap.add_argument("--rot", type=int, default=0)
    a = ap.parse_args()
    run(a.images, a.out, a.model, a.rot)
