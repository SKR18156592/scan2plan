"""Learned features for SfM: DISK keypoints + LightGlue matching (kornia).
Run as its own process (torch).

    python -m scan2plan.mono.features <images_dir> <out.npz> [--sequential W]

SIFT finds ~400 keypoints on the white walls of the sample captures and
leaves runs of consecutive frames with zero matches; DISK+LightGlue is far
more robust on low texture and blur. Pairs: a sliding window for video,
all pairs for photo folders.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import kornia.feature as KF
from PIL import Image, ImageOps

MAX_SIDE = 1024
NUM_KP = 1024


def load_img(p, dev):
    img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    w, h = img.size
    s = MAX_SIDE / max(w, h)
    img = img.resize((int(round(w * s / 16) * 16), int(round(h * s / 16) * 16)), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1)[None] / 255.0
    return t.to(dev), (w, h), (img.size[0] / w, img.size[1] / h)


@torch.no_grad()
def run(images, out, window=None):
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    disk = KF.DISK.from_pretrained("depth").to(dev).eval()
    lg = KF.LightGlue("disk").to(dev).eval()
    images = Path(images)
    names = sorted(str(p.relative_to(images)) for p in images.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    feats = []
    for n in names:
        t, size, sc = load_img(images / n, dev)
        f = disk(t, n=NUM_KP, pad_if_not_divisible=True)[0]
        kp = f.keypoints.cpu().numpy() / np.array(sc)  # back to original pixels
        feats.append(dict(kp=kp, kp_net=f.keypoints, desc=f.descriptors, size=t.shape[-2:], orig=size))
    n = len(names)
    if window:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, min(i + 1 + window, n))]
        # Sparse long-range pairs give the mapper loop closures.
        pairs += [(i, j) for i in range(0, n, 5) for j in range(i + window + 1, n, 5)]
    else:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    matches = {}
    for i, j in pairs:
        a, b = feats[i], feats[j]
        hw0 = torch.tensor(a["size"], device=dev)[None]
        hw1 = torch.tensor(b["size"], device=dev)[None]
        lafs0 = KF.laf_from_center_scale_ori(a["kp_net"][None], torch.full((1, len(a["kp_net"]), 1, 1), 16.0, device=dev))
        lafs1 = KF.laf_from_center_scale_ori(b["kp_net"][None], torch.full((1, len(b["kp_net"]), 1, 1), 16.0, device=dev))
        out_ = lg({"image0": {"keypoints": a["kp_net"][None], "descriptors": a["desc"][None], "image_size": hw0.flip(-1)},
                   "image1": {"keypoints": b["kp_net"][None], "descriptors": b["desc"][None], "image_size": hw1.flip(-1)}})
        m = out_["matches"][0].cpu().numpy()
        if len(m) >= 15:
            matches[f"{i}_{j}"] = m.astype(np.uint32)
    np.savez_compressed(out, names=np.array(names), **{f"kp_{k}": f["kp"].astype(np.float32) for k, f in enumerate(feats)},
                        **{f"m_{k}": v for k, v in matches.items()})
    print(f"{n} images, {len(pairs)} pairs tried, {len(matches)} with >= 15 matches")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("out")
    ap.add_argument("--sequential", type=int, default=None, help="window size for video")
    a = ap.parse_args()
    run(a.images, a.out, a.sequential)
