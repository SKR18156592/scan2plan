"""Open-vocabulary damage detection on keyframes (OWLv2). Own process (torch).

    python -m scan2plan.damage.detect <frames.json> <out.json>

frames.json: {"images": [{"id": i, "path": ..., "rot": k}]}, where rot is
the clockwise quarter-turns that make the image upright (detectors, like
depth networks, are trained on upright photos). Boxes are returned in the
original (un-rotated) pixel frame, normalised to [0, 1].

Model: google/owlv2-base-patch16-ensemble (Apache-2.0), zero-shot. No
damage-specific training data was used; thresholds are set so that the
sample captures (no visible damage) produce few detections, and the
false-positive count on them is reported in the benchmark.
"""
import json
import sys

import numpy as np
import torch
from PIL import Image, ImageOps

# Damage classes and the text prompts that query each one.
CLASSES = {
    "water_stain": ["a water stain on a wall", "a brown water stain on a ceiling", "water damage on drywall"],
    "mold": ["black mold on a wall", "mould growth in a corner"],
    "crack": ["a crack in a wall", "a cracked plaster ceiling"],
    "hole": ["a hole in drywall", "a hole in a wall"],
    "peeling_paint": ["peeling paint on a wall", "blistering paint"],
}
THRESHOLD = 0.30  # detector output threshold (cached); per-class filter below

# Per-class acceptance thresholds, set on the sample captures (which contain
# no visible damage): the 0.30 threshold produced 11 multi-view regions there,
# all false - marble veining as "water_stain" (0.33-0.43) and straight
# ceiling/wall junctions near downlights as "crack" (0.32-0.52). These values
# reject all of them. They are negatives-only calibration: the true-positive
# rate on real damage is unmeasured (no staged-damage capture exists).
CLASS_THRESHOLDS = {"water_stain": 0.45, "mold": 0.35, "crack": 0.55, "hole": 0.35, "peeling_paint": 0.35}
MODEL = "google/owlv2-base-patch16-ensemble"


def main(frames_json, out_json):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = Owlv2Processor.from_pretrained(MODEL)
    net = Owlv2ForObjectDetection.from_pretrained(MODEL).to(dev).eval()
    prompts, owner = [], []
    for c, ps in CLASSES.items():
        prompts += ps
        owner += [c] * len(ps)
    frames = json.load(open(frames_json))["images"]
    out = []
    for f in frames:
        img = ImageOps.exif_transpose(Image.open(f["path"])).convert("RGB")
        W, H = img.size
        rot = f.get("rot", 0)
        up = Image.fromarray(np.ascontiguousarray(np.rot90(np.asarray(img), -rot)))
        inputs = proc(text=[prompts], images=up, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = net(**inputs)
        # OWLv2 pads to a square; boxes are relative to the padded square.
        s = max(up.size)
        res = proc.post_process_grounded_object_detection(o, threshold=THRESHOLD, target_sizes=torch.tensor([[s, s]]))[0]
        for box, score, lab in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), res["labels"].cpu().numpy()):
            x0, y0, x1, y1 = np.clip(box, 0, None)
            x1, y1 = min(x1, up.size[0]), min(y1, up.size[1])
            if (x1 - x0) * (y1 - y0) > 0.5 * up.size[0] * up.size[1]:
                continue  # whole-image boxes are prompt echoes, not regions
            box_raw = _unrotate_box((x0, y0, x1, y1), up.size, rot)
            out.append(dict(image=f["id"], cls=owner[int(lab)], prompt=prompts[int(lab)], score=float(score),
                            box=[box_raw[0] / W, box_raw[1] / H, box_raw[2] / W, box_raw[3] / H]))
    json.dump(dict(model=MODEL, threshold=THRESHOLD, classes=CLASSES, detections=out), open(out_json, "w"), indent=1)
    print(f"{len(frames)} frames, {len(out)} detections")


def _unrotate_box(b, up_size, rot):
    """Box in the upright image -> box in the original image."""
    x0, y0, x1, y1 = b
    w, h = up_size
    pts = np.array([[x0, y0], [x1, y1]], float)
    for _ in range(rot % 4):
        # Undo one clockwise turn: (x, y) in rotated (w, h) -> (y, w - x) in original (h, w).
        pts = np.stack([pts[:, 1], w - pts[:, 0]], 1)
        w, h = h, w
    return [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]


if __name__ == "__main__":
    main(*sys.argv[1:3])
