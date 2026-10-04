"""Usage: python -m scan2plan <capture_dir> [--out DIR]"""
import argparse

from .pipeline import run


def main():
    ap = argparse.ArgumentParser(prog="scan2plan")
    ap.add_argument("capture")
    ap.add_argument("--out", default=None)
    ap.add_argument("--step", type=int, default=3, help="use every Nth frame")
    ap.add_argument("--cache", default=None, help="point cloud cache .npz")
    a = ap.parse_args()
    from pathlib import Path
    out = a.out or str(Path("out") / Path(a.capture).name)
    r = run(a.capture, out, step=a.step, cloud_cache=a.cache)
    print(f"{r['property']['room_count']} rooms, footprint {r['property']['footprint']['value']:.2f} m², "
          f"{r['timing_s']['total_s']:.0f} s -> {out}/plan.json, plan.png")


main()
