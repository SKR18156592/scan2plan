"""Usage: python -m scan2plan <capture_dir> [--out DIR]"""
import argparse

from .pipeline import run


def main():
    ap = argparse.ArgumentParser(prog="scan2plan")
    ap.add_argument("capture")
    ap.add_argument("--out", default=None)
    ap.add_argument("--step", type=int, default=3, help="use every Nth frame")
    ap.add_argument("--no-cache", action="store_true", help="cold run: ignore and do not write caches")
    ap.add_argument("--no-drift", action="store_true", help="ablation: use raw ARKit poses")
    a = ap.parse_args()
    from pathlib import Path
    out = a.out or str(Path("out") / Path(a.capture).name)
    r = run(a.capture, out, drift_correction=not a.no_drift, step=a.step, cache_dir=None if a.no_cache else "out/cache")
    print(f"{r['property']['room_count']} rooms, footprint {r['property']['footprint']['value']:.2f} m², "
          f"{r['timing_s']['total_s']:.0f} s -> {out}/plan.json, plan.png")


main()
