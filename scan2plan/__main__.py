"""One command per capture.

    scan2plan run lidar  <stray_scanner_folder>   [--out DIR] [--no-drift]
    scan2plan run video  <clip.mov|.mp4>          [--out DIR]
    scan2plan run photos <folder of room folders> [--out DIR]
"""
import argparse
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scan2plan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process one capture")
    r.add_argument("tier", choices=["lidar", "video", "photos"])
    r.add_argument("input")
    r.add_argument("--out", default=None)
    r.add_argument("--no-drift", action="store_true", help="LiDAR ablation: raw ARKit poses")
    r.add_argument("--no-damage", action="store_true", help="skip damage detection")
    r.add_argument("--no-cache", action="store_true", help="LiDAR: cold run, no point-cloud cache")
    r.add_argument("--model", default="base", choices=["small", "base"], help="mono depth model (video/photos)")
    a = ap.parse_args(argv)
    name = Path(a.input).stem if a.tier == "video" else Path(a.input).name
    out = a.out or str(Path("out") / f"{name}_{a.tier}")
    if a.tier == "lidar":
        from .pipeline import run
        res = run(a.input, out, drift_correction=not a.no_drift, cache_dir=None if a.no_cache else "out/cache",
                  damage=not a.no_damage)
    else:
        from .mono.run import run
        res = run("video" if a.tier == "video" else "photos", a.input, out, a.model, damage=not a.no_damage)
    p = res["property"]
    print(f"{p['room_count']} rooms, footprint {p['footprint']['value']:.2f} m² "
          f"(95% {p['footprint']['ci95'][0]:.2f}-{p['footprint']['ci95'][1]:.2f}), "
          f"{len(res['damage_regions'])} damage regions, {res['timing_s']['total_s']:.0f} s -> {out}/plan.json, plan.png")


if __name__ == "__main__":
    main()
