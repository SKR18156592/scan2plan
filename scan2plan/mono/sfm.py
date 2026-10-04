"""Structure from motion with COLMAP (pycolmap). Run as its own process:
pycolmap and torch ship incompatible OpenMP runtimes.

    python -m scan2plan.mono.sfm <images_dir> <out_dir> [--sequential]

Writes <out_dir>/sfm.json: per registered image its camera-to-world pose,
intrinsics, and the sparse points it observes (pixel + camera-frame depth),
which later fix the metric scale of the monocular depth.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pycolmap


# iPhone 15+ main (1x) camera: horizontal field of view ~62-65 deg in 4:3,
# i.e. focal ~0.83 x the long image side. Used as a prior only when the
# image carries no EXIF focal length (video frames); bundle adjustment
# still refines it.
IPHONE_FOCAL_FACTOR = 0.83


def import_features(db_path, features_npz):
    """Replace the database's SIFT keypoints/matches with DISK + LightGlue.

    COLMAP's own extraction is still run first because it creates the
    camera/image/frame records (with the focal prior) correctly.
    """
    f = np.load(features_npz, allow_pickle=False)
    names = [str(n) for n in f["names"]]
    db = pycolmap.Database.open(str(db_path))
    ids = {im.name: im.image_id for im in db.read_all_images()}
    db.clear_keypoints()
    db.clear_descriptors()
    db.clear_matches()
    db.clear_two_view_geometries()
    for k, n in enumerate(names):
        db.write_keypoints(ids[n], f[f"kp_{k}"].astype(np.float32))
    pairs = []
    for key in f.files:
        if not key.startswith("m_"):
            continue
        i, j = map(int, key[2:].split("_"))
        db.write_matches(ids[names[i]], ids[names[j]], f[key].astype(np.uint32))
        pairs.append(f"{names[i]} {names[j]}")
    db.close()
    pairs_path = Path(db_path).with_name("pairs.txt")
    pairs_path.write_text("\n".join(pairs) + "\n")
    pycolmap.verify_matches(str(db_path), str(pairs_path))
    return len(pairs)


def run(images, out, sequential=False, max_size=1024, mapper="incremental", features=None):
    images, out = Path(images), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "database.db"
    if db.exists():
        db.unlink()
    names = sorted(str(p.relative_to(images)) for p in images.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".heic"))
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = 4096
    # Indoor walls are low-texture: a lower DoG peak threshold keeps the weak
    # corners (skirting, sockets, frames) that the default discards.
    ext.sift.peak_threshold = 0.002
    ext.max_image_size = max_size
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    reader.default_focal_length_factor = IPHONE_FOCAL_FACTOR
    # Video frames share one camera; photos may come from different lenses.
    mode = pycolmap.CameraMode.SINGLE if sequential else pycolmap.CameraMode.AUTO
    pycolmap.extract_features(db, images, image_names=names, camera_mode=mode, reader_options=reader, extraction_options=ext)
    if features:
        import_features(db, features)
    elif sequential:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = 12
        pairing.quadratic_overlap = True
        pycolmap.match_sequential(db, pairing_options=pairing)
    else:
        pycolmap.match_exhaustive(db)
    sparse = out / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir()
    if mapper == "global":
        recs = pycolmap.global_mapping(db, images, sparse)
    else:
        opts = pycolmap.IncrementalPipelineOptions()
        opts.min_model_size = 3
        opts.min_num_matches = 10
        opts.mapper.init_min_num_inliers = 50
        opts.mapper.abs_pose_min_num_inliers = 15
        opts.mapper.abs_pose_min_inlier_ratio = 0.15
        opts.mapper.init_min_tri_angle = 4.0
        opts.structure_less_registration_fallback = True
        recs = pycolmap.incremental_mapping(db, images, sparse, options=opts)
    models = []
    for idx, rec in sorted(recs.items(), key=lambda kv: -kv[1].num_reg_images()):
        models.append(_export(rec))
    result = dict(images=names, models=models)
    json.dump(result, open(out / "sfm.json", "w"))
    print(f"{len(names)} images, {len(models)} model(s), registered: {[len(m['images']) for m in models]}")
    return result


def _export(rec):
    imgs = {}
    for iid, im in rec.images.items():
        if not im.has_pose:
            continue
        cam = rec.cameras[im.camera_id]
        T_cw = im.cam_from_world().matrix()  # 3x4
        R, t = T_cw[:, :3], T_cw[:, 3]
        T_wc = np.eye(4)
        T_wc[:3, :3] = R.T
        T_wc[:3, 3] = -R.T @ t
        obs = []
        for p2d in im.points2D:
            if p2d.has_point3D():
                X = rec.points3D[p2d.point3D_id].xyz
                z = (R @ X + t)[2]
                if z > 0:
                    obs.append([float(p2d.xy[0]), float(p2d.xy[1]), float(z)])
        K = cam.calibration_matrix()
        imgs[im.name] = dict(T_wc=T_wc.tolist(), K=K.tolist(), width=cam.width, height=cam.height,
                             params=list(map(float, cam.params)), obs=obs)
    pts = np.array([p.xyz for p in rec.points3D.values()]) if len(rec.points3D) else np.zeros((0, 3))
    return dict(images=imgs, n_points=int(len(pts)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("out")
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--mapper", default="incremental", choices=["incremental", "global"])
    ap.add_argument("--features", default=None, help="DISK+LightGlue npz from scan2plan.mono.features")
    a = ap.parse_args()
    run(a.images, a.out, a.sequential, mapper=a.mapper, features=a.features)
