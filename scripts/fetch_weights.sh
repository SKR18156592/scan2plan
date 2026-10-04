#!/usr/bin/env bash
# Downloads all model weights so the photo/video tiers can run offline
# (HF_HUB_OFFLINE=1). Weights land in the Hugging Face and torch hub caches.
set -euo pipefail
PY=${PY:-.venv/bin/python}
# python.org builds of Python ship without a CA bundle; use certifi's.
export SSL_CERT_FILE=$($PY -m certifi)
$PY - <<'PY'
from huggingface_hub import snapshot_download
for repo in ["depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
             "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"]:
    print(snapshot_download(repo))
import kornia.feature as KF
KF.DISK.from_pretrained("depth")
KF.LightGlue("disk")
print("DISK + LightGlue weights cached")
PY
