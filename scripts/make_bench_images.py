"""Copy a fixed set of val-split images into ios/TEAMBench/BenchImages as bench_<i>.jpg."""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from team_sam.data import COCO_ROOT, REPO_ROOT, load_splits  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
out = REPO_ROOT / "ios" / "TEAMBench" / "BenchImages"
out.mkdir(parents=True, exist_ok=True)
ids = load_splits()["val"][:N]
for i, img_id in enumerate(ids):
    shutil.copy(COCO_ROOT / "val2017" / f"{img_id:012d}.jpg", out / f"bench_{i}.jpg")
(out / "SOURCE.txt").write_text("COCO val2017 image ids (CC BY 4.0, see cocodataset.org):\n" + "\n".join(map(str, ids)) + "\n")
print(f"wrote {len(ids)} images to {out}")
