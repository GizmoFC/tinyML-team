"""Freeze the data splits used for the whole project (run once, commit splits/splits.json).

- train: 2,000 COCO train2017 images (for feature distillation; labels unused)
- val:   500 COCO val2017 images (model selection)
- test:  1,000 COCO val2017 images, disjoint from val (final report only)

val/test images are drawn only from images with >= 1 eligible instance
(non-crowd, mask >= 100 px). Seeded, so re-running gives the same split.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from team_sam.data import SPLITS_PATH, eligible_instances, load_coco  # noqa: E402

S3 = "https://s3.us-east-1.amazonaws.com/images.cocodataset.org"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=1000)
    args = ap.parse_args()

    val_coco = load_coco("val2017")
    eligible = []
    n_inst = {}
    for img_id in sorted(val_coco.getImgIds()):
        k = len(eligible_instances(val_coco, img_id))
        if k:
            eligible.append(img_id)
            n_inst[img_id] = k
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    val_ids = sorted(eligible[: args.n_val])
    test_ids = sorted(eligible[args.n_val : args.n_val + args.n_test])
    assert not set(val_ids) & set(test_ids)

    train_coco = load_coco("train2017")
    train_all = sorted(train_coco.getImgIds())
    rng_t = random.Random(args.seed + 1)
    train_ids = sorted(rng_t.sample(train_all, args.n_train))
    train_files = {i: train_coco.loadImgs([i])[0]["file_name"] for i in train_ids}

    out = {
        "seed": args.seed,
        "rules": "val/test from val2017 images with >=1 non-crowd instance of >=100 mask px; disjoint",
        "train2017": train_ids,
        "val": val_ids,
        "test": test_ids,
        "num_instances": {
            "val": sum(n_inst[i] for i in val_ids),
            "test": sum(n_inst[i] for i in test_ids),
        },
    }
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(out, indent=1))
    (SPLITS_PATH.parent / "train2017_urls.txt").write_text(
        "".join(f"{S3}/train2017/{train_files[i]}\n" for i in train_ids)
    )
    print(
        f"eligible val2017 images: {len(eligible)}; val={len(val_ids)} ({out['num_instances']['val']} inst), "
        f"test={len(test_ids)} ({out['num_instances']['test']} inst), train={len(train_ids)}"
    )


if __name__ == "__main__":
    main()
