#!/usr/bin/env bash
# Download EfficientViT-SAM-L0 weights and COCO 2017 annotations + val images.
# Train images for distillation (2,000) are fetched later from splits/train2017_urls.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p assets/checkpoints data/coco
W=assets/checkpoints/efficientvit_sam_l0.pt
[ -f "$W" ] || curl -L --retry 5 -o "$W" https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l0.pt
S3=https://s3.us-east-1.amazonaws.com/images.cocodataset.org
cd data/coco
[ -d annotations ] || { curl -L --retry 5 -O $S3/annotations/annotations_trainval2017.zip && unzip -q annotations_trainval2017.zip && rm annotations_trainval2017.zip; }
[ -d val2017 ] || { curl -L --retry 5 -O $S3/zips/val2017.zip && unzip -q val2017.zip && rm val2017.zip; }
if [ "${1:-}" = "--train" ]; then
  mkdir -p train2017 && cd train2017
  xargs -n 1 -P 8 curl -sSLO --retry 5 < ../../../splits/train2017_urls.txt
fi
echo "assets ready"
