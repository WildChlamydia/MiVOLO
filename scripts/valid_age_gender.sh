#!/usr/bin/env bash

# inference utk
python3 eval_pretrained.py \
  --dataset_images data/utk/images \
  --dataset_annotations data/utk/annotation \
  --dataset_name utk \
  --batch-size 512 \
  --checkpoint pretrained/checkpoint-377.pth.tar \
  --split valid \
  --half \
  --with-persons \
  --device "cuda:0"

# inference fairface
python3 eval_pretrained.py \
  --dataset_images data/FairFace/fairface-img-margin125-trainval \
  --dataset_annotations data/FairFace/annotations \
  --dataset_name fairface \
  --batch-size 512 \
  --checkpoint pretrained/checkpoint-377.pth.tar \
  --split val \
  --half \
  --with-persons \
  --device "cuda:0"

# inference adience
python3 eval_pretrained.py \
  --dataset_images data/adience/faces \
  --dataset_annotations data/adience/annotations \
  --dataset_name adience \
  --batch-size 512 \
  --checkpoint pretrained/checkpoint-377.pth.tar \
  --split adience \
  --half \
  --with-persons \
  --device "cuda:0"
