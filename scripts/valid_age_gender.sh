#!/usr/bin/env bash

# inference utk
python3 eval_pretrained.py \
  --dataset_images data/utk/images \
  --dataset_annotations data/utk/annotation \
  --dataset_name utk \
  --batch-size 512 \
  --checkpoint pretrained/model_imdb_cross_person_4.24_99.46.pth.tar \
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
  --checkpoint pretrained/model_imdb_cross_person_4.24_99.46.pth.tar \
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
  --checkpoint pretrained/model_imdb_cross_person_4.24_99.46.pth.tar \
  --split adience \
  --half \
  --with-persons \
  --device "cuda:0"

# inference agedb
python3 eval_pretrained.py \
  --dataset_images data/agedb/AgeDB \
  --dataset_annotations data/agedb/annotation \
  --dataset_name agedb \
  --batch-size 512 \
  --checkpoint pretrained/model_imdb_cross_person_4.24_99.46.pth.tar \
  --split 0,1,2,3,4,5,6,7,8,9 \
  --half \
  --device "cuda:0"
