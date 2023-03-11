#!/bin/bash

cd ../..

# custom config
DATA=~/Datasets/coop_datasets/
TRAINER=LASP

DATASET=$1
SEED=$2

CFG=$3
SHOTS=16


DIR=output/base2new/train_base_all/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES all

