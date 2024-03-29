#!/bin/bash

### Environment Variables
GPU_ID=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=`pwd`

### Hyperparameters
TASK=absa
TRAIN_BATCH_SIZE=24
EVAL_BATCH_SIZE=24
EPOCHS=60
SEED=212

DOMAINS=(device laptops rest service)
for i in "${!DOMAINS[@]}"
do
  TRG_DOMAIN=${DOMAINS[$i]}
  DOMAINS_TMP=("${DOMAINS[@]}")
  unset DOMAINS_TMP[$i]
  SRC_DOMAINS=$(echo ${DOMAINS_TMP[*]} | tr ' ' ',')

  echo "Running experiment for $SRC_DOMAINS as source domains and $TRG_DOMAIN as target domain"

  set -x

  echo "Extracting DRFs for the current experiment."
  python ./src/utils/drf_extraction.py \
  --domains ${SRC_DOMAINS} \
  --dtype ${TASK} \
  --drf_set_location ./runs/${TASK}/${TRG_DOMAIN}/drf_sets

  echo "Annotating training examples with DRF-based prompts."
  python ./src/utils/prompt_annotation.py \
  --domains ${SRC_DOMAINS} \
  --root_data_dir ${TASK}_data \
  --drf_set_location ./runs/${TASK}/${TRG_DOMAIN}/drf_sets \
  --prompts_data_dir ./runs/${TASK}/${TRG_DOMAIN}/prompt_annotations

  CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --dataset_name ${TASK} \
  --src_domains ${SRC_DOMAINS} \
  --trg_domain ${TRG_DOMAIN} \
  --num_train_epochs ${EPOCHS} \
  --train_batch_size ${TRAIN_BATCH_SIZE} \
  --eval_batch_size ${EVAL_BATCH_SIZE} \
  --seed ${SEED}
done