#!/bin/bash

### Environment Variables
GPU_ID=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=`pwd`

### Hyperparameters
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=32
EPOCHS=5
ALPHA=0.2
SEED=41


DOMAINS=(charliehebdo ferguson germanwings-crash ottawashooting sydneysiege)
for i in "${!DOMAINS[@]}"
do
  TRG_DOMAIN=${DOMAINS[$i]}
  DOMAINS_TMP=("${DOMAINS[@]}")
  unset DOMAINS_TMP[$i]
  SRC_DOMAINS=$(echo ${DOMAINS_TMP[*]} | tr ' ' ',')

  set -x
  echo "Running experiment for $SRC_DOMAINS as source domains and $TRG_DOMAIN as target domain"

  echo "Extracting DRFs for the current experiment."
  python ./src/utils/drf_extraction.py \
  --domains ${SRC_DOMAINS} \
  --dtype rumor \
  --drf_set_location ./runs/rumor/${TRG_DOMAIN}/drf_sets

  echo "Annotating training examples with DRF-based prompts."
  python ./src/utils/prompt_annotation.py \
  --domains ${SRC_DOMAINS} \
  --root_data_dir rumor_data \
  --drf_set_location ./runs/rumor/${TRG_DOMAIN}/drf_sets \
  --prompts_data_dir ./runs/rumor/${TRG_DOMAIN}/prompt_annotations

  CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --dataset_name rumor \
  --src_domains ${SRC_DOMAINS} \
  --trg_domain ${TRG_DOMAIN} \
  --num_train_epochs ${EPOCHS} \
  --train_batch_size ${TRAIN_BATCH_SIZE} \
  --eval_batch_size ${EVAL_BATCH_SIZE} \
  --mixture_alpha ${ALPHA} \
  --seed ${SEED}
done