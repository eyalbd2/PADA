#!/bin/bash

### Environment Variables
GPU_ID=0
CWD=`pwd`
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${CWD}

### Hyperparameters
CKPT_PATH=${CWD}/checkpoints/
TASK=rumor
EVAL_BATCH_SIZE=32
SEED=41

DOMAINS=(charliehebdo ferguson germanwings-crash ottawashooting sydneysiege)
for i in "${!DOMAINS[@]}"
do
  TRG_DOMAIN=${DOMAINS[$i]}
  DOMAINS_TMP=("${DOMAINS[@]}")
  unset DOMAINS_TMP[$i]
  SRC_DOMAINS=$(echo ${DOMAINS_TMP[*]} | tr ' ' ',')

  echo "Running evaluation checkpoint for $SRC_DOMAINS as source domains and $TRG_DOMAIN as target domain"

  set -x

  if [ ! -d "${CKPT_PATH}/${TASK}/${TRG_DOMAIN}/prompt_annotations" ]
  then
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
  fi

  CUDA_VISIBLE_DEVICES=${GPU_ID} python ./eval.py \
  --dataset_name ${TASK} \
  --src_domains ${SRC_DOMAINS} \
  --trg_domain ${TRG_DOMAIN} \
  --eval_batch_size ${EVAL_BATCH_SIZE} \
  --seed ${SEED} \
  --ckpt_path ${CKPT_PATH}
done