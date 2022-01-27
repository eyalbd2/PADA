# PADA

### Official code repository for the TACL'22 paper - ["PADA: Example-based Prompt Learning for on-the-fly Adaptation to Unseen Domains"](https://arxiv.org/abs/2102.12206)
 
PADA is an example-based prompt generation model, which adapts on-the-fly to unseen domains (or distributions in general).
It is trained on labeled data from multiple domains, and when presented with new examples (from unknown domains), it performs an autoregressive inference: (1) First generating an example-specific signature that maps the input example to the semantic space spanned by its training domains (denoted as DRFs); and then (2) it casts the generated signature as a prompt (prefix) and performs the downstream task. 

If you use this code please cite our paper (see recommended citation below).

Our code is implemented in [PyTorch](https://pytorch.org/), using the [Transformers](https://github.com/huggingface/transformers) and [PyTorch-Lightning](https://www.pytorchlightning.ai/) libraries. 

## Usage Instructions

Running an experiment with PADA consists of the following steps:

1. Create an experimental setup - Choose a single target domain of a given task (e.g., _charliehebdo_ from 'Rumor Detection') and its corresponding source domains (_ferguson_, _germanwings-crash_, _ottawashooting_, _sydneysiege_). 
2. Extract the DRF sets for each of the source domains. 
3. Annotate training examples with DRF-based prompts.
4. Run PADA - train PADA on the prompt-annotated training set and test it on the target domain test set.

Before diving into our running example of how to run PADA, make sure your virtual environment includes all requirements (specified in 'pada_env.yml').

We ran our experiments on a single NVIDIA Quadro RTX 6000 24GB GPU, CUDA 11.1 and PyTorch 1.7.1.

### 0. Setup a conda environment
You can run the following command to create a conda environment from our .yml file:
```
conda env create --file pada_env.yml
conda activate pada
```

Next, we go through these steps using our running example:
- Task - Rumor Detection.
- Source domains - _ferguson_, _germanwings-crash_, _ottawashooting_, _sydneysiege_.
- Target domain - _charliehebdo_
We use a specific set of hyperparameters (please refer to our paper for more details). 


Notice, you can run all the above steps with a single command by running the `run-rumor-experiments.sh` script:
```
bash run-rumor-experiments.sh
```

### 1. Create an experimental setup
```
GPU_ID=<ID of GPU>
PYTHONPATH=<path to repository root>
TOKENIZERS_PARALLELISM=false

TASK_NAME=rumor
ROOT_DATA_DIR=rumor_data
SOURCES=(ferguson germanwings-crash ottawashooting sydneysiege)
SRC_DOMAINS=$(echo ${SOURCES[*]} | tr ' ' ',')
TRG_DOMAIN=charliehebdo
MODEL_NAME=PADA

TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=32
NUM_EPOCHS=5
ALPHA_VAL=0.2
GPU_ID=0
```


### 2. Extract DRF sets

Then run the following command to extract a DRF set for each of the source domains.

```
python ./src/utils/drf_extraction.py \
--domains ${SRC_DOMAINS} \
--dtype ${TASK_NAME} \
--drf_set_location ./runs/${TASK_NAME}/${TRG_DOMAIN}/drf_sets
```

This will save 4 files, each named by '<SRC_DOMAN_NAME>.pkl', in the following directory: './runs/<TASK_NAME>/<TRG_DOMAIN>/drf_sets'.

### 3. Annotate training examples with DRF-based prompts

```
python ./src/utils/prompt_annotation.py \
    --domains ${SRC_DOMAINS} \
    --root_data_dir ${ROOT_DATA_DIR} \
    --drf_set_location ./runs/${TASK_NAME}/${TRG_DOMAIN}/drf_sets \
    --prompts_data_dir ./runs/${TASK_NAME}/${TRG_DOMAIN}/prompt_annotations
```
For each source domain, this code creates a file with annotated prompt per each of its training example. The file is placed in the following path: './runs/<TASK_NAME>/<TRG_DOMAIN>/prompt_annotations/<SRC_DOMAN_NAME>/annotated_prompts_train.pt'. 
** model hyperparameters grid for this step are specified in the paper.

### 3. Run PADA

Train PADA on the prompt-generation task and the downstream task (conditioned on the annotated-prompts). Then. evaluate PADA on examples from the target domain where it first generates a prompt and then condition on this generated prompt, it performs the downstream task.   

```
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --dataset_name ${TASK_NAME} \
  --src_domains ${SRC_DOMAINS} \
  --trg_domain ${TRG_DOMAIN} \
  --num_train_epochs ${NUM_EPOCHS} \
  --train_batch_size ${TRAIN_BATCH_SIZE} \
  --eval_batch_size ${EVAL_BATCH_SIZE} \
  --mixture_alpha ${ALPHA_VAL} \
```

The final results are saved in the following path: 
  "./runs/<TASK_NAME>/<TRG_DOMAIN>/PADA/e<NUM_EPOCHS>/b<TRAIN_BATCH_SIZE>/a<ALPHA_VAL>/test_results.txt".
  For rumor detection, we report the final binary-F1 score on the target domain (of the best performing model on the source dev data), denoted as 'test_binary_f1'.


## How to Cite PADA
```
@article{DBLP:journals/corr/abs-2102-12206,
  author    = {Eyal Ben{-}David and
               Nadav Oved and
               Roi Reichart},
  title     = {{PADA:} Example-based Prompt Learning for on-the-fly Adaptation to Unseen Domains},
  journal   = {CoRR},
  volume    = {abs/2102.12206},
  year      = {2021},
  url       = {https://arxiv.org/abs/2102.12206},
  eprinttype = {arXiv},
  eprint    = {2102.12206},
  timestamp = {Tue, 02 Mar 2021 12:11:01 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2102-12206.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
