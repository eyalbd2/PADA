from src.utils.constants import DATA_DIR, EXP_DIR
from src.data_processing.absa.pada import AbsaSeq2SeqPadaDataProcessor, AbsaSeq2SeqPadaDataset
from src.data_processing.rumor.pada import RumorPadaDataProcessor, RumorPadaDataset
from src.modeling.token_classification.pada_seq2seq_token_classifier import PadaSeq2SeqTokenClassifierGeneratorMulti
from src.modeling.text_classification.pada_text_classifier import PadaTextClassifierMulti
from src.utils.train_utils import set_seed, ModelCheckpointWithResults, LoggingCallback
from pathlib import Path
from argparse import Namespace, ArgumentParser
from pytorch_lightning import Trainer
from syct import timer

SUPPORTED_MODELS = {
    "PADA-rumor": (PadaTextClassifierMulti, RumorPadaDataProcessor, RumorPadaDataset),
    "PADA-absa": (PadaSeq2SeqTokenClassifierGeneratorMulti, AbsaSeq2SeqPadaDataProcessor, AbsaSeq2SeqPadaDataset),
}

SUPPORTED_DATASETS = {
    "rumor",
    "absa"
}

args_dict = dict(
    model_name="PADA",
    dataset_name="rumor",
    src_domains="charliehebdo,ferguson,germanwings-crash,ottawashooting",
    trg_domain="sydneysiege",
    data_dir=str(DATA_DIR),  # path to data files
    experiment_dir=str(EXP_DIR),  # path to base experiment dir
    output_dir=str(EXP_DIR),  # path to save the checkpoints
    t5_model_name='t5-base',
    max_seq_len=128,
    learning_rate=5e-5,
    weight_decay=1e-5,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=32,
    eval_batch_size=32,
    num_train_epochs=5,
    gradient_accumulation_steps=1,
    n_gpu=1,
    fast_dev_run=False,
    fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1',  # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=41,
    beam_size=10,
    repetition_penalty=2.0,
    length_penalty=1.0,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
    num_return_sequences=4,
    num_beam_groups=5,
    diversity_penalty=0.2,
    eval_metrics=["binary_f1", "micro_f1", "macro_f1", "weighted_f1"],
    mixture_alpha=0.2,
    max_drf_seq_len=20,
    proportion_aspect=0.3333,
    gen_constant=1.0,
    multi_diversity_penalty=0.2,
)


@timer
def train_pada_experiment(args):
    if isinstance(args, Namespace):
        hparams = args
    elif isinstance(args, dict):
        hparams = Namespace(**args)

    hparams.src_domains = hparams.src_domains.split(",")

    experiment_name = f"{hparams.dataset_name.lower()}_{hparams.trg_domain}_{hparams.model_name}_e{hparams.num_train_epochs}_b{hparams.train_batch_size}_a{hparams.mixture_alpha}"
    hparams.output_dir = Path(hparams.output_dir) / experiment_name.replace("_", "/")
    hparams.output_dir.mkdir(exist_ok=True, parents=True)
    hparams.output_dir = str(hparams.output_dir)

    main_eval_metric = "binary_f1"
    checkpoint_callback = ModelCheckpointWithResults(dirpath=hparams.output_dir,
                                                     filename=f"best_dev_{main_eval_metric}",
                                                     monitor=f"dev_{main_eval_metric}",
                                                     mode="max",
                                                     save_top_k=1)
    logging_callback = LoggingCallback()
    logger = True
    callbacks = [logging_callback, checkpoint_callback]
    test_ckpt = "best"

    model_hparams_dict = vars(hparams)

    train_args = dict(
            default_root_dir=model_hparams_dict["output_dir"],
            accumulate_grad_batches=model_hparams_dict["gradient_accumulation_steps"],
            gpus=model_hparams_dict["n_gpu"],
            max_epochs=model_hparams_dict["num_train_epochs"],
            precision=16 if model_hparams_dict.pop("fp_16") else 32,
            amp_level=model_hparams_dict.pop("opt_level"),
            gradient_clip_val=model_hparams_dict.pop("max_grad_norm"),
            callbacks=callbacks,
            logger=logger,
            fast_dev_run=model_hparams_dict.pop("fast_dev_run"),
            deterministic=True,
            benchmark=False
        )

    set_seed(model_hparams_dict.pop("seed"))
    dataset_name = model_hparams_dict.pop("dataset_name")
    if dataset_name  == "rumor":
        model_hparams_dict.pop("proportion_aspect")
        model_hparams_dict.pop("multi_diversity_penalty")
    else:
        model_hparams_dict.pop("max_drf_seq_len")
        model_hparams_dict.pop("gen_constant")
    model_name = model_hparams_dict.pop("model_name")

    model_obj, data_procesor_obj, dataset_obj = SUPPORTED_MODELS[f"{model_name}-{dataset_name}"]
    model_hparams_dict["data_procesor_obj"] = data_procesor_obj
    model_hparams_dict["dataset_obj"] = dataset_obj

    model = model_obj(**model_hparams_dict)
    trainer = Trainer(**train_args)
    trainer.fit(model)
    trainer.test(ckpt_path=test_ckpt)


def main():
    parser = ArgumentParser()
    for key, val in args_dict.items():
        if key == "dataset_name":
            parser.add_argument(f"--{key}", default=val, type=type(val),
                                choices=SUPPORTED_DATASETS)
        elif key == "model_name":
            parser.add_argument(f"--{key}", default=val, type=type(val),
                                choices=("PADA",))
        elif type(val) is bool:
            parser.add_argument(f"--{key}", default=val, action="store_true", required=False)
        else:
            parser.add_argument(f"--{key}", default=val, type=type(val), required=False)
    args = parser.parse_args()
    train_pada_experiment(args)


if __name__ == "__main__":
    main()
