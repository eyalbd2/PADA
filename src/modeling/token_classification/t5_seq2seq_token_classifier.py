"""
Author: Nadav Oved (@nadavo, nadavo@gmail.com), 2021.
"""

from argparse import Namespace
from collections import defaultdict
from datasets import Metric, load_metric
from pandas import DataFrame
from pytorch_lightning import LightningModule
from torch import Tensor, LongTensor
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Any, Dict, Tuple, Union
from src.utils.train_utils import NUM_CPU
from src.data_processing.absa.base import AbsaSeq2SeqDataProcessor
from transformers import (
    AdamW,
    BatchEncoding,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    get_linear_schedule_with_warmup
)
import torch as pt


class T5Seq2SeqTokenClassifier(LightningModule):
    LOSS_IGNORE_ID = -100

    def __init__(self,
                 # model_args
                 t5_model_name: str,
                 eval_metrics: List[str],

                 # model_generate_args
                 beam_size: int,
                 repetition_penalty: float,
                 length_penalty: float,
                 num_beam_groups: int,
                 diversity_penalty: float,
                 skip_special_tokens: bool,
                 clean_up_tokenization_spaces: bool,

                 # model_optimizer_args
                 weight_decay: float,
                 learning_rate: float,
                 adam_epsilon: float,

                 # trainer_args
                 train_batch_size: int,
                 eval_batch_size: int,
                 gradient_accumulation_steps: int,
                 n_gpu: int,
                 num_train_epochs: int,
                 warmup_steps: int,
                 output_dir: str,

                 # dataset_args
                 dataset_obj: Any,
                 data_procesor_obj: Any,
                 src_domains: List[str],
                 trg_domain: str,
                 data_dir: str,
                 experiment_dir: str,
                 max_seq_len: int,
                 dataset_specific_kwargs: Namespace = None):
        super().__init__()
        self.save_hyperparameters()
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.t5_model_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(self.hparams.t5_model_name)
        self.eval_metric_scorer = T5Seq2SeqTokenClassifier._init_eval_metric_scorer(self.hparams.eval_metrics)
        self.data_processor, self.datasets = self._init_datasets()
        self.eval_predictions = dict()

    @staticmethod
    def _init_eval_metric_scorer(eval_metrics) -> Dict[str, Metric]:
        return {metric: load_metric(metric.split("_")[1]) for metric in eval_metrics}

    def _init_datasets(self) -> Tuple[AbsaSeq2SeqDataProcessor, Dict[str, Dataset]]:
        data_processor = self.hparams.data_procesor_obj(self.hparams.src_domains, self.hparams.trg_domain, self.hparams.data_dir, self.hparams.experiment_dir)
        dataset_kwargs = dict(
            data_processor=data_processor,
            tokenizer=self.tokenizer,
            max_seq_len=self.hparams.max_seq_len
        )
        if self.hparams.dataset_specific_kwargs is not None:
            ### Backward Compatibility
            if isinstance(self.hparams.dataset_specific_kwargs, dict):
                dataset_kwargs.update(self.hparams.dataset_specific_kwargs)
            else:
                dataset_kwargs.update(vars(self.hparams.dataset_specific_kwargs))
        datasets = {
            split: self.hparams.dataset_obj(split=split, **dataset_kwargs)
            for split in data_processor.ALL_SPLITS
        }
        return data_processor, datasets

    def forward(self, input_ids: LongTensor, attention_mask: LongTensor, labels: LongTensor, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def generate(self, input_ids: LongTensor, attention_mask: LongTensor) -> LongTensor:
        return self.model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   max_length=self.hparams.max_seq_len,
                                   num_beams=self.hparams.beam_size,
                                   repetition_penalty=self.hparams.repetition_penalty,
                                   length_penalty=self.hparams.length_penalty,
                                   early_stopping=True,
                                   num_beam_groups=self.hparams.num_beam_groups,
                                   diversity_penalty=self.hparams.diversity_penalty)

    def generate_texts(self, input_ids: LongTensor, attention_mask: LongTensor) -> List[str]:
        """
        Generates text sentences given a batch of input token ids.
        https://huggingface.co/blog/how-to-generate
        Args:
            input_ids: tensor of shape (batch_size, sequence_length) containing the token ids for the input.
            attention_mask: tensor of shape (batch_size, sequence_length) the attention masks to avoid performing
                attention on padding token indices.
        Returns:
            A list of strings with the generated sequences.
        """
        generated_texts = self.tokenizer.batch_decode(self.generate(input_ids, attention_mask),
                                                      skip_special_tokens=self.hparams.skip_special_tokens,
                                                      clean_up_tokenization_spaces=self.hparams.clean_up_tokenization_spaces)
        return generated_texts

    def _forward_step(self, input_ids: LongTensor, attention_mask: LongTensor, labels: LongTensor) -> Tensor:
        """
        Performs a training step.
        Args:
            input_ids: tensor of shape (batch_size, sequence_length) containing the token ids for the input.
            attention_mask: tensor of shape (batch_size, sequence_length) containing the attention masks to avoid
                performing attention on padding token indices.
            labels: tensor of shape (batch_size,) with labels for computing the loss.
                Labels with T5DomainPrefixGenerator.LOSS_IGNORE_ID will be ignored.
        Returns:
            Tensor of shape (1,) with the loss after the forward pass.
        """
        # Ignore the pad token during loss calculation by replacing the pad_token_id with LOSS_IGNORE_ID.
        labels[labels[:, :] == self.tokenizer.pad_token_id] = T5Seq2SeqTokenClassifier.LOSS_IGNORE_ID
        outputs = self(input_ids=input_ids,
                       attention_mask=attention_mask,
                       labels=labels)
        return outputs.loss

    def training_step(self, batch: BatchEncoding, batch_idx: int) -> Tensor:
        """
        Compute and return the training loss.
        Args:
            batch: a dictionary with the following keys:
                input_ids: tensor of shape (batch_size, sequence_length) containing the token ids for the input.
                attention_mask: tensor of shape (batch_size, sequence_length) containing the attention masks to avoid
                    performing attention on padding token indices.
                labels: tensor of shape (batch_size,) with labels for computing the loss.
            batch_idx: index of this batch.
        Returns:
            Tensor of shape (1,) with the loss after the forward pass.
        """
        loss = self._forward_step(batch["input_ids"], batch["attention_mask"], batch["output_labels_ids"])
        self.log("train_loss", loss, on_step=False, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        """
        Called at the end of the training epoch with the outputs of all training steps.
        Args:
            outputs: List of outputs like defined in training_step(), or if there are multiple dataloaders,
                a list containing a list of outputs for each dataloader.
        """
        avg_epoch_loss = pt.stack([batch["loss"] for batch in outputs]).mean()
        self.log(f"avg_train_loss", avg_epoch_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def _eval_step(self, batch: BatchEncoding) -> Dict[str, Union[Tensor, List[str], int]]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["output_labels_ids"]
        batch_labels_texts = batch["output_labels_str"]
        loss = self._forward_step(input_ids, attention_mask, labels)
        batch_generated_texts = self.generate_texts(input_ids=input_ids, attention_mask=attention_mask)

        batch_correct_preds, batch_invalid_preds, batch_total_preds = self._evaluate_predicted_sequence_batch(batch_generated_texts, batch_labels_texts)
        eval_return_dict = dict(
            loss=loss,
            example_id=batch["example_id"],
            generated_text=batch_generated_texts,
            labels_text=batch_labels_texts,
            input_text=batch["input_str"],
            correct_preds=batch_correct_preds,
            invalid_preds=batch_invalid_preds,
            total_preds=batch_total_preds
        )
        return eval_return_dict

    def _evaluate_predicted_sequence_batch(self, batch_generated_texts: List[str], batch_labels_texts: List[str]) -> Tuple[List[str], int, int]:
        # Evaluate prediction generated sequence vs label sequence
        batch_correct_preds = list()
        batch_total_preds = 0
        batch_invalid_preds = 0
        for pred_text, labels_text in zip(batch_generated_texts, batch_labels_texts):
            correct_preds, invalid_preds = self._evaluate_predicted_sequence(pred_text, labels_text)
            batch_invalid_preds += invalid_preds
            batch_total_preds += len(correct_preds)
            batch_correct_preds.append(self.data_processor.WORD_DELIMITER.join(correct_preds))
        return batch_correct_preds, batch_invalid_preds, batch_total_preds

    def _evaluate_predicted_sequence(self, pred_text: str, labels_text: str) -> Tuple[List[str], int]:
        splitted_pred_text = pred_text.strip().split(self.data_processor.WORD_DELIMITER)
        splitted_labels_text = labels_text.strip().split(self.data_processor.WORD_DELIMITER)
        is_correct_preds = list()
        num_invalid_preds = 0
        # Edge case: model generated less tokens than labels, missing tokens are considered random mistakes
        if len(splitted_pred_text) < len(splitted_labels_text):
            i = 0
            for pred_str in splitted_pred_text:
                label_str = splitted_labels_text[i]
                if not self.data_processor.is_valid_label(pred_str):
                    num_invalid_preds += 1
                is_correct_preds.append(self._add_prediction_to_metrics(pred_str, label_str))
                i += 1
            while i < len(splitted_labels_text):
                label_str = splitted_labels_text[i]
                is_correct_preds.append(str(0))
                num_invalid_preds += 1
                for metric, scorer in self.eval_metric_scorer.items():
                    # Random wrong prediction
                    label_type = "binary" if metric.startswith("binary") else "multi"
                    labels_dict = self.data_processor.labels_dict[label_type]
                    pred = self.data_processor.get_invalid_prediction(label_str, label_type)
                    label = labels_dict[label_str]
                    scorer.add(prediction=pred, reference=label)
                i += 1
        else:
            # Edge case: if model generated more tokens than labels, extra predicted tokens are truncated from the end
            for pred_str, label_str in zip(splitted_pred_text, splitted_labels_text):
                if not self.data_processor.is_valid_label(pred_str):
                    num_invalid_preds += 1
                is_correct_preds.append(self._add_prediction_to_metrics(pred_str, label_str))
        return is_correct_preds, num_invalid_preds

    def _add_prediction_to_metrics(self, pred_str: str, label_str: str) -> str:
        correct = int(pred_str == label_str)
        for metric, scorer in self.eval_metric_scorer.items():
            # Verify the model generated a valid token
            label_type = "binary" if metric.startswith("binary") else "multi"
            labels_dict = self.data_processor.labels_dict[label_type]
            pred = labels_dict.get(pred_str, self.data_processor.get_invalid_prediction(label_str, label_type))
            label = labels_dict[label_str]
            scorer.add(prediction=pred, reference=label)
        return str(correct)

    def _eval_epoch_end(self, outputs: List[Dict[str, Union[Tensor, List[str], int]]], split: str):
        """
        Called at the end of the validation epoch with.
        Args:
            outputs: the outputs of all validation steps.
        """
        epoch_eval_dict = defaultdict(list)
        for batch_eval_dict in outputs:
            for k, v in batch_eval_dict.items():
                if isinstance(v, list):
                    epoch_eval_dict[k].extend(v)
                else:
                    epoch_eval_dict[k].append(v)

        avg_epoch_loss = pt.stack(epoch_eval_dict.pop("loss")).mean()
        self.log(f"avg_{split}_loss", avg_epoch_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        epoch_invalid = sum(epoch_eval_dict.pop("invalid_preds"))
        epoch_total = sum(epoch_eval_dict.pop("total_preds"))
        self.log(f"{split}_invalid_preds", float(epoch_invalid) / epoch_total, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.eval_predictions[split] = DataFrame(epoch_eval_dict)

        self._calc_epoch_metric_scores(split)

    def _calc_epoch_metric_scores(self, split):
        try:
            for metric, scorer in self.eval_metric_scorer.items():
                average_type = metric.split("_")[0]
                if average_type == "binary":
                    epoch_scores = scorer.compute()
                    for score_name, score_value in epoch_scores.items():
                        self.log(f"{split}_{average_type}_{score_name}", score_value, on_step=False, on_epoch=True,
                                 prog_bar=True, logger=True)
                else:
                    epoch_scores = scorer.compute(average=average_type,
                                                  labels=tuple(self.data_processor.labels_dict["multi"].values()))
                    for score_name, score_value in epoch_scores.items():
                        self.log(f"{split}_{average_type}_{score_name}", score_value, on_step=False, on_epoch=True,
                                 prog_bar=True, logger=True)
        except Exception as e:
            print(f"WARN: Score calculation failed\n{e}")

    def validation_step(self, batch: BatchEncoding, batch_idx: int):
        """
        Operates on a single batch of data from the validation set.
        This step is used to generate examples or calculate anything of interest like accuracy.
        Args:
            batch: the output of the DataLoader
            batch_idx: the index of this batch.
        Returns:
            A tuple of (loss, generated_texts, labels_texts, sample_ids)
        """
        return self._eval_step(batch)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self._eval_epoch_end(outputs, "dev")

    def test_step(self, batch: BatchEncoding, batch_idx: int):
        """
        Operates on a single batch of data from the test set.
        This step is used to generate examples or calculate anything of interest like accuracy.
        Args:
            batch: the output of DataLoader
            batch_idx: the index of this batch.
        Returns:
            A tuple of (loss, generated_texts, labels_texts, sample_ids)
        """
        return self._eval_step(batch)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        """
        Called at the end of the test epoch.
        Args:
            outputs: the outputs of all test steps.
        """
        self._eval_epoch_end(outputs, "test")

    def configure_optimizers(self):
        # TODO: Add link to original snippet
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        t_total = (
                (len(self.datasets["train"]) // (self.hparams.train_batch_size * float(max(1, self.hparams.n_gpu))))
                // self.hparams.gradient_accumulation_steps * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=t_total)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(self.datasets["train"], batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=NUM_CPU)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.datasets["dev"], batch_size=self.hparams.eval_batch_size, num_workers=NUM_CPU)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.datasets["test"], batch_size=self.hparams.eval_batch_size, num_workers=NUM_CPU)

    def write_eval_predictions(self, output_dir: Union[str, Path], split: str) -> None:
        if not output_dir:
            output_dir = Path(self.hparams.output_dir)
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        if not split:
            split = "test"
        output_dir.mkdir(exist_ok=True)
        self.eval_predictions[split].to_json(output_dir / f"{split}_generated.jsonl",
                                             orient="records", lines=True, default_handler=str)
