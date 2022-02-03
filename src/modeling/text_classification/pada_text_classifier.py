from argparse import Namespace
from collections import defaultdict
from torch import Tensor, LongTensor
from typing import List, Union, Dict, Tuple
from src.modeling.text_classification.t5_text_classifier import T5TextClassifier
from transformers import BatchEncoding, T5TokenizerFast
import torch as pt


class PadaTextClassifier(T5TextClassifier):

    def __init__(self, mixture_alpha: float = 0.2, max_drf_seq_len: int = 20, gen_constant: float = 1.0, **kwargs):
        ### Backward compatibility hack
        if "dataset_specific_kwargs" in kwargs:
            super().__init__(**kwargs)
        else:
            dataset_specific_kwargs = Namespace(
                **{"mixture_alpha": mixture_alpha, "max_drf_seq_len": max_drf_seq_len})
            super().__init__(dataset_specific_kwargs=dataset_specific_kwargs, **kwargs)
            self.save_hyperparameters(dataset_specific_kwargs, gen_constant)
        self.gen_constant = gen_constant
        self.tokenized_domain_prompt = PadaTextClassifier._init_tokenized_domain_prompt(self.datasets['dev'].DOMAIN_PROMPT, self.tokenizer)

    def forward(self, input_ids: LongTensor, attention_mask: LongTensor, labels: LongTensor, decoder_labels: LongTensor,
                **kwargs):
        if decoder_labels is not None:  # train
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=decoder_labels)
            gen_loss, encoder_outputs = outputs[0], outputs[3]
            cls_logits = self.classifier(encoder_outputs)
            cls_loss = self.loss_fn(cls_logits.view(-1, self.hparams.num_labels), labels.view(-1))
            loss = cls_loss + (self.gen_constant*gen_loss)
            return {"loss": loss, "logits": cls_logits, "cls_loss": cls_loss, "gen_loss": gen_loss}
        else:  # dev & test
            encoder_outputs = self.model.encoder(input_ids, attention_mask=attention_mask)[0]
            cls_logits = self.classifier(encoder_outputs)
            cls_loss = self.loss_fn(cls_logits.view(-1, self.hparams.num_labels), labels.view(-1))
            loss = cls_loss
            return {"loss": loss, "logits": cls_logits, "cls_loss": cls_loss, "gen_loss": 0}

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

    def _forward_step(self, input_ids: LongTensor, attention_mask: LongTensor, labels: LongTensor,
                      decoder_labels: LongTensor) -> Tensor:
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
        outputs = self(input_ids=input_ids,
                       attention_mask=attention_mask,
                       labels=labels,
                       decoder_labels=decoder_labels)
        return outputs

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
        loss = self._forward_step(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                  labels=batch["output_label_id"], decoder_labels=batch["prompt_output_ids"])["loss"]
        self.log("train_loss", loss, on_step=False, on_epoch=False, prog_bar=True, logger=True)
        return loss

    @staticmethod
    def _init_tokenized_domain_prompt(domain_prompt: str, tokenizer: T5TokenizerFast) -> BatchEncoding:
        tokenized_domain_prompt = tokenizer(f"{domain_prompt}:", add_special_tokens=False)
        tokenized_domain_prompt["input_ids"].append(3)
        tokenized_domain_prompt["input_ids"] = pt.tensor(tokenized_domain_prompt["input_ids"])
        tokenized_domain_prompt["attention_mask"] = pt.ones_like(tokenized_domain_prompt["input_ids"])
        return tokenized_domain_prompt

    def _get_domain_prompted_tensors(self, input_ids: LongTensor, attention_mask: LongTensor) -> Tuple[LongTensor, LongTensor]:
        domain_input_ids = pt.cat((self.tokenized_domain_prompt["input_ids"].to(self.device).expand(input_ids.size(0), -1), input_ids), dim=-1)
        domain_attention_mask = pt.cat((self.tokenized_domain_prompt["attention_mask"].to(self.device).expand(attention_mask.size(0), -1), attention_mask), dim=-1)
        return domain_input_ids, domain_attention_mask

    def _get_gen_prompted_tensors(self, batch_generated_prompts: List[str], batch_input_texts: List[str]) -> Tuple[LongTensor, LongTensor]:
        batch_gen_prompt_input = list()
        for prompt, input_str in zip(batch_generated_prompts, batch_input_texts):
            batch_gen_prompt_input.append(prompt + ': ' + input_str)
        batch_gen_prompt_input_tokenized = self.datasets["dev"].tokenize_input_str(batch_gen_prompt_input)
        gen_prompt_input_ids = batch_gen_prompt_input_tokenized["input_ids"].to(self.device)
        gen_prompt_attention_mask = batch_gen_prompt_input_tokenized["attention_mask"].to(self.device)
        return gen_prompt_input_ids, gen_prompt_attention_mask

    def _eval_step(self, batch: BatchEncoding) -> Dict[str, Union[Tensor, List[str], int]]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        label_ids = batch["output_label_id"]
        batch_input_texts = batch["input_str"]
        domain_input_ids, domain_attention_mask = self._get_domain_prompted_tensors(input_ids, attention_mask)
        batch_generated_prompts = self.generate_texts(domain_input_ids, domain_attention_mask)
        gen_prompt_input_ids, gen_prompt_attention_mask = self._get_gen_prompted_tensors(batch_generated_prompts,
                                                                                         batch_input_texts)
        outputs = self._forward_step(input_ids=gen_prompt_input_ids, attention_mask=gen_prompt_attention_mask,
                                     labels=label_ids, decoder_labels=None)
        loss, logits = outputs["loss"], outputs["logits"]
        preds = logits.detach().cpu().argmax(dim=-1).tolist()
        label_ids = label_ids.cpu().tolist()
        self._evaluate_predicted_batch(preds, label_ids)
        eval_return_dict = dict(
            loss=loss,
            example_id=batch["example_id"],
            input_text=batch["input_str"],
            preds=preds,
            labels=label_ids,
            generated_prompt=batch_generated_prompts,
        )
        return eval_return_dict


class PadaTextClassifierMulti(PadaTextClassifier):

    def __init__(self, num_return_sequences: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def generate_multi(self, input_ids: LongTensor, attention_mask: LongTensor) -> LongTensor:
        """
        Returns generated ids tensor of size (batch_size, num_return_sequences, seq_len)
        """
        return self.model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   max_length=self.hparams.max_seq_len,
                                   num_beams=self.hparams.beam_size,
                                   repetition_penalty=self.hparams.repetition_penalty,
                                   length_penalty=self.hparams.length_penalty,
                                   early_stopping=True,
                                   num_return_sequences=self.hparams.num_return_sequences,
                                   num_beam_groups=self.hparams.num_beam_groups,
                                   diversity_penalty=self.hparams.diversity_penalty).view(input_ids.size(0), self.hparams.num_return_sequences, -1)

    def generate_multiple_texts(self, input_ids: LongTensor, attention_mask: LongTensor) -> List[List[str]]:
        """
        Generates text sequences given a batch of input token ids.
        https://huggingface.co/blog/how-to-generate
        Args:
            input_ids: tensor of shape (batch_size, sequence_length) containing the token ids for the input.
            attention_mask: tensor of shape (batch_size, sequence_length) the attention masks to avoid performing
                attention on padding token indices.
        Returns:
            A list (num_return_sequences) of lists (batch_size) of strings with the generated sequences.
        """
        outputs = self.generate_multi(input_ids, attention_mask)
        batch_multi_generated_texts = [
            self.tokenizer.batch_decode(outputs[:, i, :],
                                        skip_special_tokens=self.hparams.skip_special_tokens,
                                        clean_up_tokenization_spaces=self.hparams.clean_up_tokenization_spaces)
            for i in range(outputs.size(1))
        ]
        return batch_multi_generated_texts

    @staticmethod
    def _enssemble_preds(batch_multi_logits: List[List[str]]) -> List[int]:
        """
        Receives a list (num_return_sequences) of lists (batch_size) of generated sequences (strings).
        Returns a list (batch_size) of the final voted generated sequences (strings).
        """
        ens_logits = sum(batch_multi_logits)
        ens_preds = ens_logits.detach().cpu().argmax(dim=-1).tolist()
        return ens_preds

    def test_step(self, batch: BatchEncoding, batch_idx: int) -> Dict[str, Union[Tensor, List[str], int]]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        label_ids = batch["output_label_id"]
        batch_input_texts = batch["input_str"]
        # Generate multiple prompts
        domain_input_ids, domain_attention_mask = self._get_domain_prompted_tensors(input_ids, attention_mask)
        batch_multi = defaultdict(list)
        batch_multi["generated_prompts"] = self.generate_multiple_texts(domain_input_ids, domain_attention_mask)
        batch_multi["generated_prompts"].append([self.datasets["test"].RUMOR_PROMPT] * input_ids.size(0))
        for batch_gen_prompts in batch_multi["generated_prompts"]:
            gen_prompt_input_ids, gen_prompt_attention_mask = self._get_gen_prompted_tensors(batch_gen_prompts,
                                                                                             batch_input_texts)
            outputs = self._forward_step(input_ids=gen_prompt_input_ids, attention_mask=gen_prompt_attention_mask,
                                         labels=label_ids, decoder_labels=None)

            batch_multi["loss"].append(outputs["loss"])
            batch_multi["logits"].append(outputs["logits"])

        preds = self._enssemble_preds(batch_multi["logits"])
        loss = pt.stack(batch_multi.pop("loss")).mean()
        label_ids = label_ids.cpu().tolist()
        self._evaluate_predicted_batch(preds, label_ids)

        eval_return_dict = dict(
            loss=loss,
            example_id=batch["example_id"],
            preds=preds,
            labels=label_ids,
            input_text=batch["input_str"]
        )

        batch_examples = list()
        for i in range(len(batch_multi["generated_prompts"][0])):
            return_sequences = list()
            for j in range(len(batch_multi["generated_prompts"])):
                return_sequences.append(batch_multi["generated_prompts"][j][i])
            batch_examples.append(return_sequences)
        eval_return_dict["generated_prompts"] = batch_examples

        return eval_return_dict

