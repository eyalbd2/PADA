"""
Author: Nadav Oved (@nadavo, nadavo@gmail.com), 2021.
"""

from argparse import Namespace
from collections import defaultdict, Counter
from torch import Tensor, LongTensor
from typing import List, Union, Dict, Tuple, Callable
from src.modeling.token_classification.t5_seq2seq_token_classifier import T5Seq2SeqTokenClassifier
from transformers import BatchEncoding, T5TokenizerFast
from math import ceil
import torch as pt


class PadaSeq2SeqTokenClassifierGenerator(T5Seq2SeqTokenClassifier):

    def __init__(self, mixture_alpha: float = 0.1, proportion_aspect: float = 0.3333, **kwargs):
        ### Backward compatibility hack
        if "dataset_specific_kwargs" in kwargs:
            super().__init__(**kwargs)
        else:
            dataset_specific_kwargs = Namespace(**{"mixture_alpha": mixture_alpha, "proportion_aspect": proportion_aspect})
            super().__init__(dataset_specific_kwargs=dataset_specific_kwargs, **kwargs)
            self.save_hyperparameters(dataset_specific_kwargs)
        self.tokenized_domain_prompt = PadaSeq2SeqTokenClassifierGenerator._init_tokenized_domain_prompt(self.datasets['dev'].DOMAIN_PROMPT, self.tokenizer)

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
            if self.datasets["dev"].DRF_DELIMITER in prompt:
                prompt_tokens = f"{prompt}:".split(self.datasets["dev"].DRF_DELIMITER)
            elif self.data_processor.WORD_DELIMITER in prompt:
                prompt_tokens = f"{prompt}:".split(self.data_processor.WORD_DELIMITER)
            else:
                prompt_tokens = [f"{prompt}:"]
            input_tokens = input_str.split(self.data_processor.WORD_DELIMITER)
            batch_gen_prompt_input.append(prompt_tokens + input_tokens)
        batch_gen_prompt_input_tokenized = self.datasets["dev"].tokenize_input_tokens(batch_gen_prompt_input)
        gen_prompt_input_ids = batch_gen_prompt_input_tokenized["input_ids"].to(self.device)
        gen_prompt_attention_mask = batch_gen_prompt_input_tokenized["attention_mask"].to(self.device)
        return gen_prompt_input_ids, gen_prompt_attention_mask

    def _eval_step(self, batch: BatchEncoding) -> Dict[str, Union[Tensor, List[str], int]]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels_ids = batch["output_labels_ids"]
        batch_labels_texts = batch["output_labels_str"]
        batch_input_texts = batch["input_str"]
        domain_input_ids, domain_attention_mask = self._get_domain_prompted_tensors(input_ids, attention_mask)
        batch_generated_prompts = self.generate_texts(domain_input_ids, domain_attention_mask)
        gen_prompt_input_ids, gen_prompt_attention_mask = self._get_gen_prompted_tensors(batch_generated_prompts, batch_input_texts)
        loss = self._forward_step(gen_prompt_input_ids, gen_prompt_attention_mask, labels_ids)
        batch_generated_texts = self.generate_texts(gen_prompt_input_ids, gen_prompt_attention_mask)
        batch_correct_preds, batch_invalid_preds, batch_total_preds = self._evaluate_predicted_sequence_batch(batch_generated_texts, batch_labels_texts)
        eval_return_dict = dict(
            loss=loss,
            example_id=batch["example_id"],
            input_text=batch["input_str"],
            generated_prompt=batch_generated_prompts,
            generated_text=batch_generated_texts,
            labels_text=batch_labels_texts,
            correct_preds=batch_correct_preds,
            invalid_preds=batch_invalid_preds,
            total_preds=batch_total_preds
        )
        return eval_return_dict


class PadaSeq2SeqTokenClassifierGeneratorMulti(PadaSeq2SeqTokenClassifierGenerator):

    def __init__(self, num_return_sequences: int = 4, multi_diversity_penalty: float = 0.2, **kwargs):
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
                                   diversity_penalty=self.hparams.multi_diversity_penalty).view(input_ids.size(0), self.hparams.num_return_sequences, -1)

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
    def _vote_multiple_generated_texts(batch_multi_generated_texts: List[List[str]], token_delimiter: str, token_validation_fn: Callable) -> List[str]:
        """
        Receives a list (num_return_sequences) of lists (batch_size) of generated sequences (strings).
        Returns a list (batch_size) of the final voted generated sequences (strings).
        """
        num_sequences = len(batch_multi_generated_texts)
        majority = ceil(num_sequences / 2)
        for text in batch_multi_generated_texts:
            assert len(batch_multi_generated_texts[0]) == len(text)
        batch_size = len(batch_multi_generated_texts[0])

        voted_generated_texts = list()
        for example_idx in range(batch_size):
            generated_text_votes = PadaSeq2SeqTokenClassifierGeneratorMulti._count_token_votes(example_idx, num_sequences,
                                                                                               batch_multi_generated_texts,
                                                                                               token_delimiter, token_validation_fn)
            voted_generated_text = PadaSeq2SeqTokenClassifierGeneratorMulti._collect_top_voted_tokens(generated_text_votes, majority)
            # Add majority voted sequence to final list
            voted_generated_texts.append(token_delimiter.join(voted_generated_text))
        return voted_generated_texts

    @staticmethod
    def _count_token_votes(example_idx: int, num_sequences: int, batch_multi_generated_texts: List[List[str]], token_delimiter: str, token_validation_fn: Callable) -> List[Counter]:
        """
        Count votes per valid token
        """
        generated_text_token_votes = list()
        for sequence_idx in range(num_sequences):
            generated_text = batch_multi_generated_texts[sequence_idx][example_idx]
            generated_text_tokens = generated_text.strip().split(token_delimiter)
            for k, gen_token in enumerate(generated_text_tokens):
                if k == len(generated_text_token_votes):
                    generated_text_token_votes.append(Counter())
                if token_validation_fn(gen_token):
                    generated_text_token_votes[k][gen_token] += 1
        return generated_text_token_votes

    @staticmethod
    def _collect_top_voted_tokens(generated_text_votes: List[Counter], majority: int) -> List[str]:
        """
        Collect top voted token for each position in sequence
        """
        voted_generated_text = list()
        for counter in generated_text_votes:
            total_token_votes = 0
            top_voted_token = None
            for token, votes in counter.most_common():
                if top_voted_token is None:
                    top_voted_token = token
                total_token_votes += votes
            # Check if there is a top voted valid token in this position and if majority of sequences had a token in this position
            if top_voted_token is not None and total_token_votes >= majority:
                voted_generated_text.append(top_voted_token)
        return voted_generated_text

    def test_step(self, batch: BatchEncoding, batch_idx: int) -> Dict[str, Union[Tensor, List[str], int]]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels_ids = batch["output_labels_ids"]
        batch_labels_texts = batch["output_labels_str"]
        batch_input_texts = batch["input_str"]
        # Generate multiple prompts
        domain_input_ids, domain_attention_mask = self._get_domain_prompted_tensors(input_ids, attention_mask)
        batch_multi = defaultdict(list)
        batch_multi["generated_prompts"] = self.generate_multiple_texts(domain_input_ids, domain_attention_mask)
        batch_multi["generated_prompts"].append([self.datasets["test"].ASPECT_PROMPT] * input_ids.size(0))
        for batch_gen_prompts in batch_multi["generated_prompts"]:
            gen_prompt_input_ids, gen_prompt_attention_mask = self._get_gen_prompted_tensors(batch_gen_prompts, batch_input_texts)
            batch_multi["loss"].append(self._forward_step(gen_prompt_input_ids, gen_prompt_attention_mask, labels_ids))
            batch_multi["generated_texts"].append(self.generate_texts(gen_prompt_input_ids, gen_prompt_attention_mask))

        batch_voted_generated_texts = self._vote_multiple_generated_texts(batch_multi["generated_texts"], self.data_processor.WORD_DELIMITER, self.data_processor.is_valid_label)

        batch_correct_preds, batch_invalid_preds, batch_total_preds = self._evaluate_predicted_sequence_batch(batch_voted_generated_texts, batch_labels_texts)
        loss = pt.stack(batch_multi.pop("loss")).mean()

        eval_return_dict = dict(
            loss=loss,
            example_id=batch["example_id"],
            input_text=batch["input_str"],
            majority_voted_text=batch_voted_generated_texts,
            labels_text=batch_labels_texts,
            correct_preds=batch_correct_preds,
            invalid_preds=batch_invalid_preds,
            total_preds=batch_total_preds,
        )

        for k, v in batch_multi.items():
            batch_examples = list()
            for i in range(len(v[0])):
                return_sequences = list()
                for j in range(len(v)):
                    return_sequences.append(v[j][i])
                batch_examples.append(return_sequences)
            eval_return_dict[k] = batch_examples

        return eval_return_dict
