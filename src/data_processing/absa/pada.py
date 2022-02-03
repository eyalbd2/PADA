from collections import defaultdict
from numpy import percentile
from random import random
from pathlib import Path
from typing import List, Union, Dict
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast, BatchEncoding
from src.utils.constants import DATA_DIR, EXP_DIR
from src.data_processing.absa.base import AbsaSeq2SeqDataProcessor
import torch


class AbsaSeq2SeqPadaDataProcessor(AbsaSeq2SeqDataProcessor):

    def __init__(self, src_domains: List[str], trg_domain: str, data_dir: Union[str, Path], experiment_dir: Union[str, Path]):
        super().__init__(src_domains, trg_domain, data_dir)
        self.experiment_dir = experiment_dir
        for split in AbsaSeq2SeqDataProcessor.ALL_SPLITS:
            self.data[split]["drfs"] = self._read_drfs(split)

    def _read_drfs(self, mode: str = 'train'):
        domains = self.src_domains if mode != 'test' else [self.trg_domain]
        if mode == 'test' or mode == 'dev':
            return ['' for _ in range(len(self.data[mode]["input_str"]))]
        all_drfs = list()
        for domain in domains:
            prompts_path = Path(self.experiment_dir) / "absa" / self.trg_domain / "prompt_annotations" / domain
            with open(prompts_path / f"annotated_prompts_train.pt", "rb") as f:
                drfs = torch.load(f)
            all_drfs.extend(drfs)
        return all_drfs


def test_AbsaSeq2SeqPadaDataProcessor():
    data_processor = AbsaSeq2SeqPadaDataProcessor(["device", "laptops", "rest", "service"], "service", DATA_DIR, EXP_DIR)
    print()
    print(data_processor.data["dev"].keys())
    print(len(list(data_processor.data["dev"].values())[0]))
    for k, v in data_processor.data["dev"].items():
        if type(v) is not int:
            print(k, v[-80])
        else:
            print(k, v[-80], len(v[-80]))
    tokenizer = T5TokenizerFast.from_pretrained("t5-base")
    percent = 99.8
    for split in AbsaSeq2SeqPadaDataProcessor.ALL_SPLITS:
        print(split, len(data_processor.data[split]["example_id"]))
        print(split, max(data_processor.data[split]["input_tokens_len"]))
        print()
        tokenizer_lens = tokenizer(data_processor.data[split]["input_tokens"],
                                   is_split_into_words=True,
                                   return_length=True)["length"]
        print(split, max(tokenizer_lens))
        print(split, percentile(tokenizer_lens, percent))
        print()
        tokenizer_lens = tokenizer(data_processor.data[split]["output_labels_tokens"],
                                   is_split_into_words=True,
                                   return_length=True)["length"]
        print(split, max(tokenizer_lens))
        print(split, percentile(tokenizer_lens, percent))
        break


class AbsaSeq2SeqPadaDataset(Dataset):

    DOMAIN_PROMPT = "domain"
    ASPECT_PROMPT = "aspect"
    DRF_DELIMITER = ", "

    def __init__(self, split: str, data_processor: AbsaSeq2SeqPadaDataProcessor, tokenizer: T5TokenizerFast,
                 max_seq_len: int, mixture_alpha: float = 0.1, proportion_aspect: float = 0.3333):
        self.split = split
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mixture_alpha = mixture_alpha
        self.proportion_aspect = proportion_aspect
        self.tokenized_data = self._init_tokenized_data(split, data_processor)

    def __len__(self):
        if self.split == "train":
            return self.tokenized_data["num_examples"]
        else:
            return len(self.tokenized_data["example_id"])

    def __getitem__(self, index):
        if self.split == "train":
            if random() <= self.mixture_alpha:
                tokenized_data = self.tokenized_data["domain"]
            else:
                if random() <= self.proportion_aspect:
                    tokenized_data = self.tokenized_data["aspect"]
                else:
                    tokenized_data = self.tokenized_data["drf"]
        else:
            tokenized_data = self.tokenized_data
        return {
            k: v[index]
            for k, v in tokenized_data.items()
        }

    def tokenize_input_tokens(self, input_tokens: List[List[str]]) -> BatchEncoding:
        return self.tokenizer(input_tokens, is_split_into_words=True,
                              padding='max_length', truncation=True,
                              max_length=self.max_seq_len, return_tensors="pt", return_attention_mask=True)

    def tokenize_output_tokens(self, input_tokens: List[List[str]]) -> torch.LongTensor:
        return self.tokenizer(input_tokens, is_split_into_words=True,
                              padding='max_length', truncation=True,
                              max_length=self.max_seq_len, return_tensors="pt")["input_ids"]

    @staticmethod
    def _build_training_tokenized_data(data):
        domain_data = defaultdict(list)
        aspect_data = defaultdict(list)
        drf_data = defaultdict(list)
        for i in range(len(data["example_id"])):
            drfs_str = f"{data['domain_label'][i]}{AbsaSeq2SeqPadaDataset.DRF_DELIMITER}{data['drfs'][i]}"
            prompt = f"{AbsaSeq2SeqPadaDataset.DOMAIN_PROMPT}:"
            domain_data["input_tokens"].append([prompt] + data["input_tokens"][i])
            domain_data["output_labels_str"].append(drfs_str)
            domain_data["output_labels_tokens"].append(drfs_str.split(AbsaSeq2SeqPadaDataset.DRF_DELIMITER))
            domain_data["prompt"].append(prompt)

            prompt = f"{AbsaSeq2SeqPadaDataset.ASPECT_PROMPT}:"
            aspect_data["input_tokens"].append([prompt] + data["input_tokens"][i])
            aspect_data["output_labels_str"].append(data["output_labels_str"][i])
            aspect_data["output_labels_tokens"].append(data["output_labels_tokens"][i])
            aspect_data["prompt"].append(prompt)

            prompt = f"{drfs_str}:"
            drf_data["input_tokens"].append(prompt.split(AbsaSeq2SeqPadaDataset.DRF_DELIMITER) + data["input_tokens"][i])
            drf_data["output_labels_str"].append(data["output_labels_str"][i])
            drf_data["output_labels_tokens"].append(data["output_labels_tokens"][i])
            drf_data["prompt"].append(prompt)
        return {"domain": domain_data, "aspect": aspect_data, "drf": drf_data}

    def _init_tokenized_data(self, split, data_processor) -> Union[BatchEncoding, Dict[str, BatchEncoding]]:
        data = data_processor.get_split_data(split)
        if split == "train":
            tokenized_data_dict = dict(num_examples=len(data["example_id"]))
            for key, data_dict in self._build_training_tokenized_data(data).items():
                tokenized_data = self.tokenize_input_tokens(data_dict["input_tokens"])
                tokenized_data["output_labels_ids"] = self.tokenize_output_tokens(data_dict["output_labels_tokens"])
                tokenized_data["output_labels_str"] = data_dict["output_labels_str"]
                tokenized_data["prompt"] = data_dict["prompt"]
                tokenized_data["input_str"] = data["input_str"]
                tokenized_data["domain_label"] = data["domain_label"]
                tokenized_data["drfs"] = data["drfs"]
                tokenized_data["example_id"] = data["example_id"]
                tokenized_data_dict[key] = tokenized_data
            return tokenized_data_dict
        else:
            tokenized_data = self.tokenize_input_tokens(data["input_tokens"])
            tokenized_data["output_labels_ids"] = self.tokenize_output_tokens(data["output_labels_tokens"])
            tokenized_data["output_labels_str"] = data["output_labels_str"]
            tokenized_data["input_str"] = data["input_str"]
            tokenized_data["domain_label"] = data["domain_label"]
            tokenized_data["drfs"] = data["drfs"]
            tokenized_data["example_id"] = data["example_id"]
        return tokenized_data


def test_AbsaSeq2SeqPadaDataset():
    split = "train"
    data_processor = AbsaSeq2SeqPadaDataProcessor(["device", "laptops", "rest"], "service", DATA_DIR, EXP_DIR)
    print()
    print(data_processor.data[split].keys())
    print(len(list(data_processor.data[split].values())[0]))
    for k, v in data_processor.data[split].items():
        if type(v) is not int:
            print(k, v[0])
        else:
            print(k, v[0], len(v[0]))
    dataset = AbsaSeq2SeqPadaDataset(split, data_processor, T5TokenizerFast.from_pretrained("t5-base"), 64, 0.5, 0.5)
    print(len(dataset))
    for i, example in enumerate(dataset):
        for k, v in example.items():
            print(k, v)
        if i == 8:
            break
    print()
    dataloader = DataLoader(dataset, 8)
    for batch in dataloader:
        for k, v in batch.items():
            print(k, v)
        break

