from numpy import percentile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Union, Tuple
from random import choice
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast
from src.utils.constants import DATA_DIR
import pickle


class AbsaSeq2SeqDataProcessor:

    ALL_SPLITS = ("train", "dev", "test")
    WORD_DELIMITER = " "

    def __init__(self, src_domains: List[str], trg_domain: str, data_dir: Union[str, Path]):
        self.data_dir = data_dir
        self.src_domains = src_domains
        self.trg_domain = trg_domain
        self.reduce_label_dict = {
            'O': 'O',
            'B-AS': 'B',
            'I-AS': 'I'
        }
        self.labels_dict = {
            'binary': {'O': 0, 'B': 1, 'I': 1},
            'multi': {'O': 0, 'B': 1, 'I': 2},
        }
        self.data = self.load_data()

    def read_data_from_file(self, mode: str = 'train') -> Dict[str, List[Union[str, List[str], Tuple[int]]]]:
        domains = self.src_domains if mode != 'test' else [self.trg_domain]
        data_dict = defaultdict(list)
        for domain_idx, domain in enumerate(domains):
            data_path = Path(self.data_dir) / "absa_data" / domain / mode
            with open(data_path, 'rb') as f:
                (text, labels) = pickle.load(f)
            for i, (txt, lbls) in enumerate(zip(text, labels)):
                data_dict["input_str"].append(AbsaSeq2SeqDataProcessor.WORD_DELIMITER.join(txt))
                data_dict["input_tokens"].append(txt)
                data_dict["input_tokens_len"].append(len(txt))
                reduced_labels = tuple([self.reduce_label_dict[lbl] for lbl in lbls])
                data_dict["output_labels_tokens"].append(reduced_labels)
                data_dict["output_labels_str"].append(AbsaSeq2SeqDataProcessor.WORD_DELIMITER.join(reduced_labels))
                for k in self.labels_dict:
                    data_dict[f"output_labels_{k}"].append(tuple([self.labels_dict[k][lbl] for lbl in reduced_labels]))
                data_dict["domain_label"].append(domain)
                data_dict["domain_label_id"].append(domain_idx)
                data_dict["example_id"].append(f"{domain}_{i+1}")
        return data_dict

    def load_data(self) -> Dict[str, Dict[str, List[Union[str, List[str], List[int]]]]]:
        return {split: self.read_data_from_file(mode=split) for split in AbsaSeq2SeqDataProcessor.ALL_SPLITS}

    def get_invalid_prediction(self, label: str, label_type: str) -> int:
        labels_set = set(self.labels_dict[label_type].keys())
        labels_set.remove(label)
        return self.labels_dict[label_type][choice(tuple(labels_set))]

    def get_split_data(self, split: str) -> Dict[str, List[Union[str, List[str], List[int]]]]:
        assert split in AbsaSeq2SeqDataProcessor.ALL_SPLITS
        return self.data[split]

    def is_valid_label(self, label: str) -> bool:
        return bool(label in self.reduce_label_dict.values())

    def get_split_domain_data(self, split: str, domain: str) -> Dict[str, List[Union[str, List[str], List[int]]]]:
        assert split in AbsaSeq2SeqDataProcessor.ALL_SPLITS
        assert domain in self.src_domains + [self.trg_domain]
        l, r = 0, len(self.data[split]["domain_label"]) - 1
        while self.data[split]["domain_label"][l] != domain:
            l += 1
        while self.data[split]["domain_label"][r] != domain:
            r -= 1
        split_domain_data = defaultdict(list)
        for k, v in self.data[split].items():
            split_domain_data[k] = v[l:r+1]
        return split_domain_data


def test_AbsaSeq2SeqDataProcessor():
    data_processor = AbsaSeq2SeqDataProcessor(["device", "laptops", "rest"], "service", DATA_DIR)
    print()
    print(data_processor.data["dev"].keys())
    print(len(list(data_processor.data["dev"].values())[0]))
    for k, v in data_processor.data["dev"].items():
        if type(v) is not int:
            print(k, v[-80])
        else:
            print(k, v[-80], len(v[-80]))
    tokenizer = T5TokenizerFast.from_pretrained("t5-base")
    for split in AbsaSeq2SeqDataProcessor.ALL_SPLITS:
        print(split, len(data_processor.data[split]["example_id"]))
        print(split, max(data_processor.data[split]["input_tokens_len"]))
        tokenizer_lens = tokenizer(data_processor.data[split]["output_labels_tokens"], is_split_into_words=True,
                                   max_length=128, return_length=True)["length"]
        print(split, max(tokenizer_lens))
        print(split, percentile(tokenizer_lens, 99.5))


class AbsaSeq2SeqDataset(Dataset):
    def __init__(self, split: str, data_processor: AbsaSeq2SeqDataProcessor, tokenizer: T5TokenizerFast, max_seq_len: int):
        self.split = split
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tokenized_data = self._init_tokenized_data(split, data_processor, tokenizer, max_seq_len)

    def __len__(self):
        return len(self.tokenized_data["example_id"])

    def __getitem__(self, index):
        return {
            k: v[index]
            for k, v in self.tokenized_data.items()
        }

    @staticmethod
    def _init_tokenized_data(split, data_processor, tokenizer, max_seq_len):
        data = data_processor.get_split_data(split)
        tokenized_data = tokenizer(data["input_tokens"], is_split_into_words=True,
                                   padding=True, truncation=True,
                                   max_length=max_seq_len, return_tensors="pt", return_attention_mask=True)
        tokenized_data["output_labels_ids"] = tokenizer(data["output_labels_tokens"], is_split_into_words=True,
                                                        padding=True, truncation=True,
                                                        max_length=max_seq_len, return_tensors="pt")["input_ids"]
        tokenized_data["input_str"] = data["input_str"]
        tokenized_data["output_labels_str"] = data["output_labels_str"]
        tokenized_data["domain_label"] = data["domain_label"]
        tokenized_data["example_id"] = data["example_id"]
        return tokenized_data


def test_AbsaSeq2SeqDataset():
    data_processor = AbsaSeq2SeqDataProcessor(["device", "laptops", "rest"], "service", DATA_DIR)
    print()
    print(data_processor.data["dev"].keys())
    print(len(list(data_processor.data["dev"].values())[0]))
    for k, v in data_processor.data["dev"].items():
        if type(v) is not int:
            print(k, v[0])
        else:
            print(k, v[0], len(v[0]))
    dataset = AbsaSeq2SeqDataset("dev", data_processor, T5TokenizerFast.from_pretrained("t5-base"), 64)
    print(len(dataset))
    for example in dataset:
        for k, v in example.items():
            print(k, v)
        break
    dataloader = DataLoader(dataset, 4)
    for batch in dataloader:
        for k, v in batch.items():
            print(k, v)
        break


