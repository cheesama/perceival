from torch.utils.data import Dataset
from datasets import arrow_dataset
from transformers import PerceiverTokenizer

import itertools
import logging
import torch

class LanguageMultiTaskDataset(Dataset):
    def __init__(
        self,
        task_type: str,
        task_name: str,
        data: arrow_dataset.Dataset,
        feature_keys: list,
        label_key,
        config: str = "deepmind/language-perceiver",
    ):
        assert task_type in [
            "SequenceClassification",
            "TokenClassification",
            "Regression",
            "NextSentencePrediction",
            "CausalLM",
            "Seq2SeqLM",
            "QuestionAnswering",
        ]

        self.task_type = task_type
        self.task_name = task_name
        self.data = data
        self.feature_keys = feature_keys
        self.label_key = label_key
        self.tokenizer = PerceiverTokenizer.from_pretrained(config)

        if self.task_type != "Regression":
            if type(self.data[label_key][0]) == list:
                self.label_set = set(list(itertools.chain(*self.data[label_key])))
            else:
                self.label_set = set(self.data[label_key])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print(self.task_name)
        item = self.data[idx]
        features = self.tokenizer(
            self.tokenizer.sep_token.join(
                item[feature] for feature in self.feature_keys
            ),
            padding="max_length",
            return_tensors="pt",
        )
        label = item[self.label_key]

        return features, torch.tensor([label])
