from torch.utils.data import Dataset
from datasets import arrow_dataset

import itertools


class LanguageMultiTaskDataset(Dataset):
    def __init__(
        self,
        task_type: str,
        task_name: str,
        data: arrow_dataset.Dataset,
        feature_keys: list,
        label_key: str,
    ):
        assert task_type in [
            "SequenceClassification",
            "TokenClassification",
            "Regression",
            "NextSentencePrediction",
            "CausalLM",
            "Seq2SeqLM",
        ]
        pass

        self.task_type = task_type
        self.task_name = task_name
        self.data = data
        self.feature_keys = feature_keys
        self.label_key = label_key

        if self.task_type != "Regression":
            if type(self.data[label_key][0]) == list:
                self.label_set = set(list(itertools.chain(*self.data[label_key])))
            else:
                self.label_set = set(self.data[label_key])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        return [item[feature] for feature in self.feature_keys] + item[self.label_key]
