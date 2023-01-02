from torch.utils.data import Dataset
from datasets import arrow_dataset

class LanguageMultiTaskDataset(Dataset):
    def __init__(self, task_type: str, data: arrow_dataste.Dataset, feature_keys: list, label_key: str):
        assert task_type in [
            "SequenceClassification",
            "TokenClassification",
            "Regression",
            "NextSentencePrediction",
            "CausalLM",
            "Seq2SeqLM",
        ]
        pass

        self.data = data
        self.feature_keys = feature_keys
        self.label_set = set(self.data[label_key])
        self.label_key = label_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        return [ item[feature] for feature in self.feature_keys ] + item[self.label_key]

