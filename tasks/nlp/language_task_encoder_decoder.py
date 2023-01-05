from torch.utils.data import Dataset, DataLoader
from torch import optim
from transformers import PerceiverConfig, PerceiverTokenizer, PerceiverForMaskedLM
from pytorch_lightning.trainer.supporters import CombinedLoader

import torch.nn as nn
import pytorch_lightning as pl


class LanguageMultiTaskEncoderDecoder(pl.LightningModule):
    def __init__(
        self,
        train_datasets: [Dataset],
        valid_datasets: [Dataset],
        config: str = "deepmind/language-perceiver",
        baseModelPath: str = None,
        lr: float = 1e-5,
        batch_size: int = 16,
    ):
        super().__init__()

        self.train_datasets = train_datasets
        self.valid_datasets = valid_datasets

        if baseModelPath is None:
            self.base_model = PerceiverForMaskedLM.from_pretrained(config)
        else:
            self.base_model = PerceiverForMaskedLM.from_pretrained(baseModelPath)

        self.tokenzier = PerceiverTokenizer.from_pretrained(config)
        self.config = PerceiverConfig.from_pretrained(config)
        self.lr = lr
        self.batch_size = batch_size

        # add multiple predictions layers
        self.feature_layers = nn.ModuleDict(
            {
                f"{each_dataset.task_type}-{each_dataset.task_name}": nn.Linear(
                    self.config.d_model,
                    len(each_dataset.label_set)
                    if hasattr(each_dataset, "label_set")
                    else 1,
                )
                for each_dataset in self.train_datasets
            }
        )

    def forward(self, text):
        # prepare input
        encoding = tokenizer(text, padding="max_length", return_tensors="pt")
        base_features = self.base_model(
            inputs=encoding.input_ids, attention_mask=encoding.attention_mask
        )

    def predict(self, text):
        pass

    def save(self):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        train_loaders = {
            f"{dataset.task_type}-{dataset.task_name}": DataLoader(
                dataset, batch_size=self.batch_size
            )
            for dataset in self.train_datasets
        }
        combined_loader = CombinedLoader(train_loaders, mode="max_size_cycle")
        return combined_loader

    def training_step(self, train_batch, batch_idx):
        print(train_batch)

    def val_dataloader(self):
        valid_loaders = {
            f"{dataset.task_type}-{dataset.task_name}": DataLoader(
                dataset, batch_size=self.batch_size
            )
            for dataset in self.valid_datasets
        }
        combined_loader = CombinedLoader(valid_loaders, mode="max_size_cycle")
        print (combined_loader)
        return combined_loader

    def validation_step(self, val_batch, batch_idx):
        print(val_batch)
