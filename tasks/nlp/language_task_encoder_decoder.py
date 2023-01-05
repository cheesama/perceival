from torch.utils.data import Dataset, DataLoader
from torch import optim
from transformers import PerceiverConfig, PerceiverTokenizer, PerceiverForMaskedLM
from transformers.models.perceiver.modeling_perceiver import PerceiverTextPreprocessor
from pytorch_lightning.trainer.supporters import CombinedLoader

import torch
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
        self.preprocessor = PerceiverTextPreprocessor(self.config)
        self.lr = lr
        self.batch_size = batch_size

        self.classification_loss_fn = torch.nn.CrossEntropyLoss(
            weight=None,
            size_average=None,
            ignore_index=-100,
            reduce=None,
            reduction="mean",
        )
        self.regression_loss_fn = torch.nn.MSELoss(
            size_average=None, reduce=None, reduction="mean"
        )

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

    def forward(self, encoding, task_name):
        # encoding = tokenizer(text, padding="max_length", return_tensors="pt")
        print (encoding)
        print (encoding.input_ids.size())

        base_features = self.base_model(
            encoding.input_ids.unsqueeze(-1), attention_mask=encoding.attention_mask
        )

        predictions = self.fature_layers[task](base_features)

        # pick just first [CLS] poistion for just SequenceClassification & Regression
        return predictions[:, 0, :]

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
        loss_dict = {}
        for k, v in train_batch.items():
            if "SequenceClassification" in k:
                loss_dict[k] = self.classification_loss_func(
                    self.forward(train_batch[k][0], k), train_batch[k][1]
                )
            elif "Regression" in k:
                loss_dict[k] = self.regression_loss_func(
                    self.forward(train_batch[k][0], k), train_batch[k][1]
                )
            else:
                raise Error(
                    "SequenceClassification & Regression task_type only available now"
                )

            self.log(f"train/{k}", loss_dict[k])

        total_loss = 0
        for k, v in loss_dict.items():
            total_loss += loss_dict[k]

        return loss

    def val_dataloader(self):
        valid_loaders = {
            f"{dataset.task_type}-{dataset.task_name}": DataLoader(
                dataset, batch_size=self.batch_size
            )
            for dataset in self.valid_datasets
        }
        combined_loader = CombinedLoader(valid_loaders, mode="max_size_cycle")
        return combined_loader

    def validation_step(self, val_batch, batch_idx):
        loss_dict = {}
        for k, v in val_batch.items():
            if "SequenceClassification" in k:
                loss_dict[k] = self.classification_loss_fn(
                    self.forward(val_batch[k][0], k), val_batch[k][1]
                )
            elif "Regression" in k:
                loss_dict[k] = self.regression_loss_fn(
                    self.forward(val_batch[k][0], k), val_batch[k][1]
                )
            else:
                raise Exception(
                    "SequenceClassification & Regression task_type only available now"
                )

            self.log(f"val/{k}", loss_dict[k])
