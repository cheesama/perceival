from transformers import PerceiverConfig, PerceiverTokenizer, PerceiverForMaskedLM
from pytorch_lightning.trainer.supporters import CombinedLoader

import torch.nn as nn
import pytorch_lightning as pl

class LanguageMultiTaskEncoderDecoder(pl.LightningModule):
    def __init__(
        self,
        task_configs: [dict],
        decoder_dim: int,
        config: str = "deepmind/language-perceiver",
        baseModelPath: str = None,
        lr: float = 1e-3,
        batch_size: int = 16
    ):
        super().__init__()
        if baseModelPath is None:
            self.base_model = PerceiverForMaskedLM.from_pretrained(config)
        else:
            self.base_model = PerceiverForMaskedLM.from_pretrained(baseModelPath)

        self.task_configs = task_configs
        self.tokenzier = PerceiverTokenizer.from_pretrained(config)
        self.config = PerceiverConfig.from_pretrained(config)
        self.lr = lr
        self.batch_size = batch_size

        # add multiple classifier layers
        for task_config in self.task_configs:
            pass

    def forward(self, text):
        # prepare input
        encoding = tokenizer(text, padding="max_length", return_tensors="pt")
        base_features = self.base_model(inputs=encoding.input_ids, attention_mask=encoding.attention_mask)

    def predict(self, text):
        pass

    def save(self):
        pass

    def configure_optimizers(self):
        # return multiple optimizers
        pass

    def train_dataloader(self):
        pass

    def training_step(self, train_batch, batch_idx):
        pass

    def val_dataloader(self):
        pass

    def validation_step(self, val_batch, batch_idx):
        pass
           


