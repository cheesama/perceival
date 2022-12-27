from transformers import PerceiverConfig, PerceiverTokenizer

import torch.nn as nn
import pytorch_lightning as pl


class LanguageTaskDecoder:
    def __init__(
        self,
        task_name: str,
        decoder_dim: int,
        labels: dict,
        config: str = "deepmind/language-perceiver",
    ):
        super().__init__()
        self.task_name = task_name
        self.config = PerceiverConfig.from_pretrained(config)
        self.labels = labels
        self.linear = nn.Linear(self.config.vocab_size, len(self.labels))

    def forward(self, batch, batch_idx):
        pass
