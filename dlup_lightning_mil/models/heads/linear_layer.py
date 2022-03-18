# coding=utf-8
# Copyright (c) DLUP Contributors

from typing import Tuple, Union

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn


class LinearLayer(LightningModule):
    def __init__(self, in_features: Union[int, Tuple[int, int]], out_features: Union[int, Tuple[int, int]], lr: float):
        self.lr = lr
        super().__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        x = self.layer(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
