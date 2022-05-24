# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Category Embedding Model"""
import logging
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from pytorch_tabular.utils import _initialize_layers, _linear_dropout_bn


from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class CategoryEmbeddingBackbone(pl.LightningModule):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__()
        self.save_hyperparameters(config)
        self._build_network()

    def _build_network(self):
        # Linear Layers
        layers = []
        _curr_units = self.hparams.embedded_cat_dim + self.hparams.continuous_dim
        if self.hparams.embedding_dropout != 0 and self.hparams.embedded_cat_dim != 0:
            layers.append(nn.Dropout(self.hparams.embedding_dropout))
        for units in self.hparams.layers.split("-"):
            layers.extend(
                _linear_dropout_bn(
                    self.hparams.activation,
                    self.hparams.initialization,
                    self.hparams.use_batch_norm,
                    _curr_units,
                    int(units),
                    self.hparams.dropout,
                )
            )
            _curr_units = int(units)
        self.linear_layers = nn.Sequential(*layers)
        _initialize_layers(
            self.hparams.activation, self.hparams.initialization, self.linear_layers
        )
        self.output_dim = _curr_units
        # Embedding layers
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in self.hparams.embedding_dims]
        )
        if self.hparams.embedding_dropout > 0:
            self.embd_dropout = nn.Dropout(self.hparams.embedding_dropout)
        # Continuous Layers
        if self.hparams.batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(self.hparams.continuous_dim)

    def unpack_input(self, x: Dict):
        continuous_data, categorical_data = x["continuous"], x["categorical"]
        if self.hparams.embedded_cat_dim != 0:
            x = [
                embedding_layer(categorical_data[:, i])
                for i, embedding_layer in enumerate(self.embedding_layers)
            ]
            x = torch.cat(x, 1)
            if self.hparams.embedding_dropout > 0:
                x = self.embd_dropout(x)

        if self.hparams.continuous_dim != 0:
            if self.hparams.batch_norm_continuous_input:
                continuous_data = self.normalizing_batch_norm(continuous_data)

            if self.hparams.embedded_cat_dim != 0:
                x = torch.cat([x, continuous_data], 1)
            else:
                x = continuous_data
        return x

    def forward(self, x):
        x = self.unpack_input(x)
        x = self.linear_layers(x)
        return x


class CategoryEmbeddingModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _build_network(self):
        # Backbone
        self.backbone = CategoryEmbeddingBackbone(self.hparams)
        self.head = self._get_head_from_config()

    def extract_embedding(self):
        return self.backbone.embedding_layers
