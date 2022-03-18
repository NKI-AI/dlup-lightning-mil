# coding=utf-8
# Copyright (c) DLUP Contributors

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI


class DLMCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_class_arguments(ModelCheckpoint, "model_checkpointing")

    def before_instantiate_classes(self) -> None:
        self.config["trainer"]["callbacks"] = [ModelCheckpoint(**self.config["model_checkpointing"])]


