# coding=utf-8
# Copyright (c) DLUP Contributors


from pytorch_lightning import Trainer


class DLMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_validation_output_to_disk = False

