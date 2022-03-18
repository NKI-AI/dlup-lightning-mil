# coding=utf-8
# Copyright (c) DLUP Contributors

import pytorch_lightning as pl
from dlup_lightning_mil.trainer import DLMTrainer
from dlup_lightning_mil.cli import DLMCLI


def main():
    cli = DLMCLI(model_class=pl.LightningModule,
                 subclass_mode_model=True,
                 trainer_class=DLMTrainer,
                 subclass_mode_data=True,
                 datamodule_class=pl.LightningDataModule,
                 run=False,
                 save_config_overwrite=True)

    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
    cli.trainer.save_validation_output_to_disk = True
    cli.trainer.validate(model=cli.model, dataloaders=cli.trainer.val_dataloaders, ckpt_path='best')
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path='best')


if __name__ == "__main__":
    main()
