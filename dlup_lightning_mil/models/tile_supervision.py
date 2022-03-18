# coding=utf-8
# Copyright (c) DLUP Contributors
import json
from pathlib import Path
from typing import Tuple, Union

import h5py
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from dlup_lightning_mil.utils.model import get_backbone
from torch import nn
import numpy as np
import torchmetrics


class TileSupervision(LightningModule):
    """
    Standard supervision on labelled tiles. Works with disk_filelist_dataset and dlup_wsi_dataset

    Args
        backbone: str
            Description of backbone model. For now: "resnet18" or "shufflenet_v2_x1_0"
        load_weights: str
            Which weights to load. If none given, random initialization of model. Can be either "imagenet" or
            the absolute path to saved model weights by VISSL
        lr: float
            learning rate for ADAM optimizer
    """

    def __init__(self, backbone: str, load_weights: str, lr: float, weight_decay: float, num_classes: int, metric_level: str = 'slide'):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay

        backbone_dict = get_backbone(backbone=backbone, load_weights=load_weights)
        num_features = backbone_dict['num_features']
        self.feature_extractor = backbone_dict['model']
        self.classifier = nn.Linear(in_features=num_features, out_features=num_classes)
        self.save_hyperparameters()

        # Initialize validation output
        self.validation_output = self._reset_output()
        self.test_output = self._reset_output()

        # Initialize metrics
        self.metric_level = metric_level
        self.auroc = torchmetrics.AUROC()
        self.f1 = torchmetrics.F1()
        self.pr_curve = torchmetrics.PrecisionRecallCurve()

    def _reset_output(self):
        #TODO Add paths and/or metadata.... so that we can save this so that we can get back all the images
        # for the predictions that we made.
        return {'slide_id': [], 'patient_id': [], 'loss': [], 'target': [], 'prediction': [],
                'paths': [], 'meta': {}, 'root_dir': []}

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y.long())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y.long())
        self.track_values(x, y, y_hat, loss, batch, self.validation_output)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y.long())
        self.track_values(x, y, y_hat, loss, batch, self.test_output)
        return loss

    def validation_epoch_end(self, validation_step_outputs) -> None:
        self.log_metrics(prefix='val', output=self.validation_output)
        self.validation_output = self._reset_output()

    def test_epoch_end(self, test_step_outputs) -> None:
        self.log_metrics(prefix='test', output=self.test_output)
        self.test_output = self._reset_output()

    def track_values(self, x, y, y_hat, loss, batch, output_reference):
        output_reference["target"] += y.tolist()  # Bathc size values
        output_reference["prediction"] += torch.nn.functional.softmax(y_hat.cpu(), dim=1)[:,
                                                1].tolist()  # batch size values
        output_reference["loss"].append(loss.tolist())  # 1 value
        output_reference["patient_id"] += batch["patient_id"]  # Batch size values
        output_reference["slide_id"] += batch["slide_id"]  # batch size values
        output_reference["paths"] += batch["paths"]
        output_reference["root_dir"] += batch["root_dir"]

        # For the DLUPSlideImage dataset...
        if 'meta' in batch.keys():
            for key in batch["meta"].keys():
                if key not in output_reference["meta"].keys():
                    output_reference["meta"][key] = []
                output_reference["meta"][key] += batch["meta"][key].tolist()

    def log_metrics(self, prefix, output):
        target = torch.ShortTensor(output["target"])
        prediction = torch.Tensor(output["prediction"])

        # Do something to do slide-level or patient-level AUC
        # torch default collate returns strings as [[('string',)], [('string',)]]
        slide_ids = np.array(output["slide_id"])
        patient_ids = np.array(output["patient_id"])

        # Do it slide-level for now
        if self.metric_level == "patient":
            ids = patient_ids
        elif self.metric_level == "slide":
            ids = slide_ids
        else:
            raise ValueError

        unique_ids = np.unique(ids)
        id_level_targets = torch.ShortTensor([target[ids == i].max() for i in unique_ids])
        id_level_predictions = torch.Tensor([prediction[ids == i].mean() for i in unique_ids])

        auroc_score = self.auroc(preds=id_level_predictions, target=id_level_targets)
        f1_score = self.f1(preds=id_level_predictions, target=id_level_targets)

        precision, recall, thresholds = self.pr_curve(preds=id_level_predictions, target=id_level_targets)

        self.log(f"{prefix}_auc", auroc_score, prog_bar=True, logger=True)
        self.log(f"{prefix}_f1", f1_score, prog_bar=True, logger=True)

        if self.trainer.save_validation_output_to_disk:
            #---- Save metrics
            if not (Path(self.trainer.log_dir) / f'output/{prefix}').is_dir():
                Path.mkdir(Path(self.trainer.log_dir) / f'output/{prefix}', parents=True)

            metrics_to_save = {'auc': float(auroc_score),
                               'f1': float(f1_score),
                               'prcurve': {
                                   'precision': precision.tolist(),
                                   'recall': recall.tolist(),
                                   'thresholds': thresholds.tolist()}
                               }

            with open(Path(self.trainer.log_dir) / f'output/{prefix}/metrics.json', 'w') as f:
                f.write(json.dumps(metrics_to_save))

            #---- Save output
            self.save_output(prefix)

    def save_output(self, fold):
        if fold == 'val':
            output = self.validation_output
        elif fold == 'test':
            output = self.test_output
        else:
            raise NotImplementedError

        unique_slide_ids = set(output['slide_id'])

        for slide_id in unique_slide_ids:
            Path.mkdir(Path(self.trainer.log_dir) / f'output/{fold}', parents=True, exist_ok=True)
            hf = h5py.File(f'{self.trainer.log_dir}/output/{fold}/{slide_id}_output.h5', "a")

            current_slide_indices = np.array(output['slide_id']) == slide_id

            # It's a bit bloated, but np.array() allows for nice [True, False] indexing
            # But h5 doesn't like the np.array string encoding, so we make them back into lists

            slide_id = np.array(output['slide_id'])[current_slide_indices].tolist()
            if len(set(slide_id)) == 1:
                slide_id = list(set(slide_id)) # always the case, really
                hf['slide_id'] = slide_id
            else:
                raise ValueError
            patient_id = np.array(output['patient_id'])[current_slide_indices].tolist()
            if len(set(patient_id)) == 1:
                patient_id = list(set(patient_id))
                hf['patient_id'] = patient_id
            else:
                raise ValueError

            root_dir = np.array(output['root_dir'])[current_slide_indices].tolist()
            if len(set(root_dir)) == 1:
                root_dir = list(set(root_dir))
                hf['root_dir'] = root_dir
            else:
                raise ValueError

            hf['tile_prediction'] = np.array(output['prediction'])[current_slide_indices].tolist()
            paths = np.array(output['paths'])[current_slide_indices].tolist()
            if len(set(paths)) == 1:
                paths = list(set(paths)) # THis is the case for DLUP Slide Image, as it refers to the .svs. Not for disk filelist
            hf['paths'] = paths
            target = np.array(output['target'])[current_slide_indices].tolist()
            if len(set(target)) == 1:
                target = list(set(target))
                hf['target'] = target
            else:
                raise ValueError
            if 'meta' in output.keys():
                # Only DLUP SlideImage dataset gives keys and values in 'meta'
                for key in output['meta'].keys():
                    hf[f'meta/{key}'] = np.array(output['meta'][key])[current_slide_indices].tolist()

            hf.close()
