# coding=utf-8
# Copyright (c) DLUP Contributors
import h5py
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import torchmetrics
from torch import nn
from pathlib import Path
import json


class Attention(LightningModule):
    #TODO implement slide-level or patient-level AUC?
    def __init__(self, lr: float, hidden_features: int, in_features: int, num_classes: int, weight_decay: float):
        super(Attention, self).__init__()

        # DeepMIL specific initialization
        self.num_classes = num_classes
        self.L = in_features
        self.D = hidden_features
        self.K = 1
        self.lr = lr
        self.weight_decay = weight_decay
        self.attention = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K))
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.num_classes),
        )

        # Initialize validation output
        self.validation_output = self._reset_output()
        self.test_output = self._reset_output()


        # Initialize metrics
        self.auroc = torchmetrics.AUROC()
        self.f1 = torchmetrics.F1()
        self.pr_curve = torchmetrics.PrecisionRecallCurve()

        self.save_hyperparameters()

    def _reset_output(self):
        return {'loss': [], 'target': [], 'prediction': []}

    def forward(self, x):
        # Since we have batch_size = 1, squeezes from (1,num,features) to (num, features)
        H = x.squeeze(0)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_hat = self.classifier(M)
        return Y_hat, A

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y = y.squeeze(0).long()
        y_hat, _ = self(x)
        train_loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y = y.squeeze(0).long()
        y_hat, A = self(x)
        val_loss = F.cross_entropy(y_hat, y)

        self.validation_output["target"].append(y.cpu())
        self.validation_output["prediction"].append(torch.nn.functional.softmax(y_hat.cpu(), dim=1)[:, 1])
        self.validation_output["loss"].append(val_loss.cpu())

        if self.trainer.save_validation_output_to_disk:
            self.save_output(batch, A, y_hat, fold='val')
        else:
            self.log("val_loss", val_loss)

        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y = y.squeeze(0).long()
        y_hat, A = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.test_output["target"].append(y.cpu())
        self.test_output["prediction"].append(torch.nn.functional.softmax(y_hat.cpu(), dim=1)[:, 1])
        self.test_output["loss"].append(loss.cpu())

        self.save_output(batch, A, y_hat, fold='test')
        return loss

    def log_metrics(self, prefix, output):
        target = torch.ShortTensor(output["target"])
        prediction = torch.Tensor(output["prediction"])

        auroc_score = self.auroc(preds=prediction, target=target)
        f1_score = self.f1(preds=prediction, target=target)

        #TODO Save these or do this afterwards from the patient-level outputs?
        precision, recall, thresholds = self.pr_curve(preds=prediction, target=target)

        #TODO save the scores and cut-offs... otherwise we can't do proper statistical testing.
        self.log(f"{prefix}_auc", auroc_score, prog_bar=True, logger=True)
        self.log(f"{prefix}_f1", f1_score, prog_bar=True, logger=True)

        if self.trainer.save_validation_output_to_disk:
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

    def validation_epoch_end(self, validation_step_outputs) -> None:
        self.log_metrics(prefix='val', output=self.validation_output)
        self.validation_output = self._reset_output()

    def test_epoch_end(self, test_step_outputs) -> None:
        self.log_metrics(prefix='test', output=self.test_output)
        self.test_output = self._reset_output()

    def save_output(self, batch, As, y_hats, fold):
        #TODO make this work for both the BC and CRC dataset.
        # They  have some different information since
        # 1) crc only needs the paths
        # 2) bc needs metadata about the tiles

        if not (Path(self.trainer.log_dir)/f'output/{fold}').is_dir():
            Path.mkdir(Path(self.trainer.log_dir)/f'output/{fold}', parents=True)

        batch_size = len(batch['y']) # could be any other

        for i in range(batch_size):
            y, case_id, slide_id, paths, root_dir, features_path = batch['y'][i], batch['case_id'][i], batch['slide_id'][i], batch['paths'][i], \
                                                                     batch['root_dir'][i], batch['features_path'][i]

            # This data is shared among BC and CRC dataset
            hf = h5py.File(f'{self.trainer.log_dir}/output/{fold}/{slide_id[0]}_output.h5', "a")
            hf['slide_id'] = slide_id[0]
            hf['patient_id'] = case_id[0]
            hf['attention'] = As[i].cpu()
            # CRCk gives a list of paths for all the tiles: [(path,), (path,)...], but BC only gives (path,) for the WSI
            hf['paths'] = [path[0] for path in paths] if len(paths) > 1 else paths[0]# TODO this is not OK
            hf['target'] = y.cpu()
            hf['prediction'] = torch.nn.functional.softmax(y_hats.cpu(), dim=1)[:,1].cpu().tolist()
            hf['features_path'] = features_path[0]
            hf['root_dir'] = batch['root_dir'][i][0]

            # If the data is saved using a DLUP SLide Image dataset, we need meta information about the tiles
            if 'meta' in batch.keys():
                hf['meta/tile_x'] = batch['meta']['tile_x'][0].cpu()
                hf['meta/tile_y'] = batch['meta']['tile_y'][0].cpu()
                hf['meta/tile_w'] = batch['meta']['tile_w'][0].cpu()
                hf['meta/tile_h'] = batch['meta']['tile_h'][0].cpu()
                hf['meta/tile_mpp'] = batch['meta']['tile_mpp'][0].cpu()
                hf['meta/tile_region_index'] = batch['meta']['tile_region_index'][0].cpu()

            hf.close()


class VarAttention(Attention):
    def __init__(self, *args, **kwargs):
        super(VarAttention, self).__init__(*args, **kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.L * self.K, self.num_classes) # 2x since we also have variance
        )

    def compute_weighted_std(self, A, H, M):
        #TODO Now implemented to work with output as given above which is only for batch size of 1
        A, H = A.unsqueeze(2), H.unsqueeze(0)
        # Following https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
        # A: Attention (weight):    batch x instances x 1
        # H: Hidden:                batch x instances x channels
        H = H.permute(0, 2, 1)  # batch x channels x instances

        # M: Weighted average:      batch x channels
        M = M.unsqueeze(dim=2)  # batch x channels x 1
        # ---> S: weighted stdev:   batch x channels

        # N is non-zero weights for each bag: batch x 1

        N = (A != 0).sum(dim=1)

        upper = torch.einsum('abc, adb -> ad', A, (H - M) ** 2)  # batch x channels
        lower = ((N - 1) * torch.sum(A, dim=1)) / N  # batch x 1

        # Square root leads to infinite gradients when input is 0
        # Solution: No square root, or add eps=1e-8 to the input
        # But adding the eps will still lead to a large gradient from the sqrt through these 0-values.
        # Whether we look at stdev or variance shouldn't matter much, so we choose to go for the variance.
        S = (upper / lower)

        return S

    def forward(self, x):
        # Since we have batch_size = 1, squeezes from (1,num,features) to (num, features)
        H = x.squeeze(0)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        S = self.compute_weighted_std(A, H, M)

        MS = torch.cat((M, S),
                       dim=1)  # concatenate the two tensors among the feature dimension, giving a twice as big feature

        Y_hat = self.classifier(MS)
        return Y_hat, A


class GatedAttention(Attention):
    def __init__(self, *args, **kwargs):
        super(GatedAttention, self).__init__(*args, **kwargs)
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, self.num_classes),
        )
    def forward(self, x):
        H = x.squeeze(0)
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_hat = self.classifier(M)

        return Y_hat, A
