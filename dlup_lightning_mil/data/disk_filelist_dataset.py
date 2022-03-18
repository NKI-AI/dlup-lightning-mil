# coding=utf-8
# Copyright (c) DLUP Contributors

# TODO Implement (DLUP) pre-tiled dataset to get all tiles from a WSI to throw them into a (frozen) extractor and classifier
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import PIL

from dlup_lightning_mil.utils import txt_of_paths_to_list, txt_of_ints_to_list, txt_of_ids_to_list
from .transforms import RotationAndFlipTransform


class DiskFilelist(Dataset):

    def __init__(self, root_dir: Path, input_path: Path, label_path: Path, id_path: Path, transforms):
        super().__init__()
        self.paths = txt_of_paths_to_list(input_path)
        self.labels = txt_of_ints_to_list(label_path)
        self.patient_ids, self.slide_ids = txt_of_ids_to_list(id_path)
        self.transforms = transforms
        self.root_dir = root_dir

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data_obj = {"x": [], "y": [], "patient_id": [], "slide_id": [], "paths": [], "root_dir": []}
        relative_path = self.paths[idx]
        absolute_path = self.root_dir / relative_path
        x = PIL.Image.open(absolute_path).convert("RGB")
        y = self.labels[idx]

        if self.transforms:
            x = self.transforms(x)

        data_obj["paths"] = str(relative_path)
        data_obj["x"] = x
        data_obj["y"] = y
        data_obj["root_dir"] = str(self.root_dir)
        data_obj["patient_id"] = self.patient_ids[idx]
        data_obj["slide_id"] = self.slide_ids[idx]

        return data_obj


class DiskFilelistModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir: str,
                 train_path: str,
                 val_path: str,
                 test_path: str,
                 train_labels: str,
                 val_labels: str,
                 test_labels: str,
                 train_ids: str,
                 val_ids: str,
                 test_ids: str,
                 num_workers: int,
                 batch_size: int,
                 transform: str="rotate_flip",
                 im_size: int=224):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.root_dir = Path(root_dir)
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)
        self.train_labels = Path(train_labels)
        self.test_labels = Path(test_labels)
        self.val_labels = Path(val_labels)
        self.train_ids = Path(train_ids)
        self.val_ids = Path(val_ids)
        self.test_ids = Path(test_ids)

        if transform == "rotate_flip":
            self.transform = RotationAndFlipTransform(im_size)
        else:
            raise ValueError

    def prepare_data(self):
        self.train_dataset = DiskFilelist(root_dir=self.root_dir, input_path=self.train_path, label_path=self.train_labels, id_path=self.train_ids, transforms=self.transform)
        self.val_dataset = DiskFilelist(root_dir=self.root_dir, input_path=self.val_path, label_path=self.val_labels, id_path=self.val_ids, transforms=self.transform)
        self.test_dataset = DiskFilelist(root_dir=self.root_dir, input_path=self.test_path, label_path=self.test_labels, id_path=self.test_ids, transforms=self.transform)

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size)

    def teardown(self, stage):
        pass
        # clean up after fit or test
        # called on every process in DDP
