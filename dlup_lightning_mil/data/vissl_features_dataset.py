# coding=utf-8
# Copyright (c) DLUP Contributors

from pathlib import Path

import h5py
import numpy as np
import torch

# TODO Implement dataset to read the single .h5 object that VISSL exports when saving extracted features
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from dlup_lightning_mil.utils import txt_of_paths_to_list


class CompiledH5Dataset(Dataset):
    """
    This class only works for the exact h5 dataset for TCGA-CRCk MSI/MSS as compiled by compile_h5_features from this repo
    """

    def __init__(self, input_path: Path, dataset: str, root_dir: str):
        self.paths = txt_of_paths_to_list(input_path)
        self.dataset = dataset
        self.root_dir = root_dir

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.dataset == "tcga-crck":
            data_obj = {"x": [], "y": [], "case_id": [], "slide_id": [], "paths": [], "root_dir": []}
            hf = h5py.File(Path(self.root_dir) / Path(self.paths[idx]), "r")

            data_obj["paths"] = [list(hf["paths"].asstr()[()])]
            data_obj["x"] = hf["data"][()]
            data_obj["case_id"] = [hf["case_id"].asstr()[()]]
            data_obj["slide_id"] = [hf["slide_id"].asstr()[()]]
            data_obj["y"] = [hf["target"][()]]
            data_obj["root_dir"] = [hf["root_dir"].asstr()[()]]
            data_obj["features_path"] = [str(self.paths[idx])]

            data_obj["x"] = torch.Tensor(np.array(data_obj["x"]))
            data_obj["y"] = torch.Tensor(np.array(data_obj["y"]))

            hf.close()

        elif self.dataset == "tcga-bc":
            data_obj = {"x": [], "y": [], "case_id": [], "slide_id": [], "path": [], "root_dir": [],
                        'tile_x': [], 'tile_y': [], 'tile_h': [], 'tile_w': [], 'tile_mpp': [],
                        'tile_region_index': [], 'meta': {}}
                        # 'tile_vissl_index': []}

            svs_path, case_id, slide_id, target = str(self.paths[idx]).split(',')

            hf = h5py.File(f'{self.root_dir}/{svs_path}.h5', "r")

            data_obj['y'] = [int(float(target))]

            data_obj["paths"] = [hf["path"].asstr()[()]]
            data_obj["x"] = hf["data"][()]
            data_obj["case_id"] = [case_id]
            data_obj["slide_id"] = [slide_id]
            data_obj["root_dir"] = [hf["root_dir"].asstr()[()]]

            data_obj['meta']['tile_x'] = hf['x'][()]
            data_obj['meta']['tile_y'] = hf['y'][()]
            data_obj['meta']['tile_h'] = hf['h'][()]
            data_obj['meta']['tile_w'] = hf['w'][()]
            data_obj['meta']['tile_region_index'] = hf['region_index'][()]
            # data_obj['tile_vissl_index'] = hf['vissl_index'][()]
            data_obj['meta']['tile_mpp'] = hf['mpp'][()]

            data_obj["features_path"] = [svs_path]

            data_obj["x"] = torch.Tensor(np.array(data_obj["x"]))
            data_obj["y"] = torch.Tensor(np.array(data_obj["y"]))

            hf.close()
        else:
            raise ValueError

        return data_obj


class CompiledH5DataModule(LightningDataModule):
    def __init__(self, train_path: str, val_path: str, test_path: str, root_dir: str, dataset: str, num_workers: int):
        super().__init__()
        self.num_workers = num_workers
        self.root_dir = Path(root_dir)
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)
        self.dataset = dataset

    def prepare_data(self):
        self.train_dataset = CompiledH5Dataset(input_path=self.train_path, dataset=self.dataset, root_dir=self.root_dir)
        self.val_dataset = CompiledH5Dataset(input_path=self.val_path, dataset=self.dataset, root_dir=self.root_dir)
        self.test_dataset = CompiledH5Dataset(input_path=self.test_path, dataset=self.dataset, root_dir=self.root_dir)

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers)

    def teardown(self, stage):
        pass
        # clean up after fit or test
        # called on every process in DDP
