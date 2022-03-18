# coding=utf-8
# Copyright (c) DLM Contributors

import logging
import pathlib
import time
import pandas as pd
from pathlib import Path

import PIL.Image

from dlup_lightning_mil.data.transforms import RotationAndFlipTransform
from dlup_lightning_mil.utils import txt_of_paths_to_list
import torchvision

try:
    from dlup.background import AvailableMaskFunctions, get_mask, load_mask
    from dlup.data.dataset import ConcatDataset, SlideImage, TiledROIsSlideImageDataset
    from dlup.tiling import TilingMode
    from dlup import DlupUnsupportedSlideError
except ImportError:
    raise ImportError("Make sure that DLUP is installed with 'vissl/third_party/dlup$ python setup.py develop'")

from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import Dataset, DataLoader

class MaskGetter:
    """
    Takes the VISSL config and sets all parameters required for getting a mask for the dataset
    """
    def __init__(self, mask_factory: str, mask_root_dir: str):
        self.mask_options = {
            'load_from_disk': self.load_from_disk,
            'no_mask': self.no_mask
        }
        self.mask_factory = mask_factory

        if self.mask_factory == 'load_from_disk':
            self.mask_root_dir = mask_root_dir
            assert pathlib.Path(self.mask_root_dir).is_dir()
        else:
            self.mask_root_dir = None

        self.current_slide_image = None
        self.current_idx = None

    def return_mask_from_config(self, slide_image, idx, relative_wsi_path):
        """
        Returns a mask with the given mask_factory
        """
        self.current_idx = idx
        mask = self.mask_options[self.mask_factory](slide_image=slide_image, relative_wsi_path=relative_wsi_path)
        return mask

    def load_from_disk(self, *args, **kwargs):
        """
        Loads mask from disk. Reads a .png saved by DLUP and converts into a npy object
        """
        mask = load_mask(mask_file_path=pathlib.Path(self.mask_root_dir) / pathlib.Path(kwargs['relative_wsi_path']).parent / pathlib.Path(kwargs['relative_wsi_path']).stem / pathlib.Path('mask.png'))
        return mask

    def no_mask(self, *args, **kwargs):
        """
        Returns no mask
        """
        return None


class TransformDLUP2HISSL:
    """
    A small class to transform the objects returned by a DLUP dataset to the expected object by VISSL.
    Essentially, it ensures the image object is of type PIL.Image.Image, and ensures that the paths are strings.
    This is used in (the nki-ai fork of) vissl, in data.dlup_dataset.
    There, it only returns the image and an is_success flag.
    """

    def __init__(self, transform):
        self.transform=transform
        pass

    def __call__(self, sample):
        # torch and VISSL collate functions can not handle a pathlib.path object, \
        # and want a string instead
        sample["path"] = str(sample["path"])
        # Openslide returns RGBA, but most neural networks want RGB
        sample["image"] = self.transform(sample["image"].convert("RGB"))
        return sample


class DLUPSlideImageDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 wsi_paths_and_targets: str,
                 mpp: float,
                 tile_size_x: int,
                 tile_size_y: int,
                 tile_overlap_x: int,
                 tile_overlap_y: int,
                 tile_mode: str,
                 crop: bool,
                 mask_factory: str,
                 mask_foreground_threshold: float,
                 mask_root_dir: str,
                 transform: torchvision.transforms.Compose):
        # --------------------------
        # Set main variables used
        # --------------------------
        super().__init__()

        self.root_dir = Path(root_dir)

        path = pathlib.Path(wsi_paths_and_targets)
        self.df = pd.read_csv(path, header=None)

        self.relative_wsi_paths = self.df[0]

        self.df = self.df.set_index(0)

        tile_mode = TilingMode[tile_mode]

        # --------------------------
        # Set transform
        # --------------------------
        # only transforms a Pathlib object to a string to work with the standard pytorch collate. VISSL has a
        # very involved collate functions, see vissl.data.collators, which we would rather not touch to manage
        # pathlib objects. Can easily add it though: https://vissl.readthedocs.io/en/v0.1.5/extend_modules/data_collators.html
        self.transform = TransformDLUP2HISSL(transform)

        # --------------------------
        # Init the class that takes care of the mask options
        # --------------------------
        if mask_factory != "no_mask":
            self.foreground_threshold = mask_foreground_threshold
        else:
            self.foreground_threshold = 0.1  # DLUP dataset erroneously requires a float instead of optional None

        self.mask_getter = MaskGetter(
            mask_factory=mask_factory,
            mask_root_dir=mask_root_dir
        )

        # --------------------------
        # Build dataset
        # --------------------------
        single_wsi_datasets: list = []
        logging.info(f"Building dataset...")
        for idx, relative_wsi_path in enumerate(self.relative_wsi_paths):
            absolute_wsi_path = self.root_dir / Path(relative_wsi_path)
            try:
                slide_image = SlideImage.from_file_path(absolute_wsi_path)
            except DlupUnsupportedSlideError:
                logging.warning(f"{absolute_wsi_path} is unsupported. Skipping WSI.")
                continue
            mask = self.mask_getter.return_mask_from_config(slide_image=slide_image,
                                                            idx=idx,
                                                            relative_wsi_path=relative_wsi_path
                                                            )
            single_wsi_datasets.append(
                TiledROIsSlideImageDataset.from_standard_tiling(
                    path=absolute_wsi_path,
                    mpp=mpp,
                    tile_size=(tile_size_x, tile_size_y),
                    tile_overlap=(tile_overlap_x, tile_overlap_y),
                    tile_mode=tile_mode,
                    crop=crop,
                    mask=mask,
                    mask_threshold=self.foreground_threshold,
                    transform=self.transform,
                )
            )
        self.dlup_dataset = ConcatDataset(single_wsi_datasets)
        logging.info(f"Built dataset successfully")

    def num_samples(self) -> int:
        """
        Size of the dataset
        """
        return len(self.dlup_dataset)  # Use the implementation from the DLUP class

    def __len__(self) -> int:
        """
        Size of the dataset
        """
        return self.num_samples()

    def __getitem__(self, index) -> (PIL.Image.Image, bool):
        sample = self.dlup_dataset.__getitem__(index)
        relative_path = sample["path"].replace(str(self.root_dir) + '/', '')
        (patient_id, slide_id, target) = self.df.loc[relative_path, [1, 2, 3]]
        return_object = {
            'x': sample["image"],
            'y': int(target),
            'slide_id': slide_id,
            'patient_id': patient_id,
            'paths': str(relative_path),
            'root_dir': str(self.root_dir),
            'meta': {
                'tile_x': sample["coordinates"][0],
                'tile_y': sample["coordinates"][1],
                'tile_mpp': sample["mpp"],
                'tile_w': sample["region_size"][0],
                'tile_h': sample["region_size"][1],
                'tile_region_index': sample['region_index']
            }

        }
        return return_object


class DLUPSlideImageModule(LightningDataModule):
    def __init__(self,
                 root_dir: str,
                 train_wsi_paths_and_targets: str,
                 val_wsi_paths_and_targets: str,
                 test_wsi_paths_and_targets: str,
                 mpp: float,
                 tile_size_x: int,
                 tile_size_y: int,
                 tile_overlap_x: int,
                 tile_overlap_y: int,
                 tile_mode: str,
                 crop: bool,
                 mask_factory: str,
                 mask_foreground_threshold: float,
                 mask_root_dir: str,
                 num_workers: int,
                 batch_size: int,
                 transform: str = "rotate_flip"
                 ):
        super().__init__()
        self.num_workers = num_workers
        self.root_dir = Path(root_dir)
        self.train_path = Path(train_wsi_paths_and_targets)
        self.val_path = Path(val_wsi_paths_and_targets)
        self.test_path = Path(test_wsi_paths_and_targets)
        self.mpp = mpp
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y
        self.tile_overlap_x = tile_overlap_x
        self.tile_overlap_y = tile_overlap_y
        self.crop = crop
        self.tile_mode = tile_mode
        self.mask_factory = mask_factory
        self.mask_foreground_threshold = mask_foreground_threshold
        self.mask_root_dir = mask_root_dir
        self.batch_size = batch_size

        assert tile_size_x == tile_size_y

        if transform == "rotate_flip":
            self.transform = RotationAndFlipTransform(tile_size_x)
        else:
            raise ValueError

    def prepare_data(self):
        self.train_dataset, self.val_dataset, self.test_dataset = [DLUPSlideImageDataset(
                                root_dir=self.root_dir,
                                wsi_paths_and_targets=paths,
                                mpp=self.mpp,
                                tile_size_x=self.tile_size_x,
                                tile_size_y=self.tile_size_y,
                                tile_overlap_x=self.tile_overlap_x,
                                tile_overlap_y=self.tile_overlap_y,
                                tile_mode=self.tile_mode,
                                crop=self.crop,
                                mask_factory=self.mask_factory,
                                mask_foreground_threshold=self.mask_foreground_threshold,
                                mask_root_dir=self.mask_root_dir,
                                transform=self.transform)
            for paths in [self.train_path, self.val_path, self.test_path]]

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
