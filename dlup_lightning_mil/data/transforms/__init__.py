# coding=utf-8
# Copyright (c) DLUP Contributors

import torchvision
import torch


class MyRotationTransform:
    """Rotate by one of the given angles.

    Arguments:
        angles: list(ints). List of integer degrees to pick from. E.g. [0, 90, 180, 270] for a random 90-degree-like rotation
        """

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = self.angles[torch.randperm(len(self.angles))[0]]
        return torchvision.transforms.functional.rotate(x, angle)


class RotationAndFlipTransform:
    def __init__(self, size):
        """
        Collection of transforms for rotation invariance
        """

        self.rotation_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                MyRotationTransform([0, 90, 180, 270]),
                torchvision.transforms.ToTensor()
            ]
        )

    def __call__(self, x):
        return self.rotation_transform(x)

