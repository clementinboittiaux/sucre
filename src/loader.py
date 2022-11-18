# Copyright (C) 2022 Ifremer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Author: Clementin Boittiaux <boittiauxclementin at gmail dot com>

import cv2
import torch
import numpy as np
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from sfm import SfMImage


class ImageDataset(Dataset):
    def __init__(self, image_list: list[SfMImage], return_image: bool = True, return_depth: bool = True):
        self.images = image_list
        self.return_image = return_image
        self.return_depth = return_depth

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.return_image and not self.return_depth:
            return image.name, load_image(image.image_path)
        elif self.return_depth and not self.return_image:
            return image.name, load_depth(image.depth_path)
        elif self.return_image and self.return_depth:
            return image.name, load_image(image.image_path), load_depth(image.depth_path)


def load_image(image_path: Path) -> Tensor:
    return torch.tensor(cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)) / 255


def load_depth(depth_path: Path) -> Tensor:
    match depth_path.suffix.lower():
        case '.png':
            return torch.tensor(np.int32(cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED))) / 1000
        case '.tif' | '.tiff':
            return torch.tensor(cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED))
        case _:
            raise ValueError(f'Incorrect depth map extension for {depth_path.name}. '
                             f'Only PNG, TIF and TIFF are supported.')


def image_loader(
        image_list: list[SfMImage],
        return_image: bool = True,
        return_depth: bool = True,
        num_workers: int = 0,
        device: str = 'cpu'
):
    pin_memory = False if device.lower() == 'cpu' else True
    pin_memory_device = device if pin_memory else ''
    dataset = ImageDataset(image_list, return_image=return_image, return_depth=return_depth)
    loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x[0], pin_memory=pin_memory,
                        pin_memory_device=pin_memory_device)
    return loader
