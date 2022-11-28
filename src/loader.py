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

from __future__ import annotations

import cv2
import torch
import numpy as np
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from colmap.scripts.python import read_write_model
import sfm
import tqdm
import h5py


class ImageDataset(Dataset):
    def __init__(self, image_list: list[sfm.Image], return_image: bool = True, return_depth: bool = True):
        self.images = image_list
        self.return_image = return_image
        self.return_depth = return_depth

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.return_image and not self.return_depth:
            return idx, load_image(image.image_path)
        elif self.return_depth and not self.return_image:
            return idx, load_depth(image.depth_path)
        elif self.return_image and self.return_depth:
            return idx, load_image(image.image_path), load_depth(image.depth_path)


def load_cameras(model_dir: Path) -> dict[int, read_write_model.Camera]:
    cameras_bin = model_dir / 'cameras.bin'
    cameras_txt = model_dir / 'cameras.txt'
    if cameras_bin.exists():
        return read_write_model.read_cameras_binary(cameras_bin)
    elif cameras_txt.exists():
        return read_write_model.read_cameras_text(cameras_txt)
    else:
        raise FileExistsError(f'No cameras file found in {model_dir}.')


def load_images(model_dir: Path) -> dict[int, read_write_model.Image]:
    images_bin = model_dir / 'images.bin'
    images_txt = model_dir / 'images.txt'
    if images_bin.exists():
        return read_write_model.read_images_binary(images_bin)
    elif images_txt.exists():
        return read_write_model.read_images_text(images_txt)
    else:
        raise FileExistsError(f'No cameras file found in {model_dir}.')


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


def loader(
        image_list: list[sfm.Image],
        return_image: bool = True,
        return_depth: bool = True,
        num_workers: int = 0
) -> DataLoader:
    dataset = ImageDataset(image_list, return_image=return_image, return_depth=return_depth)
    return DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x[0])


def prepare_matches(
        image_list: list[sfm.Image],
        matches_path: Path,
        data_path: Path,
        num_workers: int = 0,
        device: str = 'cpu'
):
    with h5py.File(matches_path, 'r', libver='latest') as r, h5py.File(data_path, 'w', libver='latest') as w:

        # Find total number of observations
        size = 0
        for dataset in r.values():
            size += len(dataset['u1'])

        # Create one HDF5 dataset per parameter
        w.create_dataset('u', (size,), dtype='int16')
        w.create_dataset('v', (size,), dtype='int16')
        w.create_dataset('z', (size,), dtype='float32')
        w.create_dataset('Ir', (size,), dtype='float32')
        w.create_dataset('Ig', (size,), dtype='float32')
        w.create_dataset('Ib', (size,), dtype='float32')

        # Initialize HDF5 datasets for contiguous memory
        w['u'][()] = -1
        w['v'][()] = -1
        w['z'][()] = np.nan
        w['Ir'][()] = np.nan
        w['Ig'][()] = np.nan
        w['Ib'][()] = np.nan

        # Fill HDF5 datasets with J coordinates, pixel intensities and distances
        cursor = 0
        for image_idx, image_image, image_depth in tqdm.tqdm(loader(image_list, num_workers=num_workers)):
            image = image_list[image_idx]
            image_distance = image.distance_map(image_depth.to(device))
            matches = r[image.name]
            u1 = matches['u1'][()]
            v1 = matches['v1'][()]
            u2 = torch.tensor(matches['u2'][()], dtype=torch.int64)
            v2 = torch.tensor(matches['v2'][()], dtype=torch.int64)
            z = image_distance[v2, u2].cpu().numpy()
            Ir, Ig, Ib = image_image[v2, u2].T.numpy()
            length = u1.shape[0]
            w['u'][cursor:cursor + length] = u1
            w['v'][cursor:cursor + length] = v1
            w['z'][cursor:cursor + length] = z
            w['Ir'][cursor:cursor + length] = Ir
            w['Ig'][cursor:cursor + length] = Ig
            w['Ib'][cursor:cursor + length] = Ib
            cursor += length


def load_data(
        data_path: Path,
        chunk_size: int = 2**20,
        device: str = 'cpu'
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    with h5py.File(data_path, 'r', libver='latest') as f:
        for cursor in range(0, len(f['u']), chunk_size):
            u = torch.tensor(f['u'][cursor:cursor + chunk_size], dtype=torch.int64, device=device)
            v = torch.tensor(f['v'][cursor:cursor + chunk_size], dtype=torch.int64, device=device)
            z = torch.tensor(f['z'][cursor:cursor + chunk_size], device=device)
            Ir = torch.tensor(f['Ir'][cursor:cursor + chunk_size], device=device)
            Ig = torch.tensor(f['Ig'][cursor:cursor + chunk_size], device=device)
            Ib = torch.tensor(f['Ib'][cursor:cursor + chunk_size], device=device)
            yield u, v, z, Ir, Ig, Ib


def load_data_single_channel(
        data_path: Path,
        channel: int,
        chunk_size: int = 2**20,
        device: str = 'cpu'
) -> tuple[Tensor, Tensor]:
    with h5py.File(data_path, 'r', libver='latest') as f:
        for cursor in range(0, len(f['u']), chunk_size):
            z = torch.tensor(f['z'][cursor:cursor + chunk_size], device=device)
            match channel:
                case 0:
                    Ic = torch.tensor(f['Ir'][cursor:cursor + chunk_size], device=device)
                case 1:
                    Ic = torch.tensor(f['Ig'][cursor:cursor + chunk_size], device=device)
                case 2:
                    Ic = torch.tensor(f['Ib'][cursor:cursor + chunk_size], device=device)
            yield z, Ic
