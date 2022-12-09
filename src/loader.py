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

from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import sfm
from colmap.scripts.python import read_write_model


class MatchesFile:
    def __init__(self, path: Path, overwrite: bool = True):
        if overwrite:
            path.unlink(missing_ok=True)
        self.path = path

    def save_matches(self, matches: sfm.Matches):
        with h5py.File(self.path, 'a', libver='latest') as f:
            group = f.create_group(matches.image2.name)
            group.create_dataset('u1', data=matches.u1.short().cpu().numpy())
            group.create_dataset('v1', data=matches.v1.short().cpu().numpy())
            group.create_dataset('u2', data=matches.u2.short().cpu().numpy())
            group.create_dataset('v2', data=matches.v2.short().cpu().numpy())
            group.create_dataset('z', data=np.full(len(matches), np.nan, dtype=np.float32))
            group.create_dataset('I', data=np.full((3, len(matches)), np.nan, dtype=np.float32))

    def prepare_matches(self, colmap_model: sfm.COLMAPModel, num_workers: int = 0, device: str = 'cpu'):
        with h5py.File(self.path, 'r', libver='latest') as f:
            image_list = [colmap_model[group_name] for group_name in f.keys()]
        with h5py.File(self.path, 'r+', libver='latest') as f:
            for image_idx, image_image, image_depth in tqdm.tqdm(load_images(image_list, num_workers=num_workers)):
                image = image_list[image_idx]
                image_distance = image.distance_map(image_depth.to(device))
                group = f[image.name]
                u2 = torch.tensor(group['u2'][()], dtype=torch.int64)
                v2 = torch.tensor(group['v2'][()], dtype=torch.int64)
                group['z'][()] = image_distance[v2, u2].cpu().numpy()
                group['I'][()] = image_image[v2, u2].T.numpy()

    def load_channel(self, channel: int, pin_memory: bool = False) -> list[tuple[Tensor, Tensor, Tensor, Tensor]]:
        data = []
        with h5py.File(self.path, 'r', libver='latest') as f:
            for group in f.values():
                if pin_memory:
                    data.append((
                        torch.tensor(group['u1'][()]).pin_memory(),
                        torch.tensor(group['v1'][()]).pin_memory(),
                        torch.tensor(group['z'][()]).pin_memory(),
                        torch.tensor(group['I'][channel]).pin_memory()
                    ))
                else:
                    data.append((
                        torch.tensor(group['u1'][()]),
                        torch.tensor(group['v1'][()]),
                        torch.tensor(group['z'][()]),
                        torch.tensor(group['I'][channel])
                    ))
        return data

    def __len__(self) -> int:
        size = 0
        if self.path.exists():
            with h5py.File(self.path, 'r', libver='latest') as f:
                for group in f.values():
                    size += group['z'].shape[0]
        return size


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


def load_colmap_cameras(model_dir: Path) -> dict[int: read_write_model.Camera]:
    cameras_bin = model_dir / 'cameras.bin'
    cameras_txt = model_dir / 'cameras.txt'
    if cameras_bin.exists():
        return read_write_model.read_cameras_binary(cameras_bin)
    elif cameras_txt.exists():
        return read_write_model.read_cameras_text(cameras_txt)
    else:
        raise FileExistsError(f'No cameras file found in {model_dir}.')


def load_colmap_images(model_dir: Path) -> dict[int: read_write_model.Image]:
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


def load_images(
        image_list: list[sfm.Image],
        return_image: bool = True,
        return_depth: bool = True,
        num_workers: int = 0
) -> DataLoader:
    dataset = ImageDataset(image_list, return_image=return_image, return_depth=return_depth)
    return DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x[0])
