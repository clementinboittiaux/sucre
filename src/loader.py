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


# class Data:
#     def __init__(self):
#         self.data: list[dict[str, Tensor]] = []
#
#     def append(self, u: Tensor, v: Tensor, z: Tensor, Ic: Tensor):
#         self.data.append({'u': u, 'v': v, 'z': z, 'Ic': Ic})
#
#     def to_Ic_z(self, device: str = 'cpu') -> tuple[Tensor, Tensor]:
#         z = torch.full((len(self),), torch.nan, dtype=torch.float32, device=device)
#         Ic = torch.full((len(self),), torch.nan, dtype=torch.float32, device=device)
#         cursor = 0
#         for sample in self.data:
#             length = sample['z'].shape[0]
#             z[cursor: cursor + length] = sample['z'].to(device)
#             Ic[cursor: cursor + length] = sample['Ic'].to(device)
#             cursor += length
#         return Ic, z
#
#     def iter(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
#         for sample in self.data:
#             yield sample['u'].long(), sample['v'].long(), sample['z'], sample['Ic']
#
#     def iterbatch(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
#         for i in range(0, len(self.data), batch_size):
#             yield (
#                 torch.hstack([sample['u'] for sample in self.data[i:i + batch_size]]).long(),
#                 torch.hstack([sample['v'] for sample in self.data[i:i + batch_size]]).long(),
#                 torch.hstack([sample['z'] for sample in self.data[i:i + batch_size]]),
#                 torch.hstack([sample['Ic'] for sample in self.data[i:i + batch_size]])
#             )
#
#     def __len__(self):
#         return sum([sample['Ic'].shape[0] for sample in self.data])


class MatchesFile:
    def __init__(self, path: Path, colmap_model: sfm.COLMAPModel, overwrite: bool = False):
        if overwrite:
            path.unlink(missing_ok=True)
        self.path = path
        self.colmap_model = colmap_model

    def get_image_list(self) -> list[sfm.Image]:
        with h5py.File(self.path, 'r', libver='latest') as f:
            image_list = [self.colmap_model[group_name] for group_name in f]
        return image_list

    def save_matches(self, matches: sfm.Matches, d: Tensor):
        with h5py.File(self.path, 'a', libver='latest') as f:
            group = f.create_group(matches.image2.name)
            group.create_dataset('u1', data=matches.u1.short().cpu().numpy())
            group.create_dataset('v1', data=matches.v1.short().cpu().numpy())
            group.create_dataset('u2', data=matches.u2.short().cpu().numpy())
            group.create_dataset('v2', data=matches.v2.short().cpu().numpy())
            group.create_dataset('d', data=d.cpu().numpy())
            group.create_dataset('I', data=np.full((3, len(matches)), np.nan, dtype=np.float32))

    def prepare_matches(self, num_workers: int = 0):
        image_list = self.get_image_list()
        with h5py.File(self.path, 'r+', libver='latest') as f:
            for image_idx, image_rgb in tqdm.tqdm(
                    load_image_list(image_list, return_depth_map=False, num_workers=num_workers)):
                image = image_list[image_idx]
                group = f[image.name]
                u2 = torch.tensor(group['u2'][()], dtype=torch.int64)
                v2 = torch.tensor(group['v2'][()], dtype=torch.int64)
                group['I'][()] = image_rgb[v2, u2].T.numpy()

    def check_integrity(self):
        with h5py.File(self.path, 'r', libver='latest') as f:
            for group in f.values():
                for dataset_name in ['u1', 'v1', 'u2', 'v2', 'd', 'I']:
                    dataset = group[dataset_name]
                    dataset_data = dataset[()]
                    assert not np.isnan(dataset_data).any(), f'In {self.path}, dataset {dataset.name} contains NaN(s).'
                    if dataset_name in ['u1', 'v1', 'u2', 'v2', 'I']:
                        assert np.all(dataset_data >= 0), \
                            f'In {self.path}, dataset {dataset.name} contains invalid value(s).'
                    if dataset_name == 'd':
                        assert np.all(dataset_data > 0), \
                            f'In {self.path}, dataset {dataset.name} contains null of negative depth(s).'

    def load_matches(self, image: sfm.Image,
                     device: str = 'cpu') -> tuple[Tensor, Tensor]:
        size = len(self)
        i_idx = torch.zeros(size, dtype=torch.int64, device=device)
        p_idx = torch.zeros(size, dtype=torch.int64, device=device)
        cP = torch.zeros((3, size), dtype=torch.float32, device=device)
        I = torch.zeros((3, size), dtype=torch.float32, device=device)

        cursor = 0
        with h5py.File(self.path, 'r', libver='latest') as f:
            for i, (group_name, group) in enumerate(f.items()):
                u1 = torch.tensor(group['u1'][()], device=device, dtype=torch.int64)
                v1 = torch.tensor(group['v1'][()], device=device, dtype=torch.int64)
                length = u1.shape[0]
                i_idx[cursor:cursor+length] = i
                p_idx[cursor:cursor+length] = v1 * image.camera.width + u1
                u2 = torch.tensor(group['u2'][()], device=device) + 0.5
                v2 = torch.tensor(group['v2'][()], device=device) + 0.5
                d = torch.tensor(group['d'][()], device=device)
                cP[:, cursor:cursor+length] = image.unproject_depth(u=u2, v=v2, d=d)
                I[:, cursor:cursor+length] = torch.tensor(group['I'][()], device=device)
                cursor += length

        cP = torch.sparse_coo_tensor(
            indices=torch.stack([i_idx, p_idx]),
            values=cP.T,
            size=(i, image.camera.height * image.camera.width, 3)
        ).coalesce()
        I = torch.sparse_coo_tensor(
            indices=torch.stack([i_idx, p_idx]),
            values=I.T,
            size=(i, image.camera.height * image.camera.width, 3)
        ).coalesce()

        return cP, I

        # u, v, cP, I = [], [], [], []
        # with h5py.File(self.path, 'r', libver='latest') as f:
        #     for group_name, group in f.items():
        #         image = self.colmap_model[group_name]
        #         u.append(torch.tensor(group['u1'][()], device=device))
        #         v.append(torch.tensor(group['v1'][()], device=device))
        #         u2 = torch.tensor(group['u2'][()], device=device) + 0.5
        #         v2 = torch.tensor(group['v2'][()], device=device) + 0.5
        #         d = torch.tensor(group['d'][()], device=device)
        #         cP.append(image.unproject_depth(u=u2, v=v2, d=d))
        #         I.append(torch.tensor(group['I'][()], device=device))
        # return u, v, cP, I

    def __len__(self) -> int:
        size = 0
        if self.path.exists():
            with h5py.File(self.path, 'r', libver='latest') as f:
                for group in f.values():
                    size += group['u1'].shape[0]
        return size

    def __repr__(self) -> str:
        """String representation"""
        return f'MatchesFile(path={self.path}, {len(self)} observations)'


class ImageDataset(Dataset):
    def __init__(self, image_list: list[sfm.Image], return_rgb: bool = True, return_depth_map: bool = True):
        self.images = image_list
        self.return_rgb = return_rgb
        self.return_depth_map = return_depth_map

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.return_rgb and not self.return_depth_map:
            return idx, load_rgb(image.rgb_path, width=image.camera.width, height=image.camera.height)
        elif self.return_depth_map and not self.return_rgb:
            return idx, load_depth_map(image.depth_map_path, width=image.camera.width, height=image.camera.height)
        elif self.return_rgb and self.return_depth_map:
            return (
                idx,
                load_rgb(image.rgb_path, width=image.camera.width, height=image.camera.height),
                load_depth_map(image.depth_map_path, width=image.camera.width, height=image.camera.height)
            )


def load_rgb(rgb_path: Path, width: int, height: int) -> Tensor:
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB) / 255
    if (rgb.shape[0] != height) or (rgb.shape[1] != width):
        rgb = cv2.resize(
            rgb, (width, height),
            interpolation=cv2.INTER_AREA if width < rgb.shape[1] else cv2.INTER_CUBIC
        )
    return torch.tensor(rgb, dtype=torch.float32)


def load_depth_map(depth_map_path: Path, width: int, height: int) -> Tensor:
    depth_map = cv2.imread(str(depth_map_path), cv2.IMREAD_UNCHANGED) / 1000
    if (depth_map.shape[0] != height) or (depth_map.shape[1] != width):
        depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_NEAREST)
    return torch.tensor(depth_map, dtype=torch.float32)


def load_image_list(
        image_list: list[sfm.Image],
        return_rgb: bool = True,
        return_depth_map: bool = True,
        num_workers: int = 0
) -> DataLoader:
    dataset = ImageDataset(image_list, return_rgb=return_rgb, return_depth_map=return_depth_map)
    return DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x[0])
