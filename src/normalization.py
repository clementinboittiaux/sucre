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

import numpy as np
import torch
from torch import Tensor

import loader
import sfm


def estimate_Jcmin_zdcp(
        data: loader.Data,
        image: sfm.Image,
        channel: int,
        Bc: float,
        betac: float,
        gammac: float,
        mode: str = 'single-view',
        device: str = 'cpu'
) -> float:
    print(f'Estimate expected minimum value of Jc with distance-dependant DCP ({mode} mode).')
    match mode:
        case 'single-view':
            Ic = loader.load_image(image.image_path)[:, :, channel].to(device)
            z = image.distance_map(loader.load_depth(image.depth_path).to(device))
            args_valid = z > 0
            Ic = Ic[args_valid]
            z = z[args_valid]
        case 'multi-view':
            Ic, z = data.to_Ic_z()
        case 'none':
            return -torch.inf
        case _:
            raise ValueError("`mode` only supports 'single-view', 'multi-view' and 'none'.")
    z_bounds = np.linspace(z.min().item(), z.max().item(), 11)
    Ic_low, z_low = [], []
    for z_min, z_max in zip(z_bounds[:-1], z_bounds[1:]):
        args_range = (z >= z_min) & (z < z_max)
        if args_range.sum() > 0:
            Ic_range = Ic[args_range].cpu()
            z_range = z[args_range].cpu()
            args_low = Ic_range < np.percentile(Ic_range, 100 / 256)
            Ic_low.append(Ic_range[args_low])
            z_low.append(z_range[args_low])
    Ic_low, z_low = torch.hstack(Ic_low), torch.hstack(z_low)
    Jcmin = torch.median((Ic_low - Bc * (1 - torch.exp(-gammac * z_low))) * torch.exp(betac * z_low))
    return Jcmin.item()


def filter_min_outliers(image: Tensor, thresholds: Tensor):
    image = image.clone()
    image[image < thresholds] = torch.nan
    return image


def histogram_stretching(image: Tensor):
    image = image.numpy().copy()
    valid = np.all(~np.isnan(image), axis=2)
    image_valid = image[valid]
    image_valid = np.clip(image_valid, np.percentile(image_valid, 1, axis=0), np.percentile(image_valid, 99, axis=0))
    image_valid = image_valid - np.min(image_valid, axis=0)
    image_valid = image_valid / np.max(image_valid, axis=0)
    image[~valid] = 0
    image[valid] = image_valid
    return image
