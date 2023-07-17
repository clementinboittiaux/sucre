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

from pathlib import Path

import numpy as np
import torch
import kornia
from torch import Tensor

import loader
import sfm
import utils


def estimate_Jc_bounds(
        data: loader.Data,
        image: sfm.Image,
        channel: int,
        Bc: float,
        betac: float,
        gammac: float,
        mode: str = 'single-view',
        params_path: Path = None,
        device: str = 'cpu'
) -> tuple[float, float]:
    print(f'Estimate expected minimum and maximum value of Jc with distance-dependant DCP ({mode} mode).')
    match mode:
        case 'global':
            params = utils.read_params_path(params_path)
            return params['Jmin'][channel], params['Jmax'][channel]
        case 'single-view':
            Ic = loader.load_image(image.image_path)[:, :, channel].to(device)
            z = image.distance_map(loader.load_depth(image.depth_path).to(device))
            args_valid = z > 0
            Ic = Ic[args_valid]
            z = z[args_valid]
        case 'multi-view':
            Ic, z = data.to_Ic_z(device=device)
        case 'none':
            return -torch.inf, torch.inf
        case _:
            raise ValueError("`mode` only supports 'global', 'single-view', 'multi-view' and 'none'.")
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
    return Jcmin.item(), torch.inf


def filter_outliers(J: Tensor, Jmin: Tensor, Jmax: Tensor) -> Tensor:
    J = J.clone()
    J[J < Jmin] = torch.nan
    J = J.clip(max=Jmax)
    return J


def histogram_stretching(image: Tensor) -> np.array:
    image = image.numpy().copy()
    valid = np.all(~np.isnan(image), axis=2)
    image_valid = image[valid]
    image_valid = np.clip(image_valid, np.percentile(image_valid, 1, axis=0), np.percentile(image_valid, 99, axis=0))
    image_valid = image_valid - np.min(image_valid, axis=0)
    image_valid = image_valid / np.max(image_valid, axis=0)
    image[~valid] = 0
    image[valid] = image_valid
    return image


def white_balance(image: Tensor) -> np.array:
    image = image.numpy().copy()
    valid = np.all(~np.isnan(image), axis=2)
    image_valid = image[valid]
    image_valid = image_valid / image_valid.mean(axis=0)
    image_valid = np.clip(image_valid, np.percentile(image_valid, 3), np.percentile(image_valid, 97))
    image_valid = image_valid - np.min(image_valid)
    image_valid = image_valid / np.max(image_valid)
    image[~valid] = 0
    image[valid] = image_valid
    return image


def tone_map(image: np.array) -> np.array:
    image = torch.tensor(image)
    image = kornia.color.linear_rgb_to_rgb(image.movedim(2, 0)).movedim(0, 2)
    return image.numpy()
