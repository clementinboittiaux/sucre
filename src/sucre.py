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

import argparse
from pathlib import Path

import numpy as np
import torch
import loader
import yaml
from PIL import Image
from torch import Tensor
import gaussian_seathru
import normalization
from scipy.optimize import minimize
import sfm


def initialize_sucre_parameters(image: sfm.Image, channel: int, device: str = 'cpu') -> tuple[float, float, float]:
    Ic = loader.load_image(image.image_path)[:, :, channel].to(device)
    z = image.distance_map(loader.load_depth(image.depth_path).to(device))
    args_valid = z > 0
    B, beta, gamma = gaussian_seathru.solve_linear(Ic[args_valid], z[args_valid])
    return B, beta, gamma


def iter_data(data: list[tuple[Tensor, Tensor, Tensor, Tensor]],
              device: str = 'cpu') -> tuple[Tensor, Tensor, Tensor, Tensor]:
    for u, v, z, Ic in data:
        yield u.to(device).long(), v.to(device).long(), z.to(device), Ic.to(device)


def compute_B_and_J(
        image: sfm.Image,
        data: list[tuple[Tensor, Tensor, Tensor, Tensor]],
        beta: float,
        gamma: float,
        device: str = 'cpu'
) -> tuple[Tensor, Tensor]:
    sum_Ii_betai = torch.zeros(image.camera.height, image.camera.width, device=device)
    sum_bi_betai = torch.zeros_like(sum_Ii_betai)
    sum_betai2 = torch.zeros_like(sum_Ii_betai)
    sum_mi_ni = torch.zeros_like(sum_Ii_betai)
    sum_ni2 = torch.zeros_like(sum_Ii_betai)

    for ui, vi, zi, Ii in iter_data(data, device=device):
        betai = torch.exp(-beta * zi)
        bi = 1 - torch.exp(-gamma * zi)
        sum_Ii_betai[vi, ui] += Ii * betai
        sum_bi_betai[vi, ui] += bi * betai
        sum_betai2[vi, ui] += torch.square(betai)

    A = sum_Ii_betai / sum_betai2
    D = sum_bi_betai / sum_betai2

    for ui, vi, zi, Ii in iter_data(data, device=device):
        betai = torch.exp(-beta * zi)
        bi = 1 - torch.exp(-gamma * zi)
        mi = A[vi, ui] * betai - Ii
        ni = D[vi, ui] * betai - bi
        sum_mi_ni[vi, ui] += mi * ni
        sum_ni2[vi, ui] += torch.square(ni)

    B = sum_mi_ni.sum() / sum_ni2.sum()
    J = A - B * D
    return B, J


def solve_sucre(image: sfm.Image, channel: int, matches_file: loader.MatchesFile, device: str = 'cpu'):
    print('Initialize parameters with Gaussian Sea-thru.')
    Bc_init, betac_init, gammac_init = initialize_sucre_parameters(image, channel=channel, device=device)
    print(f'B: {Bc_init}\nbeta: {betac_init}\ngamma: {gammac_init}')

    data = matches_file.load_channel(channel)

    def residuals(x):
        betac_hat, gammac_hat = x.tolist()
        Bc_hat, Jc_hat = compute_B_and_J(image=image, data=data, beta=betac_hat, gamma=gammac_hat, device=device)
        res = torch.zeros(image.camera.height, image.camera.width, device=device)
        for ui, vi, zi, Ii in iter_data(data, device=device):
            res[vi, ui] += torch.square(
                Ii - Jc_hat[vi, ui] * torch.exp(-betac_hat * zi) - Bc_hat * (1 - torch.exp(-gammac_hat * zi))
            )
        res = (res / len(matches_file)).sum()
        return res.item()

    print('Start SUCRe optimization.')
    parameters = minimize(
        residuals,
        np.array([betac_init, gammac_init]),
        method='Nelder-Mead',
        bounds=[(0, np.inf), (0, np.inf)],
        options={'maxiter': 10000, 'disp': True}
    )

    betac, gammac = parameters.x.tolist()
    Bc, Jc = compute_B_and_J(image=image, data=data, beta=betac, gamma=gammac, device=device)

    print(f'B: {Bc}\nbeta: {betac}\ngamma: {gammac}')

    return Jc.cpu(), Bc.item(), betac, gammac


def sucre(
        colmap_model: sfm.COLMAPModel,
        image_name: str,
        output_dir: Path,
        min_cover: float,
        filter_image_names: list[str] = None,
        num_workers: int = 0,
        device: str = 'cpu'
):
    image = colmap_model[image_name]
    image_list = list(colmap_model.images.values())

    # Filter images that should not be used for pairing
    if filter_image_names is not None:
        image_list = [im for im in image_list if im.name not in filter_image_names]

    print(f'Compute {image_name} matches.')
    matches_file = loader.MatchesFile((output_dir / image_name).with_suffix('.h5'))
    image.match_images(
        image_list=image_list,
        matches_file=matches_file,
        min_cover=min_cover,
        num_workers=num_workers,
        device=device
    )

    print('Prepare matches for optimization.')
    matches_file.prepare_matches(num_workers=num_workers, device=device)

    J = torch.full((image.camera.height, image.camera.width, 3), torch.nan, dtype=torch.float32)
    for channel in range(3):
        print(f'SUCRe optimization on {["red", "green", "blue"][channel]} channel.')
        J[:, :, channel], Bc, betac, gammac = solve_sucre(
            image=image, channel=channel, matches_file=matches_file, device=device
        )

    print('Save restored image.')
    Image.fromarray(
        np.uint8(normalization.histogram_stretching(J) * 255)
    ).save((output_dir / image_name).with_suffix('.png'))


def parse_args(args: argparse.Namespace):
    print('Loading COLMAP model.')
    colmap_model = sfm.COLMAPModel(model_dir=args.model_dir, image_dir=args.image_dir, depth_dir=args.depth_dir)

    filter_image_names = args.filter_images_path.read_text().splitlines() if args.filter_images_path else None

    if args.image_name is not None:
        image_names = [args.image_name]
    elif args.image_list_path is not None:
        image_names = args.image_list_path.read_text().splitlines()
    else:
        image_names = []
        for image_id in range(args.image_id_range[0], args.image_id_range[1] + 1):
            if image_id in colmap_model.images:
                image_names.append(colmap_model.images[image_id].name)

    for image_name in image_names:
        sucre(
            colmap_model=colmap_model,
            image_name=image_name,
            output_dir=args.output_dir,
            min_cover=args.min_cover,
            filter_image_names=filter_image_names,
            num_workers=args.num_workers,
            device=args.device
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SUCRe.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image-dir', required=True, type=Path, help='path to images directory.')
    parser.add_argument('--depth-dir', required=True, type=Path, help='path to depth maps directory.')
    parser.add_argument('--model-dir', required=True, type=Path, help='path to undistorted COLMAP model directory.')
    parser.add_argument('--output-dir', required=True, type=Path, help='path to output directory.')
    parser_images = parser.add_mutually_exclusive_group(required=True)
    parser_images.add_argument('--image-name', type=str, help='name of image to restore.')
    parser_images.add_argument('--image-list', type=Path,
                               help='path to .txt file with names of images to restore, one name per line.')
    parser_images.add_argument('--image-ids', type=int, nargs=2, metavar=('MIN_ID', 'MAX_ID'),
                               help='range of ids of images to restore in the COLMAP model [min, max].')
    parser.add_argument('--min-cover', type=float, default=0.01,
                        help='minimum percentile of shared observations to keep the pairs of an image.')
    parser.add_argument('--filter-images-path', type=Path,
                        help='path to a .txt file with names of images to '
                             'discard when computing matches, one name per line.')
    parser.add_argument('--num-workers', type=int, default=0, help='number of threads, 0 is the main thread.')
    parser.add_argument('--device', type=str, default='cpu', help='device for heavy computation.')

    parse_args(parser.parse_args())
