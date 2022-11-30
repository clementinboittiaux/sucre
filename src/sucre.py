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
import datetime
from pathlib import Path

import numpy as np
import torch
import tqdm
import loader
import yaml
from PIL import Image
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import gaussian_seathru
import sfm


def initialize_sucre_parameters(image: sfm.Image, channel: int, device: str = 'cpu') -> tuple[float, float, float]:
    Ic = loader.load_image(image.image_path)[:, :, channel].to(device)
    z = image.distance_map(loader.load_depth(image.depth_path).to(device))
    args_valid = z > 0
    B, beta, gamma = gaussian_seathru.solve_linear(Ic[args_valid], z[args_valid])
    return B, beta, gamma


class SUCReModel(torch.nn.Module):
    def __init__(self, image: sfm.Image, B: list[float, float, float], beta, gamma):
        """SUCRe model

        :param image: image to restore.
        """
        super().__init__()
        self.J = torch.nn.Parameter(loader.load_image(image.image_path))
        self.B = torch.nn.Parameter(torch.tensor(B, dtype=torch.float32))
        self.beta = torch.nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        self.gamma = torch.nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.args_valid = loader.load_depth(image.depth_path) > 0

    def forward(self, u: Tensor, v: Tensor, z: Tensor) -> Tensor:
        """Compute the underwater image formation model

        :param u: u pixels coordinates for J.
        :param v: v pixels coordinates for J.
        :param z: pixels distances.
        :return: the image at coordinates (u, v), as seen at distances z,
        given the current image formation model parameters.
        """
        z = z.unsqueeze(dim=1).repeat(1, 3)
        return self.J[v, u] * torch.exp(-self.beta * z) + self.B * (1 - torch.exp(-self.gamma * z))

    def restore(self) -> np.array:
        """Normalize J (histogram stretching)

        :return: normalized J
        """
        J = self.J.detach().cpu().numpy()
        args_valid = self.args_valid.cpu().numpy()
        J_valid = J[args_valid]
        J_valid = np.clip(J_valid, np.percentile(J_valid, 1, axis=0), np.percentile(J_valid, 99, axis=0))
        J_valid = J_valid - J_valid.min(axis=0)
        J_valid = J_valid / J_valid.max(axis=0)
        J_plot = np.zeros_like(J)
        J_plot[args_valid] = J_valid
        return J_plot


def optimize(image: sfm.Image, channel: int, matches_file: loader.MatchesFile, device: str = 'cpu'):
    print('Initialize parameters with Gaussian Sea-thru.')
    B, beta, gamma = initialize_sucre_parameters(image, channel=channel, device=device)
    print(f'B: {B}\nbeta: {beta}\ngamma: {gamma}')

    beta = torch.tensor(beta, dtype=torch.float32, device=device)
    gamma = torch.tensor(gamma, dtype=torch.float32, device=device)

    data = matches_file.load_channel(channel)

    sum_Ii_betai = torch.zeros(image.camera.height, image.camera.height, device=device)
    sum_bi_betai = torch.zeros_like(sum_Ii_betai)
    sum_betai2 = torch.zeros_like(sum_Ii_betai)
    for ui, vi, zi, Ii in data:
        ui, vi = ui.long(), vi.long()
        zi, Ii = zi.to(device), Ii.to(device)
        betai = torch.exp(-beta * zi)
        bi = 1 - torch.exp(-gamma * zi)
        sum_Ii_betai[vi, ui] += Ii * betai
        sum_bi_betai[vi, ui] += bi * betai
        sum_betai2[vi, ui] += torch.square(betai)
    A = sum_Ii_betai / sum_betai2
    D = sum_bi_betai / sum_betai2


    print('super')






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

    B, beta, gamma = [], [], []
    for channel in range(3):
        print(f'SUCRe optimization on {["red", "green", "blue"][channel]} channel.')
        optimize(image=image, channel=channel, matches_file=matches_file, device=device)
        # Bc, betac, gammac = initialize_sucre_parameters(image, channel=channel, device=device)
        # B.append(Bc)
        # beta.append(betac)
        # gamma.append(gammac)

    raise Exception

    # Initialize SUCRe's model parameters
    print(f'Restoring {image_name}.')
    model = SUCReModel(image, B, beta, gamma)
    model.to(device)

    # Setup Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # Setup logger
    writer = SummaryWriter(str(output_dir / datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))

    # Setup data
    u, v, z, I = matches_file.load_all()

    epochs = 150
    split_size = 2097152
    epoch_losses = []

    for epoch in tqdm.tqdm(range(epochs)):  # For each optimization step

        epoch_loss = 0
        optimizer.zero_grad()  # Set the gradient to 0

        for u_split, v_split, I_split, z_split in zip(u.split(split_size), v.split(split_size), I.split(split_size),
                                                      z.split(split_size)):  # For each data sample
            # Switch to correct device
            u_split = u_split.to(device)
            v_split = v_split.to(device)
            I_split = I_split.to(device)
            z_split = z_split.to(device)

            # Compute the negative log likelihood of observing the acquired image
            I_hat = model(u_split, v_split, z_split)
            loss = torch.square(I_split - I_hat).sum()

            # Compute gradient
            loss.backward()
            epoch_loss += loss.item()

        # Apply gradient
        optimizer.step()

        # Log results
        epoch_losses.append(epoch_loss)
        writer.add_scalar('loss', epoch_loss, epoch)
        writer.flush()

    writer.close()

    # Save J without any normalization
    np.savez(
        str((output_dir / image_name).with_suffix('.npz')),
        J=model.J.detach().cpu().numpy(),
        args_valid=model.args_valid.cpu().numpy()
    )

    # Normalize and then save J as an image
    Image.fromarray(np.uint8(model.restore() * 255)).save((output_dir / image_name).with_suffix('.png'))

    # Save estimated underwater image formation model parameters
    sucre_params_path = (output_dir / image_name).with_suffix('.yml')
    with open(sucre_params_path, 'w') as sucre_params:
        yaml.dump({
            'beta': model.beta.detach().cpu().numpy().tolist(),
            'B': model.B.detach().cpu().numpy().tolist(),
            'gamma': model.gamma.detach().cpu().numpy().tolist()
        }, sucre_params)


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
