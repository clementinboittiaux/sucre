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
import yaml
from PIL import Image
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from sfm import COLMAPModel, SfMImage, Matches


class SUCRe(torch.nn.Module):
    def __init__(self, image: SfMImage, args_valid: Tensor):
        """SUCRe wrapper

        :param image: image to restore.
        :param args_valid: map of valid pixels to restore.
        """
        super().__init__()
        self.J = torch.nn.Parameter(image.image())
        self.B = torch.nn.Parameter(torch.tensor([0.25, 0.25, 0.25]))
        self.beta = torch.nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))
        self.gamma = torch.nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))
        self.args_valid = args_valid

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


def build_data(matches: list[Matches], count: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build observations from matches

    :param matches: list of pairs.
    :param count: pairs count map.
    :return: pixels coordinates and observations.
    """
    u, v, I, z = [], [], [], []
    for match in matches:
        u.append(match.u1)
        v.append(match.v1)
        I.append(match.I)
        z.append(match.z)
    u = torch.hstack(u).pin_memory()
    v = torch.hstack(v).pin_memory()
    I = torch.vstack(I).pin_memory()
    z = torch.hstack(z).pin_memory()
    return u, v, I, z, count


def restore_image(
        colmap_model: COLMAPModel,
        image_name: str,
        output_dir: Path,
        min_cover: float,
        filter_image_names: list[str] = None,
        num_workers: int = 0,
        device: str = 'cpu'
):
    """Apply SUCRe to given image

    :param colmap_model: COLMAP model object.
    :param image_name: name of image to be restored.
    :param output_dir: path to output directory.
    :param min_cover: minimum percentile of shared observations to keep the pairs of an image.
    :param filter_image_names: list of image names to discard for image pairing.
    :param num_workers: number of threads.
    :param device: device for heavy computation.
    """
    print(f'Computing {image_name} matches.')

    image = colmap_model[image_name]

    # Find all images that share voxels observations
    image_list = colmap_model.voxels.pairs(image.voxel_list, min_cover=min_cover)

    # Filter images that should not be used for pairing
    if filter_image_names is not None:
        image_list = [im for im in image_list if im.name not in filter_image_names]

    # Compute all pairs and prepare data
    u, v, I, z, count = build_data(*image.compute_matches(
        image_list=image_list,
        min_cover=min_cover,
        num_workers=num_workers,
        device=device
    ))

    # Only estimate pixels that have at least one pair
    args_valid = count > 0

    # Initialize SUCRe's model parameters
    print(f'Restoring {image_name}.')
    model = SUCRe(image, args_valid)
    model.to(device)

    # Setup Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # Setup logger
    writer = SummaryWriter(str(output_dir / datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))

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
    with open((output_dir / image_name).with_suffix('.yaml'), 'w') as yaml_file:
        yaml.dump({
            'beta': model.beta.detach().cpu().numpy().tolist(),
            'B': model.B.detach().cpu().numpy().tolist(),
            'gamma': model.gamma.detach().cpu().numpy().tolist()
        }, yaml_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SUCRe.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', required=True, type=Path,
                        help='path to data directory with folders `depth_maps`, `images` and `sparse`.')
    parser.add_argument('--output-dir', required=True, type=Path, help='path to output directory.')
    parser_images = parser.add_mutually_exclusive_group(required=True)
    parser_images.add_argument('--image-name', type=str, help='name of image to restore.')
    parser_images.add_argument('--image-list', type=Path,
                               help='path to .txt file with names of images to restore, one name per line.')
    parser_images.add_argument('--image-ids', type=int, nargs=2, metavar=('MIN_ID', 'MAX_ID'),
                               help='range of ids of images to restore in the COLMAP model [min, max].')
    parser.add_argument('--voxel-size', type=float, default=0.25,
                        help='voxel size for fast image retrieval, smaller is faster but requires more RAM.')
    parser.add_argument('--min-cover', type=float, default=0.01,
                        help='minimum percentile of shared observations to keep the pairs of an image.')
    parser.add_argument('--filter-images-path', type=Path,
                        help='path to a .txt file with names of images to '
                             'discard when computing matches, one name per line.')
    parser.add_argument('--num-workers', type=int, default=0, help='number of threads, 0 is the main thread.')
    parser.add_argument('--device', type=str, default='cpu', help='device for heavy computation.')
    args = parser.parse_args()

    print('Loading COLMAP model.')
    args_colmap = COLMAPModel(data_dir=args.data_dir, voxel_size=args.voxel_size)

    print('Computing voxels for fast image retrieval.')
    args_colmap.compute_voxels(list(args_colmap.images.values()), num_workers=args.num_workers, device=args.device)

    if args.filter_images_path is not None:
        args_filter_image_names = args.filter_images_path.read_text().splitlines()
    else:
        args_filter_image_names = None

    if args.image_name is not None:
        args_image_names = [args.image_name]
    elif args.image_list_path is not None:
        args_image_names = args.image_list_path.read_text().splitlines()
    else:
        args_image_names = []
        for args_image_id in range(args.image_id_range[0], args.image_id_range[1] + 1):
            if args_image_id in args_colmap.images:
                args_image_names.append(args_colmap.images[args_image_id].name)
    for args_image_name in args_image_names:
        restore_image(
            colmap_model=args_colmap,
            image_name=args_image_name,
            output_dir=args.output_dir,
            min_cover=args.min_cover,
            filter_image_names=args_filter_image_names,
            num_workers=args.num_workers,
            device=args.device
        )
