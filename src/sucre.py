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
import matplotlib.pyplot as plt
from PIL import Image
from torch import Tensor

import loader
import sfm
import se3


class SUCRe(torch.nn.Module):
    def __init__(self, image: sfm.Image, light_model: bool = False):
        super().__init__()
        self.image = image
        self.light_model = light_model
        self.J = torch.nn.Parameter(image.get_rgb())
        with torch.no_grad():
            self.J[image.get_depth_map() <= 0] = torch.nan
        self.B = torch.nn.Parameter(torch.tensor([[0.1], [0.1], [0.1]]))
        self.beta = torch.nn.Parameter(torch.tensor([[0.1], [0.1], [0.1]]))
        self.gamma = torch.nn.Parameter(torch.tensor([[0.1], [0.1], [0.1]]))
        if self.light_model:
            self.cam2light = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            self.sigma = torch.nn.Parameter(torch.eye(2))

    def compute_l_z(self, cP: Tensor) -> tuple[float | Tensor, Tensor]:
        z = cP.norm(dim=0)
        if self.light_model:
            R, t = se3.exp(self.cam2light)
            Sigma = self.sigma.T @ self.sigma
            lP = R @ cP + t
            lp = lP[:2] / lP[2]
            lp = lp.T.unsqueeze(dim=2)
            l = torch.exp(-torch.flatten(lp.transpose(1, 2) @ Sigma.inverse() @ lp) / 2)
            z += lP.norm(dim=0)
        else:
            l = 1.0
        return l, z

    def compute_J(self, matches_data: loader.MatchesData) -> Tensor:
        J_numerator = torch.zeros((self.image.camera.height, self.image.camera.width, 3), device=self.B.device)
        J_denominator = torch.zeros((self.image.camera.height, self.image.camera.width, 3), device=self.B.device)

        for u, v, cP, I in matches_data.iter(device=self.B.device):
            l, z = self.compute_l_z(cP)
            absorption = l * torch.exp(-self.beta * z)
            backscatter = l * self.B * (1 - torch.exp(-self.gamma * z))
            J_numerator[v, u] += ((I - backscatter) * absorption).T
            J_denominator[v, u] += absorption.square().T

        J = J_numerator / J_denominator
        return J

    def forward(self, u: Tensor, v: Tensor, cP: Tensor) -> Tensor:
        l, z = self.compute_l_z(cP)
        I_hat = l * (self.J[v, u].T * torch.exp(-self.beta * z) + self.B * (1 - torch.exp(-self.gamma * z)))
        return I_hat

    def plot_J(self, matches_data: loader.MatchesData = None):
        with torch.no_grad():
            if matches_data is not None:
                J = self.compute_J(matches_data)
            else:
                J = self.J.cpu().numpy().copy()
            valid = np.all(~np.isnan(J), axis=2)
            J_valid = J[valid]
            J_valid = np.clip(J_valid, np.percentile(J_valid, 1, axis=0), np.percentile(J_valid, 99, axis=0))
            J_valid = J_valid - np.min(J_valid, axis=0)
            J_valid = J_valid / np.max(J_valid, axis=0)
            J[~valid] = 0
            J[valid] = J_valid
        return Image.fromarray(np.uint8(J * 255))

    def plot_l(self):
        with torch.no_grad():
            u, v, cP = self.image.unproject_depth_map(
                self.image.get_depth_map().to(self.cam2light.device), to_world=False
            )
            l, _ = self.compute_l_z(cP)
            l_map = torch.zeros((self.image.camera.height, self.image.camera.width), device=l.device)
            l_map[v, u] = l
        return Image.fromarray(np.uint8(plt.colormaps['jet'](l_map.cpu().numpy())[:, :, :3] * 255))


def adam(
        sucre: SUCRe,
        matches_data: loader.MatchesData,
        num_iter: int = 200,
        batch_size: int = 1,
        save_dir: Path = None,
        save_interval: int = None,
        device: str = 'cpu'
):
    print(f'Solve least squares with Adam optimizer ({num_iter} iterations).')
    n_obs = len(matches_data)
    optimizer = torch.optim.Adam(sucre.parameters(), lr=0.05)

    for iteration in range(num_iter):
        cost = 0
        optimizer.zero_grad()

        for u, v, cP, I in matches_data.iter(batch_size=batch_size, device=device):
            loss = torch.square(I - sucre(u=u, v=v, cP=cP)).sum() / n_obs / 3
            loss.backward()
            cost += loss.item()

        optimizer.step()
        with np.printoptions(precision=4):
            print(f'iter: {iteration:04d}, cost: {cost:.4e}, B: {sucre.B.detach().cpu().flatten().numpy()}, '
                  f'beta: {sucre.beta.detach().cpu().flatten().numpy()}, '
                  f'gamma: {sucre.gamma.detach().cpu().flatten().numpy()}')
        if save_dir is not None and save_interval is not None and iteration % save_interval == 0:
            save_path = (save_dir / sucre.image.name).with_suffix('.png')
            sucre.plot_J().save(save_path.with_stem(f'{save_path.stem}_rgb_{iteration:04d}'))
            if sucre.light_model:
                sucre.plot_l().save(save_path.with_stem(f'{save_path.stem}_vignetting_{iteration:04d}'))


def restore_image(
        image: sfm.Image,
        colmap_model: sfm.COLMAPModel,
        min_cover: float,
        image_list: list[sfm.Image],
        output_dir: Path,
        light_model: bool = False,
        solver: str = 'adam',
        max_iter: int = 200,
        batch_size: int = 1,
        save_interval: int = None,
        force_compute_matches: bool = False,
        keep_matches: bool = False,
        num_workers: int = 0,
        device: str = 'cpu'
):
    print(f'Restore {image.name}.')
    matches_path = (output_dir / image.name).with_suffix('.h5')
    matches_file = loader.MatchesFile(matches_path, colmap_model=colmap_model, overwrite=force_compute_matches)

    if force_compute_matches or not matches_path.exists():
        print(f'Compute {image.name} matches.')
        image.match_images(
            image_list=image_list,
            matches_file=matches_file,
            min_cover=min_cover,
            num_workers=num_workers,
            device=device
        )
        print('Prepare matches for optimization.')
        matches_file.prepare_matches(num_workers=num_workers)

    print('Check matches integrity.')
    matches_file.check_integrity()

    print('Load matches.')
    matches_data = matches_file.load_matches(pin_memory=True)
    print(f'Total of {len(matches_data)} observations.')

    sucre = SUCRe(image=image, light_model=light_model).to(device)

    match solver:
        case 'adam':
            adam(sucre=sucre, matches_data=matches_data, num_iter=max_iter, batch_size=batch_size,
                 save_dir=output_dir, save_interval=save_interval, device=device)
        case _:
            raise ValueError('Currently, only `adam` optimizer is supported.')

    if not keep_matches:
        print(f'Erase {matches_path}.')
        matches_path.unlink()


def parse_args(args: argparse.Namespace):
    print('Loading COLMAP model.')
    colmap_model = sfm.COLMAPModel(
        model_dir=args.model_dir, image_dir=args.image_dir, depth_dir=args.depth_dir, image_scale=args.image_scale
    )

    if args.image_name is not None:
        images = [colmap_model[args.image_name]]
    elif args.image_list is not None:
        images = [colmap_model[image_name] for image_name in args.image_list.read_text().splitlines()]
    else:
        images = [
            colmap_model.images[image_id] for image_id in range(*args.image_ids) if image_id in colmap_model.images
        ]

    # Filter images that should not be used for pairing
    filter_image_names = args.filter_images_path.read_text().splitlines() if args.filter_images_path else []
    image_list = [im for im in colmap_model.images.values() if im.name not in filter_image_names]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image in images:
        restore_image(
            image=image,
            colmap_model=colmap_model,
            min_cover=args.min_cover,
            image_list=image_list,
            output_dir=args.output_dir,
            light_model=args.light_model,
            solver=args.solver,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            force_compute_matches=args.force_compute_matches,
            keep_matches=args.keep_matches,
            num_workers=args.num_workers,
            device=args.device
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SUCRe.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image-dir', required=True, type=Path, help='path to images directory.')
    parser.add_argument('--depth-dir', required=True, type=Path, help='path to depth maps directory.')
    parser.add_argument('--model-dir', required=True, type=Path,
                        help='path to undistorted COLMAP model directory.')
    parser.add_argument('--output-dir', required=True, type=Path, help='path to output directory.')
    parser_images = parser.add_mutually_exclusive_group(required=True)
    parser_images.add_argument('--image-name', type=str, help='name of image to restore.')
    parser_images.add_argument('--image-list', type=Path,
                               help='path to .txt file with names of images to restore, one name per line.')
    parser_images.add_argument('--image-ids', type=int, nargs=2, metavar=('MIN_ID', 'MAX_ID'),
                               help='range of ids of images to restore in the COLMAP model [min, max].')
    parser.add_argument('--light-model', action='store_true', help='model artificial lights.')
    parser.add_argument('--min-cover', type=float, default=0.01,
                        help='minimum percentile of shared observations to keep the pairs of an image.')
    parser.add_argument('--image-scale', type=float, default=1.0, help='rescale all images by this factor.')
    parser.add_argument('--filter-images-path', type=Path,
                        help='path to a .txt file with names of images to '
                             'discard when computing matches, one name per line.')
    parser.add_argument('--solver', type=str, choices=['adam', 'simplex'],
                        default='adam', help='method to solve SUCRe least squares.')
    parser.add_argument('--max-iter', type=int, default=200, help='maximum number of optimization steps.')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='batch size for adam optimization, higher is faster but requires more memory.')
    parser.add_argument('--save-interval', type=int,
                        help='save restored image every given iterations steps.')
    parser.add_argument('--force-compute-matches', action='store_true',
                        help='if matches file already exist, erase it and recompute matches.')
    parser.add_argument('--keep-matches', action='store_true',
                        help='keep matches file (can take a lot a space).')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of threads, 0 is the main thread.')
    parser.add_argument('--device', type=str, default='cpu', help='device for heavy computation.')

    parse_args(parser.parse_args())
