# Copyright (C) 2023 Ifremer
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
from tqdm import tqdm
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor

import loader
import se3
import sfm


class SUCRe(torch.nn.Module):
    def __init__(self, image: sfm.Image, light_model: bool = False, use_closed_form: bool = False):
        super().__init__()
        self.image = image
        self.light_model = light_model
        self.use_closed_form = use_closed_form
        self.B = torch.nn.Parameter(torch.tensor([[0.1], [0.1], [0.1]]))
        self.beta = torch.nn.Parameter(torch.tensor([[0.1], [0.1], [0.1]]))
        self.gamma = torch.nn.Parameter(torch.tensor([[0.1], [0.1], [0.1]]))
        if light_model:
            self.cam2light = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            self.sigma = torch.nn.Parameter(torch.eye(2))
        if not use_closed_form:
            self.J = image.get_rgb()
            self.J[image.get_depth_map() <= 0] = torch.nan
            self.J = torch.nn.Parameter(self.J)

    def compute_l_z(self, cP: Tensor) -> tuple[float | Tensor, Tensor]:
        z = cP.norm(dim=0)
        if self.light_model:
            R, t = se3.exp(self.cam2light)
            Sigma = self.sigma.T @ self.sigma
            lP = R @ cP + t
            lp = lP[:2] / lP[2]
            lp = lp.T.unsqueeze(dim=2)
            l = torch.exp(-torch.flatten(lp.transpose(1, 2) @ Sigma.inverse() @ lp) / 2)
        else:
            l = 1.0
        return l, z

    @torch.no_grad()
    def update_J(self, matches_data: loader.MatchesData, l: float | Tensor, z: Tensor, force_update: bool = False):
        if self.use_closed_form or force_update:
            absorption = l * torch.exp(-self.beta * z)
            backscatter = l * self.B * (1 - torch.exp(-self.gamma * z))
            J_numerator = torch.zeros((self.image.camera.height, self.image.camera.width, 3), device=self.B.device)
            J_denominator = torch.zeros((self.image.camera.height, self.image.camera.width, 3), device=self.B.device)
            J_index = (matches_data.v, matches_data.u)
            J_numerator.index_put_(J_index, ((matches_data.I - backscatter) * absorption).T, accumulate=True)
            J_denominator.index_put_(J_index, absorption.square().T, accumulate=True)
            self.J = J_numerator / J_denominator

    def forward(self, u: Tensor, v: Tensor, l: float | Tensor, z: Tensor) -> Tensor:
        I_hat = l * (self.J[v, u].T * torch.exp(-self.beta * z) + self.B * (1 - torch.exp(-self.gamma * z)))
        return I_hat

    @torch.no_grad()
    def plot_J(self):
        J = self.J.cpu().numpy().copy()
        valid = np.all(~np.isnan(J), axis=2)
        J_valid = J[valid]
        J_valid = np.clip(J_valid, np.percentile(J_valid, 1, axis=0), np.percentile(J_valid, 99, axis=0))
        J_valid = J_valid - np.min(J_valid, axis=0)
        J_valid = J_valid / np.max(J_valid, axis=0)
        J[~valid] = 0.0
        J[valid] = J_valid
        return Image.fromarray(np.uint8(J * 255))

    @torch.no_grad()
    def plot_l(self):
        u, v, cP = self.image.unproject_depth_map(
            self.image.get_depth_map().to(self.cam2light.device), to_world=False
        )
        l, _ = self.compute_l_z(cP)
        l_map = torch.zeros((self.image.camera.height, self.image.camera.width), device=l.device)
        l_map[v, u] = l
        return Image.fromarray(np.uint8(plt.colormaps['jet'](l_map.cpu().numpy())[:, :, :3] * 255))

    @torch.no_grad()
    def plot_reconstruction(self):
        u, v, cP = self.image.unproject_depth_map(
            self.image.get_depth_map().to(self.B.device), to_world=False
        )
        l, z = self.compute_l_z(cP)
        I_reconstructed = torch.zeros((self.image.camera.height, self.image.camera.width, 3), device=cP.device)
        I_reconstructed[v, u] = self(u=u, v=v, l=l, z=z).clip(0, 1).T
        return Image.fromarray(np.uint8(I_reconstructed.cpu().numpy() * 255))

    def save_plots(self, save_dir: Path, iteration: int = None):
        save_path = (save_dir / self.image.name).with_suffix('.png')
        suffix = '' if iteration is None else f'_{iteration:04d}'
        self.plot_J().save(save_path.with_stem(f'{save_path.stem}_rgb{suffix}'))
        self.plot_reconstruction().save(save_path.with_stem(f'{save_path.stem}_reconstruction{suffix}'))
        if self.light_model:
            self.plot_l().save(save_path.with_stem(f'{save_path.stem}_vignetting{suffix}'))


def adam(
        sucre: SUCRe,
        matches_data: loader.MatchesData,
        lr: float = 0.05,
        num_iter: int = 200,
        light_regularizer: float = 1e-4,
        save_dir: Path = None,
        save_interval: int = None
) -> SUCRe:
    print(f'Solve least squares with Adam optimizer ({num_iter} iterations).')
    n_obs = len(matches_data)
    optimizer = torch.optim.Adam(sucre.parameters(), lr=lr)

    for iteration in tqdm(range(num_iter)):
        optimizer.zero_grad()
        l, z = sucre.compute_l_z(matches_data.cP)
        sucre.update_J(matches_data=matches_data, l=l, z=z)

        I_hat = sucre(u=matches_data.u, v=matches_data.v, l=l, z=z)
        loss = torch.square(matches_data.I - I_hat).sum()
        (loss / n_obs / 3).backward()

        if sucre.light_model and light_regularizer > 0:
            t_z = se3.exp(sucre.cam2light)[1][2, 0]
            (light_regularizer * t_z.square()).backward()

        optimizer.step()

        with np.printoptions(precision=4):
            tqdm.write(f'iter: {iteration:04d}, cost: {loss.item():.4e}, '
                       f'B: {sucre.B.detach().cpu().flatten().numpy()}, '
                       f'beta: {sucre.beta.detach().cpu().flatten().numpy()}, '
                       f'gamma: {sucre.gamma.detach().cpu().flatten().numpy()}')
        if save_dir is not None and save_interval is not None and iteration % save_interval == 0:
            sucre.save_plots(save_dir=save_dir, iteration=iteration)

    l, z = sucre.compute_l_z(matches_data.cP)
    sucre.update_J(matches_data=matches_data, l=l, z=z)
    return sucre


def restore_image(
        image: sfm.Image,
        colmap_model: sfm.COLMAPModel,
        output_dir: Path,
        light_model: bool = False,
        use_closed_form: bool = False,
        min_cover: float = 0.000001,
        image_list: list[sfm.Image] = None,
        lr: float = 0.05,
        num_iter: int = 200,
        light_regularizer: float = 1e-4,
        save_interval: int = None,
        params_path: Path = None,
        force_compute_matches: bool = False,
        keep_matches: bool = False,
        num_workers: int = 0,
        device: str = 'cpu'
):
    print(f'Restore {image.name}.')
    matches_path = (output_dir / image.name).with_suffix('.h5')
    matches_file = loader.MatchesFile(matches_path, colmap_model=colmap_model, overwrite=force_compute_matches)

    if image_list is None:
        image_list = list(colmap_model.images.values())

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
    matches_data = matches_file.load_matches(device=device)
    print(f'Total of {len(matches_data)} observations.')

    sucre = SUCRe(image=image, light_model=light_model, use_closed_form=use_closed_form).to(device)

    if params_path is not None:
        sucre.load_state_dict(torch.load(params_path), strict=False)

    adam(sucre=sucre, matches_data=matches_data, lr=lr, num_iter=num_iter,
         light_regularizer=light_regularizer, save_dir=output_dir, save_interval=save_interval)

    sucre.save_plots(save_dir=output_dir)
    torch.save({
        **sucre.cpu().state_dict(), 'J': sucre.J.detach().cpu()
    }, (output_dir / image.name).with_suffix('.pt'))

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
            output_dir=args.output_dir,
            light_model=args.light_model,
            use_closed_form=args.use_closed_form,
            min_cover=args.min_cover,
            image_list=image_list,
            lr=args.learning_rate,
            num_iter=args.num_iter,
            light_regularizer=args.light_regularizer,
            save_interval=args.save_interval,
            params_path=args.params_path,
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
                               help='range of ids of images to restore in the COLMAP model [min, max).')
    parser.add_argument('--light-model', action='store_true', help='model artificial lights.')
    parser.add_argument('--use-closed-form', action='store_true',
                        help='use the partial closed-form solution for computing the restored image from '
                             'absorption, backscatter and light parameters.')
    parser.add_argument('--light-regularizer', type=float, default=1e-4,
                        help='weight fixing light model gauge freedom (tz vs covariance scale).')
    parser.add_argument('--min-cover', type=float, default=0.000001,
                        help='minimum percentile of shared observations to keep the pairs of an image.')
    parser.add_argument('--image-scale', type=float, default=1.0,
                        help='rescale all images by this factor.')
    parser.add_argument('--filter-images-path', type=Path,
                        help='path to a .txt file with names of images to '
                             'discard when computing matches, one name per line.')
    parser.add_argument('--learning-rate', type=float, default=0.05,
                        help='learning rate for Adam optimizer.')
    parser.add_argument('--num-iter', type=int, default=200, help='number of optimization steps.')
    parser.add_argument('--save-interval', type=int,
                        help='save restored image every given optimization step.')
    parser.add_argument('--params-path', type=Path,
                        help='load underwater image formation model parameters from .pt file.')
    parser.add_argument('--force-compute-matches', action='store_true',
                        help='if matches file already exists, erase it and recompute matches.')
    parser.add_argument('--keep-matches', action='store_true',
                        help='keep matches file (can take a lot a space).')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of threads, 0 is the main thread.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device for heavy computation (`cpu` if cuda is not available).')

    parse_args(parser.parse_args())
