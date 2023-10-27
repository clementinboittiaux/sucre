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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.optimize import minimize
from torch import Tensor

import loader
import se3
import sfm


class SUCRe(torch.nn.Module):
    def __init__(
            self,
            image: sfm.Image,
            B: tuple[float, float, float] = (0.1, 0.1, 0.1),
            beta: tuple[float, float, float] = (0.1, 0.1, 0.1),
            gamma: tuple[float, float, float] = (0.1, 0.1, 0.1),
            cam2light: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            sigma: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
            light_model: bool = False,
            matches_data: loader.MatchesData = None,
            device: str = 'cpu'
    ):
        super().__init__()
        self.image = image
        self.light_model = light_model
        self.B = torch.nn.Parameter(torch.tensor(B, dtype=torch.float32, device=device).view(3, 1))
        self.beta = torch.nn.Parameter(torch.tensor(beta, dtype=torch.float32, device=device).view(3, 1))
        self.gamma = torch.nn.Parameter(torch.tensor(gamma, dtype=torch.float32, device=device).view(3, 1))
        if self.light_model:
            self.cam2light = torch.nn.Parameter(torch.tensor(cam2light, dtype=torch.float32, device=device))
            self.sigma = torch.nn.Parameter(torch.tensor(sigma, dtype=torch.float32, device=device).view(2, 2))
        with torch.no_grad():
            if matches_data is not None:
                self.J = torch.nn.Parameter(self.compute_J(matches_data))
            else:
                self.J = torch.nn.Parameter(image.get_rgb().to(device))
                self.J[image.get_depth_map().to(device) <= 0] = torch.nan

    def compute_l_z(self, cP: Tensor) -> tuple[float | Tensor, Tensor]:
        z = cP.norm(dim=0)
        if self.light_model:
            R, t = se3.exp(self.cam2light)
            Sigma = self.sigma.T @ self.sigma
            lP = R @ cP + t
            lp = lP[:2] / lP[2]
            lp = lp.T.unsqueeze(dim=2)
            l = torch.exp(-torch.flatten(lp.transpose(1, 2) @ Sigma.inverse() @ lp) / 2)
            z = z + lP.norm(dim=0)
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
        image: sfm.Image,
        matches_data: loader.MatchesData,
        light_model: bool = False,
        num_iter: int = 200,
        batch_size: int = 1,
        save_dir: Path = None,
        save_interval: int = None,
        device: str = 'cpu'
) -> SUCRe:
    print(f'Solve least squares with Adam optimizer ({num_iter} iterations).')
    sucre = SUCRe(image=image, light_model=light_model, device=device)

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
    return sucre


def simplex(
        image: sfm.Image,
        matches_data: loader.MatchesData,
        light_model: bool = False,
        max_iter: int = 200,
        batch_size: int = 1,
        device: str = 'cpu'
) -> SUCRe:
    print(f'Solve least squares with Nelder-Mead simplex algorithm ({max_iter} maximum iterations).')
    n_obs = len(matches_data)

    def build_sucre(x: list) -> SUCRe:
        B_r, B_g, B_b, beta_r, beta_g, beta_b, gamma_r, gamma_g, gamma_b = x[:9]
        cam2light, sigma = None, None
        if light_model:
            w1, w2, w3, p1, p2, p3, sigma11, sigma12, sigma21, sigma22 = x[9:]
            cam2light = (w1, w2, w3, p1, p2, p3)
            sigma = (sigma11, sigma12, sigma21, sigma22)
        sucre_hat = SUCRe(
            image=image,
            B=(B_r, B_g, B_b),
            beta=(beta_r, beta_g, beta_b),
            gamma=(gamma_r, gamma_g, gamma_b),
            cam2light=cam2light,
            sigma=sigma,
            light_model=light_model,
            matches_data=matches_data,
            device=device
        )
        return sucre_hat

    def residuals(x: np.array) -> float:
        sucre_hat = build_sucre(x.tolist())
        cost = 0.0
        for u, v, cP, I in matches_data.iter(batch_size=batch_size, device=device):
            cost += torch.square(I - sucre_hat(u=u, v=v, cP=cP)).sum().item() / n_obs / 3
        return cost

    x0 = [0.1] * 9  # initial B, beta and gamma parameters for all color channels
    bounds = [(1e-6, 5)] * 9  # bounds for B, beta and gamma parameters
    if light_model:
        x0 += [0.0] * 6  # initial se(3) camera to light pose
        x0 += [1.0, 0.0, 0.0, 1.0]  # initial light pattern
        bounds += [(-np.pi, np.pi)] * 3  # bounds for so(3) camera rotation
        bounds += [(-100, 100)] * 3  # bounds for R3 camera translation
        bounds += [(-100, 100)] * 4  # bounds for light pattern

    with torch.no_grad():
        sucre = build_sucre(minimize(
            residuals,
            np.array(x0),
            method='BFGS',
            # bounds=bounds,
            options={'disp': True}
        ).x.tolist())
    return sucre


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

    match solver:
        case 'adam':
            sucre = adam(image=image, matches_data=matches_data, light_model=light_model, num_iter=max_iter,
                         batch_size=batch_size, save_dir=output_dir, save_interval=save_interval, device=device)
        case 'simplex':
            sucre = simplex(image=image, matches_data=matches_data, light_model=light_model,
                            max_iter=max_iter, batch_size=batch_size, device=device)
        case _:
            raise ValueError('Currently, only `adam` and `simplex` optimizer are supported.')

    sucre.plot_J().save((output_dir / image.name).with_suffix('.png'))
    torch.save(sucre.cpu().state_dict(), (output_dir / image.name).with_suffix('.pt'))

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
                        help='save restored image every given optimization step.')
    parser.add_argument('--force-compute-matches', action='store_true',
                        help='if matches file already exists, erase it and recompute matches.')
    parser.add_argument('--keep-matches', action='store_true',
                        help='keep matches file (can take a lot a space).')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of threads, 0 is the main thread.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device for heavy computation (`cpu` if cuda is not available).')

    parse_args(parser.parse_args())
