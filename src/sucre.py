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
import yaml
from PIL import Image
from scipy.optimize import minimize
from torch import Tensor

import gaussian_seathru
import loader
import normalization
import sfm


def initialize_sucre_parameters(
        data: loader.Data,
        image: sfm.Image,
        channel: int,
        mode: str = 'single-view',
        device: str = 'cpu'
) -> tuple[Tensor, float, float, float]:
    print(f'Initialize parameters with Gaussian Sea-thru ({mode} mode).')
    match mode:
        case 'single-view':
            Ic = loader.load_image(image.image_path)[:, :, channel].to(device)
            z = image.distance_map(loader.load_depth(image.depth_path).to(device))
            args_valid = z > 0
            Bc, betac, gammac = gaussian_seathru.solve_gaussian_seathru(Ic[args_valid], z[args_valid], linear_beta=True)
            Jc = torch.full((image.camera.height, image.camera.width), torch.nan, dtype=torch.float32, device=device)
            Jc[args_valid] = gaussian_seathru.compute_J(Ic[args_valid], z[args_valid], Bc, betac, gammac)
        case 'multi-view':
            Bc, betac, gammac = gaussian_seathru.solve_gaussian_seathru(
                *data.to_Ic_z(device=device), linear_beta=True
            )
            Bc, Jc = compute_Bc_Jc(image=image, data=data, betac=betac, gammac=gammac, device=device)
            Bc = Bc.item()
        case _:
            raise ValueError("`mode` only supports 'single-view' and 'multi-view'.")
    return Jc.cpu(), Bc, betac, gammac


def compute_Bc_Jc(
        image: sfm.Image,
        data: loader.Data,
        betac: Tensor | float,
        gammac: Tensor | float,
        device: str = 'cpu'
) -> tuple[Tensor, Tensor]:
    Xc_numerator = torch.zeros(image.camera.height, image.camera.width, dtype=torch.float32, device=device)
    Yc_numerator = torch.zeros(image.camera.height, image.camera.width, dtype=torch.float32, device=device)
    XYc_denominator = torch.zeros(image.camera.height, image.camera.width, dtype=torch.float32, device=device)
    Bc_numerator = torch.zeros(image.camera.height, image.camera.width, dtype=torch.float32, device=device)
    Bc_denominator = torch.zeros(image.camera.height, image.camera.width, dtype=torch.float32, device=device)

    for ui, vi, zi, Iic in data.iter(device=device):
        aic = torch.exp(-betac * zi)
        bic = 1 - torch.exp(-gammac * zi)
        Xc_numerator[vi, ui] += Iic * aic
        Yc_numerator[vi, ui] += bic * aic
        XYc_denominator[vi, ui] += torch.square(aic)

    Xc = Xc_numerator / XYc_denominator
    Yc = Yc_numerator / XYc_denominator

    for ui, vi, zi, Iic in data.iter(device=device):
        aic = torch.exp(-betac * zi)
        bic = 1 - torch.exp(-gammac * zi)
        Mic = Iic - Xc[vi, ui] * aic
        Nic = bic - Yc[vi, ui] * aic
        Bc_numerator[vi, ui] += Mic * Nic
        Bc_denominator[vi, ui] += torch.square(Nic)

    Bc = Bc_numerator.sum() / Bc_denominator.sum()
    Jc = Xc - Bc * Yc
    return Bc, Jc


def levenberg_marquardt(
        image: sfm.Image,
        data: loader.Data,
        betac_init: float,
        gammac_init: float,
        max_iter: int = 200,
        function_tolerance: float = 1e-5,
        device: str = 'cpu'
) -> tuple[Tensor, float, float, float]:
    print(f'Optimize Jc, Bc, betac and gammac in a Levenberg-Marquardt scheme ({max_iter} maximum iterations).')
    betac = torch.tensor(betac_init, dtype=torch.float32, device=device)
    gammac = torch.tensor(gammac_init, dtype=torch.float32, device=device)
    residuals = torch.zeros(len(data), dtype=torch.float32, device=device)
    jacobian = torch.zeros(len(data), 2, dtype=torch.float32, device=device)

    damping = 0.1
    eye = torch.eye(2, dtype=torch.float32, device=device)
    previous_cost = torch.inf

    for iteration in range(max_iter):

        Bc, Jc = compute_Bc_Jc(image=image, data=data, betac=betac, gammac=gammac, device=device)

        cursor = 0
        for ui, vi, zi, Iic in data.iter(device=device):
            length = zi.shape[0]
            residuals[cursor: cursor + length] = (
                    Iic - Jc[vi, ui] * torch.exp(-betac * zi) - Bc * (1 - torch.exp(-gammac * zi))
            )
            jacobian[cursor: cursor + length, 0] = (zi * Jc[vi, ui] * torch.exp(-betac * zi))
            jacobian[cursor: cursor + length, 1] = (-zi * Bc * torch.exp(-gammac * zi))
            cursor += length

        delta = torch.inverse(jacobian.T @ jacobian + damping * eye) @ (jacobian.T @ residuals)
        betac -= delta[0]
        gammac -= delta[1]

        cost = torch.square(residuals).sum()

        cost_change = torch.abs(previous_cost - cost) / cost
        print(f'iter: {iteration:04d}, cost: {cost.item():.8e}, cost change: {cost_change.item():.8e}, damping: '
              f'{damping:.8e}, Bc: {Bc.item():.3e}, betac: {betac.item():.3e}, gammac: {gammac.item():.3e}')
        if cost_change < function_tolerance or cost.isnan():
            break
        elif cost < previous_cost:
            damping = max(damping / 10, 1e-32)
        else:
            damping *= 10
        previous_cost = cost

    Bc, Jc = compute_Bc_Jc(image=image, data=data, betac=betac, gammac=gammac, device=device)
    return Jc.cpu(), Bc.item(), betac.item(), gammac.item()


def simplex(
        image: sfm.Image,
        data: loader.Data,
        betac_init: float,
        gammac_init: float,
        max_iter: int = 200,
        device: str = 'cpu'
) -> tuple[Tensor, float, float, float]:
    print(f'Optimize Jc, Bc, betac and gammac with Nelder-Mead simplex algorithm ({max_iter} maximum iterations).')

    def residuals(x):
        betac_hat, gammac_hat = x.tolist()
        Bc_hat, Jc_hat = compute_Bc_Jc(image=image, data=data, betac=betac_hat, gammac=gammac_hat, device=device)
        cost = torch.zeros(1, dtype=torch.float32, device=device)
        for ui, vi, zi, Iic in data.iter(device=device):
            cost += torch.square(
                Iic - Jc_hat[vi, ui] * torch.exp(-betac_hat * zi) - Bc_hat * (1 - torch.exp(-gammac_hat * zi))
            ).sum()
        return cost.item()

    betac, gammac = minimize(
        residuals,
        np.array([betac_init, gammac_init]),
        method='Nelder-Mead',
        bounds=[(1e-8, 5), (1e-8, 5)],
        options={'maxiter': max_iter, 'disp': True}
    ).x.tolist()
    Bc, Jc = compute_Bc_Jc(image=image, data=data, betac=betac, gammac=gammac, device=device)

    return Jc.cpu(), Bc.item(), betac, gammac


def adam(
        data: loader.Data,
        Jc_init: Tensor,
        Bc_init: float,
        betac_init: float,
        gammac_init: float,
        num_iter: int = 200,
        device: str = 'cpu'
) -> tuple[Tensor, float, float, float]:
    print(f'Optimize Jc, Bc, betac and gammac with Adam optimizer ({num_iter} iterations).')
    Jc = torch.nn.Parameter(Jc_init.to(device))
    Bc = torch.nn.Parameter(torch.tensor(Bc_init, dtype=torch.float32, device=device))
    betac = torch.nn.Parameter(torch.tensor(betac_init, dtype=torch.float32, device=device))
    gammac = torch.nn.Parameter(torch.tensor(gammac_init, dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam([Jc, Bc, betac, gammac], lr=0.01)

    size = len(data)
    previous_cost = np.inf

    for iteration in range(num_iter):

        cost = 0
        optimizer.zero_grad()

        for ui, vi, zi, Iic in data.iter(device=device):
            loss = torch.square(
                Iic - Jc[vi, ui] * torch.exp(-betac * zi) - Bc * (1 - torch.exp(-gammac * zi))
            ).sum()
            (loss / size).backward()
            cost += loss.item()

        optimizer.step()
        cost_change = np.abs(previous_cost - cost) / cost

        print(f'iter: {iteration:04d}, cost: {cost:.8e}, cost change: {cost_change:.8e}, '
              f'Bc: {Bc.item():.3e}, betac: {betac.item():.3e}, gammac: {gammac.item():.3e}')
        previous_cost = cost

    return Jc.detach().cpu(), Bc.item(), betac.item(), gammac.item()


def solve_sucre(
        image: sfm.Image,
        channel: int,
        matches_file: loader.MatchesFile,
        init: str = 'single-view',
        solver: str = 'lm',
        max_iter: int = 200,
        function_tolerance: float = 1e-5,
        outliers: str = 'single-view',
        device: str = 'cpu'
) -> tuple[Tensor, float, float, float, float]:
    data = matches_file.load_channel(channel)
    Jc_init, Bc_init, betac_init, gammac_init = initialize_sucre_parameters(
        data=data, image=image, channel=channel, mode=init, device=device
    )
    print(f'Bc: {Bc_init}, betac: {betac_init}, gammac: {gammac_init}')

    diverged = False

    if max_iter > 0:
        match solver:
            case 'lm':
                Jc, Bc, betac, gammac = levenberg_marquardt(
                    image=image,
                    data=data,
                    betac_init=betac_init,
                    gammac_init=gammac_init,
                    max_iter=max_iter,
                    function_tolerance=function_tolerance,
                    device=device
                )
            case 'simplex':
                Jc, Bc, betac, gammac = simplex(
                    image=image,
                    data=data,
                    betac_init=betac_init,
                    gammac_init=gammac_init,
                    max_iter=max_iter,
                    device=device
                )
            case 'adam':
                Jc, Bc, betac, gammac = adam(
                    data=data,
                    Jc_init=Jc_init,
                    Bc_init=Bc_init,
                    betac_init=betac_init,
                    gammac_init=gammac_init,
                    num_iter=max_iter,
                    device=device
                )
            case _:
                raise ValueError("`method` supports 'lm', 'simplex' or 'adam'")

        if any([np.isnan(Bc), np.isnan(betac), np.isnan(gammac)]):
            print('WARNING: optimization diverged, revert to initial parameters.')
            Jc, Bc, betac, gammac = Jc_init, Bc_init, betac_init, gammac_init
            diverged = True
    else:
        Jc, Bc, betac, gammac = Jc_init, Bc_init, betac_init, gammac_init

    if init == 'single-view' and (diverged or max_iter <= 0):
        Bc, Jc = compute_Bc_Jc(image=image, data=data, betac=betac, gammac=gammac, device=device)
        Bc = Bc.item()
        Jc = Jc.cpu()

    print(f'Bc: {Bc}, betac: {betac}, gammac: {gammac}')

    Jcmin = normalization.estimate_Jcmin_zdcp(
        data=data, image=image, channel=channel, Bc=Bc, betac=betac, gammac=gammac, mode=outliers, device=device
    )

    return Jc, Bc, betac, gammac, Jcmin


def sucre(
        colmap_model: sfm.COLMAPModel,
        image_name: str,
        output_dir: Path,
        min_cover: float,
        filter_image_names: list[str] = None,
        init: str = 'single-view',
        solver: str = 'lm',
        max_iter: int = 200,
        function_tolerance: float = 1e-5,
        outliers: str = 'single-view',
        force_compute_matches: bool = False,
        keep_matches: bool = False,
        num_workers: int = 0,
        device: str = 'cpu'
):
    print(f'Restore {image_name}.')
    image = colmap_model[image_name]
    image_list = list(colmap_model.images.values())

    # Filter images that should not be used for pairing
    if filter_image_names is not None:
        image_list = [im for im in image_list if im.name not in filter_image_names]

    matches_path = (output_dir / image_name).with_suffix('.h5')
    matches_file = loader.MatchesFile(matches_path, overwrite=force_compute_matches)

    if force_compute_matches or not matches_path.exists():
        print(f'Compute {image_name} matches.')
        # TODO: filter matches by distance
        image.match_images(
            image_list=image_list,
            matches_file=matches_file,
            min_cover=min_cover,
            num_workers=num_workers,
            device=device
        )
        print('Prepare matches for optimization.')
        matches_file.prepare_matches(colmap_model=colmap_model, num_workers=num_workers, device=device)

    print(f'Total of {len(matches_file)} observations.')

    J = torch.full((image.camera.height, image.camera.width, 3), torch.nan, dtype=torch.float32)
    B = torch.full((3,), torch.nan, dtype=torch.float32)
    beta = torch.full((3,), torch.nan, dtype=torch.float32)
    gamma = torch.full((3,), torch.nan, dtype=torch.float32)
    Jmin = torch.full((3,), torch.nan, dtype=torch.float32)
    for channel in range(3):
        print(f'----------------------{["---", "-----", "----"][channel]}---------')
        print(f'SUCRe optimization on {["red", "green", "blue"][channel]} channel.')
        print(f'----------------------{["---", "-----", "----"][channel]}---------')
        J[:, :, channel], B[channel], beta[channel], gamma[channel], Jmin[channel] = solve_sucre(
            image=image,
            channel=channel,
            matches_file=matches_file,
            init=init,
            solver=solver,
            max_iter=max_iter,
            function_tolerance=function_tolerance,
            outliers=outliers,
            device=device
        )

    print('-' * len(f'Save restored image in {output_dir}.'))
    print(f'Save restored image in {output_dir}.')
    print('-' * len(f'Save restored image in {output_dir}.'))
    torch.save(J, (output_dir / image_name).with_suffix('.pt'))
    with open((output_dir / image_name).with_suffix('.yml'), 'w') as yaml_file:
        yaml.dump(
            {'B': B.tolist(), 'beta': beta.tolist(), 'gamma': gamma.tolist(), 'Jmin': Jmin.tolist()},
            yaml_file
        )
    Image.fromarray(
        np.uint8(normalization.histogram_stretching(
            normalization.filter_min_outliers(image=J, thresholds=Jmin)
        ) * 255)
    ).save((output_dir / image_name).with_suffix('.png'))

    if not keep_matches:
        print(f'Erase {matches_path}.')
        matches_path.unlink()


def parse_args(args: argparse.Namespace):
    print('Loading COLMAP model.')
    colmap_model = sfm.COLMAPModel(model_dir=args.model_dir, image_dir=args.image_dir, depth_dir=args.depth_dir)

    filter_image_names = args.filter_images_path.read_text().splitlines() if args.filter_images_path else None

    if args.image_name is not None:
        image_names = [args.image_name]
    elif args.image_list is not None:
        image_names = args.image_list.read_text().splitlines()
    else:
        image_names = []
        for image_id in range(args.image_ids[0], args.image_ids[1] + 1):
            if image_id in colmap_model.images:
                image_names.append(colmap_model.images[image_id].name)

    for image_name in image_names:
        sucre(
            colmap_model=colmap_model,
            image_name=image_name,
            output_dir=args.output_dir,
            min_cover=args.min_cover,
            filter_image_names=filter_image_names,
            init=args.initialization,
            solver=args.solver,
            max_iter=args.max_iter,
            function_tolerance=args.function_tolerance,
            outliers=args.outliers,
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
    parser.add_argument('--initialization', type=str, choices=['single-view', 'multi-view'], default='single-view',
                        help='initialize parameters with Gaussian Sea-thru on a single image or all matches.')
    parser.add_argument('--solver', type=str, choices=['lm', 'simplex', 'adam'],
                        default='lm', help='method to solve SUCRe least squares.')
    parser.add_argument('--max-iter', type=int, default=200, help='maximum number of optimization steps.')
    parser.add_argument('--function-tolerance', type=float, default=1e-5,
                        help='stops optimization if cost function change is below this threshold.')
    parser.add_argument('--outliers', type=str, choices=['single-view', 'multi-view', 'none'], default='single-view',
                        help='estimate expected minimum values of restored image to filter outliers.')
    parser.add_argument('--force-compute-matches', action='store_true',
                        help='if matches file already exist, erase it and recompute matches.')
    parser.add_argument('--keep-matches', action='store_true', help='keep matches file (can take a lot a space).')
    parser.add_argument('--num-workers', type=int, default=0, help='number of threads, 0 is the main thread.')
    parser.add_argument('--device', type=str, default='cpu', help='device for heavy computation.')

    parse_args(parser.parse_args())
