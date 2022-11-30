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

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import minimize

from sfm import COLMAPModel
from torch import Tensor


def compute_beta(z: Tensor, tau: float, phi: float, chi: float, omega: float) -> Tensor:
    """Compute non-linear attenuation coefficient according to the Sea-thru model

    :param z: distances, shape (n,).
    :param tau: tau parameter.
    :param phi: phi parameter.
    :param chi: chi parameter.
    :param omega: omega parameter.
    :return: beta, shape(n,).
    """
    return tau * torch.exp(-phi * z) + chi * torch.exp(-omega * z)


def compute_m(z: Tensor, B: float, beta: float | Tensor, gamma: float, mu: float) -> Tensor:
    """Compute m, distance-dependent mean of acquired image

    :param z: distances, shape (n,).
    :param B: veiling light.
    :param beta: attenuation coefficient, float or shape (n,).
    :param gamma: backscatter coefficient.
    :param mu: mean of restored image.
    :return: m, distance-dependent mean of acquired image, shape (n,).
    """
    return mu * torch.exp(-beta * z) + B * (1 - torch.exp(-gamma * z))


def compute_s(z: Tensor, beta: float | Tensor, sigma: float) -> Tensor:
    """Compute s, distance-dependent standard deviation of acquired image

    :param z: distances, shape (n,).
    :param beta: attenuation coefficient, float or shape (n,).
    :param sigma: standard deviation of restored image.
    :return: s, distance-dependent standard deviation of acquired image, shape (n,).
    """
    return sigma * torch.exp(-beta * z)


def compute_loss(I: Tensor, m: Tensor, s: Tensor) -> Tensor:
    """Compute the negative log-likelihood of observing the acquired image

    :param I: acquired image, shape (n,).
    :param m: distance-dependent mean of acquired image, shape (n,).
    :param s: distance-dependent standard deviation of acquired image, shape (n,).
    :return: the negative log-likelihood of observing the acquired image, scalar to minimize.
    """
    return torch.log(s * np.sqrt(2 * torch.pi)).mean() + torch.square((I - m) / s).mean() / 2


def compute_J(I: Tensor, z: Tensor, B: float, beta: float | Tensor, gamma: float) -> Tensor:
    """Compute J from acquired image and underwater image formation model parameters

    :param I: acquired image, shape (n,).
    :param z: distances, shape (n,).
    :param B: veiling light.
    :param beta: attenuation coefficient, float or shape (n,).
    :param gamma: backscatter coefficient.
    :return: J, restored image, shape (n,).
    """
    return (I - B * (1 - torch.exp(-gamma * z))) * torch.exp(beta * z)


def solve_linear(Ic: Tensor, z: Tensor) -> np.array:
    def residuals(x):
        Bc_hat, betac_hat, gammac_hat = x.tolist()

        # Compute Jc from estimated parameters
        Jc_hat = compute_J(Ic, z, Bc_hat, betac_hat, gammac_hat)

        # Compute mu and sigma from the estimation of Jc
        muc_hat = Jc_hat.mean()
        sigmac_hat = Jc_hat.std()

        # Compute m and s from all previously estimated parameters
        mc_hat = compute_m(z, Bc_hat, betac_hat, gammac_hat, muc_hat)
        sc_hat = compute_s(z, betac_hat, sigmac_hat)

        return compute_loss(Ic, mc_hat, sc_hat).item()

    parameters_init = [0.25, 0.1, 0.1]

    # Minimize the negative log likelihood using simplex algorithm
    parameters = minimize(
        residuals,
        np.array(parameters_init),
        method='Nelder-Mead',
        bounds=[(0, np.inf)] * len(parameters_init),
        options={'maxiter': 10000, 'disp': True}
    )
    return parameters.x


def solve_normal(Ic: Tensor, z: Tensor, linear_beta: bool = False) -> dict:
    """Solve the minimization problem with simplex algorithm

    :param Ic: acquired image, shape (n,).
    :param z: distances, shape (n,).
    :param linear_beta: whether to use Sea-thru model (False) or SUCRe model (True).
    :return: parameters of the model.
    """
    def residuals(x):
        if linear_beta:
            Bc_hat, betac_hat, gammac_hat = x.tolist()
        else:
            Bc_hat, tauc_hat, phic_hat, chic_hat, omegac_hat, gammac_hat = x.tolist()

            # If beta is non-linear, compute it
            betac_hat = compute_beta(z, tauc_hat, phic_hat, chic_hat, omegac_hat)

        # Compute Jc from estimated parameters
        Jc_hat = compute_J(Ic, z, Bc_hat, betac_hat, gammac_hat)

        # Compute mu and sigma from the estimation of Jc
        muc_hat = Jc_hat.mean()
        sigmac_hat = Jc_hat.std()

        # Compute m and s from all previously estimated parameters
        mc_hat = compute_m(z, Bc_hat, betac_hat, gammac_hat, muc_hat)
        sc_hat = compute_s(z, betac_hat, sigmac_hat)

        return compute_loss(Ic, mc_hat, sc_hat).item()

    if linear_beta:
        parameters_init = [0.25, 0.1, 0.1]  # Initial parameters if beta is linear
    else:
        parameters_init = [0.25, 0.1, 0.1, 0.1, 0.1, 0.1]  # Initial parameters if beta is not linear

    # Minimize the negative log likelihood using simplex algorithm
    parameters = minimize(
        residuals,
        np.array(parameters_init),
        method='Nelder-Mead',
        bounds=[(0, np.inf)] * len(parameters_init),
        options={'maxiter': 10000, 'disp': True}
    )

    print(f'Final cost: {parameters.fun}')

    if linear_beta:
        Bc, betac, gammac = parameters.x.tolist()
        return {
            'Bc': Bc,
            'betac': betac,
            'gammac': gammac
        }
    else:
        Bc, tauc, phic, chic, omegac, gammac = parameters.x.tolist()
        betac = compute_beta(z, tauc, phic, chic, omegac)
        return {
            'Bc': Bc,
            'tauc': tauc,
            'phic': phic,
            'chic': chic,
            'omegac': omegac,
            'betac': betac,
            'gammac': gammac
        }


def gaussian_seathru(
        colmap_model: COLMAPModel,
        image_name: str,
        output_dir: Path,
        linear_beta: bool = False,
        device: str = 'cpu'
):
    """Applies Gaussian Sea-thru to an image

    :param colmap_model: COLMAP model object.
    :param image_name: name of the image to be restored.
    :param output_dir: output directory.
    :param linear_beta: whether to use Sea-thru model (False) or SUCRe model (True).
    :param device: device on which to compute the function to minimize.
    """
    print(f'Restoring {image_name}.')

    colmap_image = colmap_model[image_name]

    # Load the image
    image = colmap_image.image().to(device)

    # Compute the distance map from the image's depth map and intrinsics
    distance_map = colmap_image.distance_map(colmap_image.depth_map().to(device))

    # Only select pixels where there is depth information
    args_valid = distance_map > 0
    image_valid = image[args_valid]
    distance_map_valid = distance_map[args_valid]

    params = []
    for c in range(3):  # For each color channel
        # Solve the minimization problem
        print(f'Minimizing cost function for {["red", "green", "blue"][c]} channel.')
        params.append(solve_normal(image_valid[:, c], distance_map_valid, linear_beta=linear_beta))

    J = np.zeros(image.shape)
    for c in range(3):  # For each channel
        #  Stretch the channel histogram
        Jc = compute_J(image_valid[:, c], distance_map_valid, params[c]['Bc'], params[c]['betac'],
                       params[c]['gammac']).cpu().numpy()
        Jc = np.clip(Jc, np.percentile(Jc, 1), np.percentile(Jc, 99))
        Jc = Jc - Jc.min()
        Jc = Jc / Jc.max()
        J[args_valid.cpu(), c] = Jc

    # Save the image
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(J * 255)).save((output_dir / image_name).with_suffix('.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Sea-thru.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', required=True, type=Path,
                        help='path to data directory with folders `depth_maps`, `images` and `sparse`.')
    parser.add_argument('--output-dir', required=True, type=Path, help='path to output directory.')
    parser.add_argument('--image-name', required=True, type=str, help='name of the image to be restored.')
    parser.add_argument('--linear-beta', action=argparse.BooleanOptionalAction, default=False,
                        help='switch between Sea-thru model (False) and SUCRe model (True).')
    parser.add_argument('--device', type=str, default='cpu', help='device for heavy computation.')
    args = parser.parse_args()

    print('Loading COLMAP model.')
    args_colmap = COLMAPModel(data_dir=args.data_dir, voxel_size=1.0)

    gaussian_seathru(
        colmap_model=args_colmap,
        image_name=args.image_name,
        output_dir=args.output_dir,
        linear_beta=args.linear_beta,
        device=args.device
    )
