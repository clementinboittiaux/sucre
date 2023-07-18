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

import h5py
import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor

import loader
import normalization
import sfm
import utils
import sucre


class VignettingData:
    def __init__(self):
        self.data: list[dict[str, Tensor]] = []

    def append(self, u: Tensor, v: Tensor, cP: Tensor, z: Tensor, I: Tensor):
        self.data.append({'u': u, 'v': v, 'z': z, 'I': I, 'cP': cP})

    def iterbatch(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        for i in range(0, len(self.data), batch_size):
            yield (
                torch.hstack([sample['u'] for sample in self.data[i:i + batch_size]]).long(),
                torch.hstack([sample['v'] for sample in self.data[i:i + batch_size]]).long(),
                torch.hstack([sample['z'] for sample in self.data[i:i + batch_size]]),
                torch.hstack([sample['I'] for sample in self.data[i:i + batch_size]]),
                torch.hstack([sample['cP'] for sample in self.data[i:i + batch_size]])
            )

    def __len__(self):
        return sum([sample['I'].shape[0] for sample in self.data])


def load_vignetting_data(matches_path: Path, colmap_model: sfm.COLMAPModel, device: str = 'cpu') -> VignettingData:
    data = VignettingData()
    with h5py.File(matches_path, 'r', libver='latest') as f:
        for match_name, group in f.items():
            z = torch.tensor(group['z'][()], device=device)
            u2 = torch.tensor(group['u2'][()], device=device) + 0.5
            v2 = torch.tensor(group['v2'][()], device=device) + 0.5
            cP = colmap_model[match_name].camera.K_inv.to(device) @ torch.vstack([u2, v2, torch.ones_like(u2)])
            cP = cP / cP.norm(dim=0) * z
            valid = z > 0
            data.append(
                u=torch.tensor(group['u1'][()], device=device)[valid],
                v=torch.tensor(group['v1'][()], device=device)[valid],
                z=z[valid],
                I=torch.tensor(group['I'][()], device=device)[:, valid],
                cP=cP[:, valid]
            )
    return data


def se3_exp(pose: Tensor) -> tuple[Tensor, Tensor]:
    w1, w2, w3, p1, p2, p3 = pose
    zero = torch.zeros_like(w1)
    pose = torch.stack([zero, -w3, w2, p1, w3, zero, -w1, p2, -w2, w1, zero, p3, zero, zero, zero, zero])
    pose = torch.matrix_exp(pose.view(4, 4))
    return pose[:3, :3], pose[:3, 3:4]


def deepsucre(
        colmap_model: sfm.COLMAPModel,
        image_name: str,
        output_dir: Path,
        min_cover: float,
        filter_image_names: list[str] = None,
        max_iter: int = 200,
        batch_size: int = 1,
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
    B = torch.full((3, 1), torch.nan, dtype=torch.float32)
    beta = torch.full((3, 1), torch.nan, dtype=torch.float32)
    gamma = torch.full((3, 1), torch.nan, dtype=torch.float32)
    for channel in range(3):
        J[:, :, channel], B[channel], beta[channel], gamma[channel] = sucre.initialize_sucre_parameters(
            data=matches_file.load_channel(channel, device=device),
            image=image,
            channel=channel,
            params_path=output_dir / 'global.yml',
            mode='global',
            device=device
        )
    J = torch.nn.Parameter(J.to(device))
    B = torch.nn.Parameter(B.to(device))
    beta = torch.nn.Parameter(beta.to(device))
    gamma = torch.nn.Parameter(gamma.to(device))
    s_T_c = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device))
    haloc = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))
    halox = torch.nn.Parameter(torch.tensor(-0.1, dtype=torch.float32, device=device))
    haloy = torch.nn.Parameter(torch.tensor(-0.1, dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam([J, B, beta, gamma, s_T_c, haloc, halox, haloy], lr=0.05)

    data = load_vignetting_data(matches_path, colmap_model, device=device)
    size = len(data)

    for iteration in range(max_iter):

        cost = 0
        optimizer.zero_grad()

        s_R_c, s_t_c = se3_exp(s_T_c)

        for ui, vi, zi, Ii, ciP in data.iterbatch(batch_size=batch_size):
            siP = s_R_c @ ciP + s_t_c
            sip = siP[:2] / siP[2]

            halo = haloc + halox * sip[0].square() + haloy * sip[1].square()

            zi = zi + siP.norm(dim=0)

            loss = torch.square(
                Ii - halo * (J[vi, ui].T * torch.exp(-beta * zi) + B * (1 - torch.exp(-gamma * zi)))
            ).sum()
            cost = cost + loss / size / 3

        cost.backward()
        optimizer.step()

        with np.printoptions(precision=2):
            print(
                f'iter: {iteration:04d}, cost: {cost.item():.3e}, B: {B.detach().flatten().cpu().numpy()}, '
                f'beta: {beta.detach().flatten().cpu().numpy()}, gamma: {gamma.detach().flatten().cpu().numpy()}, '
                f't: {s_t_c.detach().flatten().cpu().numpy()}, '
                f'halo: [{haloc.item():.2e}, {halox.item():.2e}, {haloy.item():.2e}]'
            )

    print('-' * len(f'Save restored image in {output_dir}.'))
    print(f'Save restored image in {output_dir}.')
    print('-' * len(f'Save restored image in {output_dir}.'))
    torch.save(J.detach().cpu(), (output_dir / image_name).with_suffix('.pt'))
    with open((output_dir / image_name).with_suffix('.yml'), 'w') as yaml_file:
        yaml.dump(
            {
                'B': B.flatten().tolist(),
                'beta': beta.flatten().tolist(),
                'gamma': gamma.flatten().tolist(),
                's_T_c': s_T_c.tolist(),
                'halo': [haloc.item(), halox.item(), haloy.item()]
            },
            yaml_file
        )
    Image.fromarray(
        np.uint8(normalization.histogram_stretching(J.detach().cpu()) * 255)
    ).save((output_dir / image_name).with_suffix('.png'))

    if not keep_matches:
        print(f'Erase {matches_path}.')
        matches_path.unlink()


def parse_args(args: argparse.Namespace):
    print('Loading COLMAP model.')
    colmap_model = sfm.COLMAPModel(model_dir=args.model_dir, image_dir=args.image_dir, depth_dir=args.depth_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
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

    utils.estimate_global_parameters(
        image_list=list(colmap_model.images.values()), output_dir=args.output_dir, device=args.device
    )

    for image_name in image_names:
        deepsucre(
            colmap_model=colmap_model,
            image_name=image_name,
            output_dir=args.output_dir,
            min_cover=args.min_cover,
            filter_image_names=filter_image_names,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
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
    parser.add_argument('--max-iter', type=int, default=1000, help='maximum number of optimization steps.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size for adam optimization, higher is faster but requires more RAM.')
    parser.add_argument('--force-compute-matches', action='store_true',
                        help='if matches file already exist, erase it and recompute matches.')
    parser.add_argument('--keep-matches', action='store_true', help='keep matches file (can take a lot a space).')
    parser.add_argument('--num-workers', type=int, default=0, help='number of threads, 0 is the main thread.')
    parser.add_argument('--device', type=str, default='cpu', help='device for heavy computation.')

    parse_args(parser.parse_args())
