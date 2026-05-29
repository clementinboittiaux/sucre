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
import json
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

# Background fill (0-1 grayscale) for image pixels that carry no information from the
# target image, used by the saved reconstruction / illumination / point-cloud images.
BACKGROUND_GRAY = 0.8


class SUCRe(torch.nn.Module):
    def __init__(self, image: sfm.Image, light_model: bool = False, use_closed_form: bool = False):
        super().__init__()
        self.image = image
        self.light_model = light_model
        self.use_closed_form = use_closed_form
        self.B = torch.nn.Parameter(torch.tensor([[0.1], [0.1], [0.1]]).log())
        self.beta = torch.nn.Parameter(torch.tensor([[0.1], [0.1], [0.1]]).log())
        self.gamma = torch.nn.Parameter(torch.tensor([[0.1], [0.1], [0.1]]).log())
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
            absorption = l * torch.exp(-self.beta.exp() * z)
            backscatter = l * self.B.exp() * (1 - torch.exp(-self.gamma.exp() * z))
            J_numerator = torch.zeros((self.image.camera.height, self.image.camera.width, 3), device=self.B.device)
            J_denominator = torch.zeros((self.image.camera.height, self.image.camera.width, 3), device=self.B.device)
            J_index = (matches_data.v, matches_data.u)
            J_numerator.index_put_(J_index, ((matches_data.I - backscatter) * absorption).T, accumulate=True)
            J_denominator.index_put_(J_index, absorption.square().T, accumulate=True)
            self.J = J_numerator / J_denominator

    def forward(self, u: Tensor, v: Tensor, l: float | Tensor, z: Tensor) -> Tensor:
        I_hat = l * (self.J[v, u].T * torch.exp(-self.beta.exp() * z) + self.B.exp() * (1 - torch.exp(-self.gamma.exp() * z)))
        return I_hat

    @torch.no_grad()
    def plot_J(self):
        J = self.J.cpu().numpy().copy()
        valid = np.all(~np.isnan(J), axis=2)
        J_valid = J[valid]
        J_valid = np.clip(J_valid, np.percentile(J_valid, 1, axis=0), np.percentile(J_valid, 99, axis=0))
        J_valid = J_valid - np.min(J_valid, axis=0)
        J_valid = J_valid / np.max(J_valid, axis=0)
        J[~valid] = BACKGROUND_GRAY
        J[valid] = J_valid
        return Image.fromarray(np.uint8(J * 255))

    @torch.no_grad()
    def _view_samples(self, image: sfm.Image, depth_map: Tensor):
        """Per-pixel data for rendering `image`: pixel coords (u, v), light l, path length z,
        restored radiance J, and a valid mask.

        J is sampled from the target's restored radiance (read directly for the target image,
        reprojected for any other image) and is NaN wherever the point carries no information
        from the target image; `valid` is the corresponding ~isnan(J) mask.
        """
        u, v, cP = image.unproject_depth_map(depth_map.to(self.B.device), to_world=False)
        l, z = self.compute_l_z(cP)
        if image is self.image:
            J = self.J[v, u]
        else:
            uj, vj = self.image.project_to_view(image.pose.transform(cP)).long()
            inside = (uj >= 0) & (uj < self.image.camera.width) & (vj >= 0) & (vj < self.image.camera.height)
            J = self.J[vj.clamp(0, self.image.camera.height - 1), uj.clamp(0, self.image.camera.width - 1)]
            J[~inside] = torch.nan
        valid = ~torch.isnan(J).any(dim=1)
        return u, v, l, z, J, valid

    @torch.no_grad()
    def plot_l(self, image: sfm.Image = None, depth_map: Tensor = None):
        """Render the projected illumination pattern as seen from `image`.

        Like the reconstructed image, pixels that carry no information from the target
        image are filled with the gray background; the jet colormap is applied only to
        the valid pixels.
        """
        if image is None:
            image = self.image
        if depth_map is None:
            depth_map = image.get_depth_map()
        u, v, l, z, J, valid = self._view_samples(image, depth_map)
        l_map = torch.zeros((image.camera.height, image.camera.width), device=self.B.device)
        l_map[v[valid], u[valid]] = l if isinstance(l, float) else l[valid]
        rgb = plt.colormaps['jet'](l_map.cpu().numpy())[:, :, :3]
        mask = torch.zeros((image.camera.height, image.camera.width), dtype=torch.bool, device=self.B.device)
        mask[v[valid], u[valid]] = True
        rgb[~mask.cpu().numpy()] = BACKGROUND_GRAY  # gray where no information from the target
        return Image.fromarray(np.uint8(rgb * 255))

    @torch.no_grad()
    def plot_light_pattern(self, scale: float = 1.0):
        """Render the light's illumination pattern in the light image plane.

        The pattern is the 2D Gaussian l(lp) = exp(-lp^T Sigma^-1 lp / 2), sampled over the
        target camera's field of view (its normalized image coordinates).
        """
        K = self.image.camera.K.to(self.sigma.device)
        width = round(self.image.camera.width * scale)
        height = round(self.image.camera.height * scale)
        u = (torch.arange(width, device=self.sigma.device) + 0.5) * (self.image.camera.width / width)
        v = (torch.arange(height, device=self.sigma.device) + 0.5) * (self.image.camera.height / height)
        gy, gx = torch.meshgrid((v - K[1, 2]) / K[1, 1], (u - K[0, 2]) / K[0, 0], indexing='ij')
        lp = torch.stack([gx.flatten(), gy.flatten()])
        Sigma = self.sigma.T @ self.sigma
        l = torch.exp(-(lp * (Sigma.inverse() @ lp)).sum(dim=0) / 2).reshape(height, width)
        return Image.fromarray(np.uint8(plt.colormaps['jet'](l.cpu().numpy())[:, :, :3] * 255))

    @torch.no_grad()
    def reconstruct_view(self, image: sfm.Image, depth_map: Tensor = None):
        """Render the predicted underwater image (I_hat) as seen from `image`.

        For the target image, the restored radiance J is read directly. For any other image,
        J is sampled by reprojecting the depth-map points of `image` into the target image.
        """
        if depth_map is None:
            depth_map = image.get_depth_map()
        u, v, l, z, J, valid = self._view_samples(image, depth_map)
        l = l if isinstance(l, float) else l[valid]
        z = z[valid]
        I_hat = l * (J[valid].T * torch.exp(-self.beta.exp() * z) + self.B.exp() * (1 - torch.exp(-self.gamma.exp() * z)))
        I_reconstructed = torch.full((image.camera.height, image.camera.width, 3), BACKGROUND_GRAY,
                                     device=self.B.device)
        I_reconstructed[v[valid], u[valid]] = I_hat.clip(0, 1).T
        return Image.fromarray(np.uint8(I_reconstructed.cpu().numpy() * 255))

    @torch.no_grad()
    def plot_reconstruction(self):
        return self.reconstruct_view(self.image)

    def save_plots(self, save_dir: Path, iteration: int = None):
        save_path = (save_dir / self.image.name).with_suffix('.png')
        suffix = '' if iteration is None else f'_{iteration:04d}'
        self.plot_J().save(save_path.with_stem(f'{save_path.stem}_rgb{suffix}'))
        self.plot_reconstruction().save(save_path.with_stem(f'{save_path.stem}_reconstruction{suffix}'))
        if self.light_model:
            self.plot_l().save(save_path.with_stem(f'{save_path.stem}_vignetting{suffix}'))

    @torch.no_grad()
    def save_all_views(self, save_dir: Path, image_list: list[sfm.Image], iteration: int):
        """Save the reconstructed image and illumination pattern from every viewpoint in `image_list`.

        Files are saved as JPEG (quality 95, gray where there is no information) under
        `save_dir/<target_stem>_views/<view_stem>/` at the loaded image resolution.
        """
        views_dir = save_dir / f'{Path(self.image.name).stem}_views'
        for image in tqdm(image_list, desc=f'Save views {iteration:04d}', leave=False):
            view_dir = views_dir / Path(image.name).stem
            view_dir.mkdir(parents=True, exist_ok=True)
            depth_map = image.get_depth_map()
            plots = {'reconstruction': self.reconstruct_view(image, depth_map)}
            if self.light_model:
                plots['vignetting'] = self.plot_l(image, depth_map)
            for name, plot in plots.items():
                plot.save(view_dir / f'{name}_{iteration:04d}.jpg', quality=95)

    @torch.no_grad()
    def light_pose(self) -> Tensor:
        """Return the 4x4 light-to-camera transform (computer-vision convention)."""
        R, t = se3.exp(self.cam2light)
        pose = torch.eye(4)
        pose[:3, :3] = R.T.cpu()
        pose[:3, 3] = (-R.T @ t).flatten().cpu()
        return pose

    @torch.no_grad()
    def save_light(self, save_dir: Path, iteration: int):
        """Save the light's illumination pattern image for one optimization step."""
        light_dir = save_dir / f'{Path(self.image.name).stem}_light'
        light_dir.mkdir(parents=True, exist_ok=True)
        self.plot_light_pattern().save(light_dir / f'pattern_{iteration:04d}.jpg', quality=95)

    @torch.no_grad()
    def save_point_cloud_geometry(self, save_dir: Path):
        """Write the fixed point-cloud geometry once.

        Each target pixel with positive depth is unprojected to a 3D world point; the (u, v)
        texture coordinate into the restored RGB image is stored alongside it. Positions do
        not change across optimization steps (only the restored RGB color does), so this is
        written a single time and Blender animates only the color.
        """
        pc_dir = save_dir / f'{Path(self.image.name).stem}_pointcloud'
        pc_dir.mkdir(parents=True, exist_ok=True)
        depth_map = self.image.get_depth_map().to(self.B.device)
        u, v, wP = self.image.unproject_depth_map(depth_map, to_world=True)
        width, height = self.image.camera.width, self.image.camera.height
        xyz = wP.T.cpu().numpy()
        uv = torch.stack([(u + 0.5) / width, 1.0 - (v + 0.5) / height], dim=1).cpu().numpy()
        np.savez(pc_dir / 'geometry.npz', xyz=xyz, uv=uv)

    @torch.no_grad()
    def save_point_cloud(self, save_dir: Path, iteration: int):
        """Save the restored RGB image for one optimization step.

        This is the per-step color of the unprojected point cloud; its geometry is written
        once by :meth:`save_point_cloud_geometry`.
        """
        pc_dir = save_dir / f'{Path(self.image.name).stem}_pointcloud'
        pc_dir.mkdir(parents=True, exist_ok=True)
        self.plot_J().save(pc_dir / f'rgb_{iteration:04d}.jpg', quality=95)


def adam(
        sucre: SUCRe,
        matches_data: loader.MatchesData,
        lr: float = 0.05,
        num_iter: int = 200,
        light_regularizer: float = 1e-4,
        save_dir: Path = None,
        save_interval: int = None,
        save_views: bool = False,
        save_light: bool = False,
        save_point_cloud: bool = False,
        view_image_list: list[sfm.Image] = None,
) -> SUCRe:
    print(f'Solve least squares with Adam optimizer ({num_iter} iterations).')
    n_obs = len(matches_data)
    optimizer = torch.optim.Adam(sucre.parameters(), lr=lr)
    export_light = (save_views or save_light) and sucre.light_model
    light_poses = {}

    if save_dir is not None and save_point_cloud:
        sucre.save_point_cloud_geometry(save_dir=save_dir)

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
                       f'B: {sucre.B.detach().exp().cpu().flatten().numpy()}, '
                       f'beta: {sucre.beta.detach().exp().cpu().flatten().numpy()}, '
                       f'gamma: {sucre.gamma.detach().exp().cpu().flatten().numpy()}, '
                       f't: {se3.exp(sucre.cam2light.detach())[1][:, 0].cpu().flatten().numpy()}')
        if save_dir is not None and save_interval is not None and iteration % save_interval == 0:
            sucre.save_plots(save_dir=save_dir, iteration=iteration)
            if save_views:
                sucre.save_all_views(save_dir=save_dir, image_list=view_image_list, iteration=iteration)
            if export_light:
                sucre.save_light(save_dir=save_dir, iteration=iteration)
                light_poses[iteration] = sucre.light_pose().tolist()
            if save_point_cloud:
                sucre.save_point_cloud(save_dir=save_dir, iteration=iteration)

    l, z = sucre.compute_l_z(matches_data.cP)
    sucre.update_J(matches_data=matches_data, l=l, z=z)
    if save_dir is not None and save_views:
        sucre.save_all_views(save_dir=save_dir, image_list=view_image_list, iteration=num_iter)
    if save_dir is not None and save_point_cloud:
        sucre.save_point_cloud(save_dir=save_dir, iteration=num_iter)
    if save_dir is not None and export_light:
        sucre.save_light(save_dir=save_dir, iteration=num_iter)
        light_poses[num_iter] = sucre.light_pose().tolist()
        light_dir = save_dir / f'{Path(sucre.image.name).stem}_light'
        (light_dir / 'poses.json').write_text(json.dumps({'light_to_camera': light_poses}, indent=1))
    return sucre


def farthest_point_sample(images: list[sfm.Image], target: sfm.Image, count: int) -> list[sfm.Image]:
    """Pick `count` images whose camera centers are spatially well distributed.

    Greedy farthest-point sampling seeded with the target image: each step adds the image
    whose camera center is farthest from the closest already-selected one. The target is
    always kept. `count <= 0` or `count >= len(images)` keeps every image.
    """
    if count <= 0 or count >= len(images):
        return images
    center = {im.name: im.pose.t.flatten() for im in images}
    center.setdefault(target.name, target.pose.t.flatten())
    selected = [target]
    remaining = [im for im in images if im.name != target.name]
    while len(selected) < count and remaining:
        farthest = max(remaining, key=lambda im: min(
            (center[im.name] - center[s.name]).norm().item() for s in selected))
        selected.append(farthest)
        remaining.remove(farthest)
    return selected


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
        save_views: bool = False,
        save_light: bool = False,
        save_point_cloud: bool = False,
        num_views: int = 0,
        min_sample_cover: float = 0.0,
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

    view_image_list = matches_file.get_image_list() if save_views else None
    if view_image_list is not None and num_views > 0:
        covers = matches_file.get_covers(image)
        candidates = [im for im in view_image_list
                      if covers.get(im.name, 0.0) >= min_sample_cover or im.name == image.name]
        view_image_list = farthest_point_sample(candidates, image, num_views)
        print(f'Saving {len(view_image_list)} spatially-distributed views '
              f'(sampled from {len(candidates)} view(s) with cover >= {min_sample_cover}).')
    adam(sucre=sucre, matches_data=matches_data, lr=lr, num_iter=num_iter,
         light_regularizer=light_regularizer, save_dir=output_dir, save_interval=save_interval,
         save_views=save_views, save_light=save_light, save_point_cloud=save_point_cloud,
         view_image_list=view_image_list)

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
            save_views=args.save_all_views,
            save_light=args.save_light,
            save_point_cloud=args.save_point_cloud,
            num_views=args.num_views,
            min_sample_cover=args.min_sample_cover,
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
    parser.add_argument('--save-all-views', action='store_true',
                        help='at each --save-interval step (and at the end), also save the reconstructed '
                             'image and illumination pattern rendered from every matched image viewpoint, '
                             'as JPEG95 (used to build the 3D illustration).')
    parser.add_argument('--num-views', type=int, default=0,
                        help='if > 0, --save-all-views saves only this many spatially-distributed '
                             'views (farthest-point sampled, the restored target always kept); '
                             '0 saves every matched view.')
    parser.add_argument('--min-sample-cover', type=float, default=0.0,
                        help='when sampling views with --num-views, only consider views whose '
                             'cover (fraction of the target image they match) is at least this '
                             'value; the restored target is always kept.')
    parser.add_argument('--save-light', action='store_true',
                        help='at each --save-interval step, export the light pattern and pose to '
                             '<output>/<target>_light/ (much faster than --save-all-views, which '
                             'also implies it).')
    parser.add_argument('--save-point-cloud', action='store_true',
                        help='at each --save-interval step, export the restored RGB image to '
                             '<output>/<target>_pointcloud/ together with the fixed point-cloud '
                             'geometry (used to build the 3D point cloud in the illustration).')
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
