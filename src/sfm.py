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

from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from colmap.scripts.python.read_write_model import read_cameras_binary, read_images_binary


class Pose:
    def __init__(self, R: Tensor, t: Tensor):
        """6 degree-of-freedom camera pose

        :param R: rotation matrix, shape (3, 3).
        :param t: translation vector, shape (3, 1).
        """
        self.R = R
        self.t = t

    def inverse(self) -> Pose:
        """Inverse the pose in SE(3)

        :return: the inversed pose.
        """
        return Pose(self.R.T, -self.R.T @ self.t)

    def transform(self, P: Tensor) -> Tensor:
        """Transforms points in the pose frame

        :param P: points to transform, shape (3, n).
        :return: transformed points, shape (3, n).
        """
        return self.R.to(P.device) @ P + self.t.to(P.device)

    def __repr__(self) -> str:
        """String representation"""
        return f'Pose(R={repr(self.R)}, t={repr(self.t)})'


class Camera:
    def __init__(self, camera_id: int, width: int, height: int, K: Tensor):
        """Camera object

        :param camera_id: id of camera in COLMAP database.
        :param width: width of sensor in pixels.
        :param height: height of sensor in pixels.
        :param K: camera matrix, shape (3, 3).
        """
        self.id = camera_id
        self.width = width
        self.height = height
        self.K = K
        self.K_inv = K.inverse()

    def __repr__(self) -> str:
        """String representation"""
        return f'Camera(camera_id={self.id}, width={self.width}, height={self.height}, K={repr(self.K)})'


class SfMImage:
    def __init__(self, image_id: int, image_path: Path, pose: Pose, camera: Camera, colmap_model: COLMAPModel):
        """Image object

        :param image_id: image id in COLMAP databse.
        :param image_path: path to the image.
        :param pose: pose object of the image.
        :param camera: camera object of the image.
        :param colmap_model: COLMAP model object.
        """
        self.id = image_id
        self.name = image_path.name
        self.image_path = image_path
        self.depth_path = image_path.parents[1] / 'depth_maps' / f'depth_{image_path.with_suffix(".png").name}'
        self.pose = pose
        self.camera = camera
        self.colmap_model = colmap_model
        self.voxel_list = []

    def unproject_depth_map(self, depth_map: Tensor, transform: bool = True) -> tuple[Tensor, Tensor, Tensor]:
        """Unprojects the depth map of the image in 3D world

        :param depth_map: depth map of the image.
        :param transform: whether to unproject the depth map in the image frame (False) or the world frame (True).
        :return: pixels coordinates and their corresponding 3D coordinates in the given frame.
        """
        v, u = torch.where(depth_map > 0)
        uvw = torch.stack([u + 0.5, v + 0.5, torch.ones_like(u)])
        cp = uvw * depth_map[v, u]
        cP = self.camera.K_inv.to(cp.device) @ cp
        if not transform:
            return u, v, cP
        wP = self.pose.transform(cP)
        return u, v, wP

    def distance_map(self, depth_map: Tensor) -> Tensor:
        """Compute the distance map from the distance map using the camera intrinsics

        :param depth_map: depth map of the image.
        :return: distance map of the image.
        """
        u, v, cP = self.unproject_depth_map(depth_map, transform=False)
        distance_map = torch.zeros(self.camera.height, self.camera.width, device=cP.device)
        distance_map[v, u] = cP.norm(dim=0)
        return distance_map

    def compute_voxel_list(self, depth_map: Tensor):
        """Compute the list of voxels observed by the image by unprojecting the depth map

        :param depth_map: depth map of the image.
        """
        _, _, wP = self.unproject_depth_map(depth_map)
        voxels = self.colmap_model.voxels.voxelize(wP.T).unique(dim=0).cpu()
        for voxel in voxels:
            self.voxel_list.append(tuple(voxel.tolist()))

    def project(self, wP: Tensor) -> Tensor:
        """Projects 3D points in the image view

        :param wP: 3D points in the world frame, shape (3, n).
        :return: pixels coordinates of the 3D points, shape (2, n).
        """
        cP = self.pose.inverse().transform(wP)
        cp = self.camera.K.to(cP.device) @ cP
        px = cp[:2] / cp[2]
        return px

    def match_brut(self, other: SfMImage, u1: Tensor, v1: Tensor, wP1: Tensor) -> Tensor:
        """Pair pixels coordinates with another view

        :param other: other image with which to pair pixels.
        :param u1: u pixels coordinates to pair.
        :param v1: v pixels coordinates to pair.
        :param wP1: 3D points of (u, v) pairs in the world frame.
        :return: pixels coordinates matches, same shape as the image.
        """
        u2, v2 = other.project(wP1).long()
        args = (0 <= u2) & (u2 < other.camera.width) & (0 <= v2) & (v2 < other.camera.height)
        u1, v1, u2, v2 = u1[args], v1[args], u2[args], v2[args]
        matches = torch.zeros(self.camera.height, self.camera.width, 2, dtype=torch.int64, device=wP1.device) - 1
        matches[v1, u1, 0], matches[v1, u1, 1] = v2, u2
        return matches

    def match_dense(self, other: SfMImage, u1: Tensor, v1: Tensor, wP1: Tensor, u2: Tensor, v2: Tensor,
                    wP2: Tensor) -> Matches:
        """Dense pixels coordinates pairs between two images

        :param other: other image with which to pair pixel coordinates.
        :param u1: u pixels coordinates of this image.
        :param v1: v pixels coordinates of this image.
        :param wP1: 3D points of (u1, v1) pairs in the world frame.
        :param u2: u pixels coordinates of the other image.
        :param v2: v pixels coordinates of the other image.
        :param wP2: 3D points of (u2, v2) pairs in the world frame.
        :return: pairs between the images.
        """
        matches1 = self.match_brut(other, u1=u1, v1=v1, wP1=wP1)
        matches2 = other.match_brut(self, u1=u2, v1=v2, wP1=wP2)
        v1, u1 = torch.where(matches1[:, :, 0] != -1)
        v2, u2 = matches1[v1, u1].T
        args = torch.all(matches2[v2, u2] == torch.stack([v1, u1]).T, dim=1)
        u1, v1, u2, v2 = u1[args], v1[args], u2[args], v2[args]
        return Matches(image1=self, image2=other, u1=u1, v1=v1, u2=u2, v2=v2)

    def compute_matches(self, image_list: list[SfMImage], min_cover: float = 0.01, num_workers: int = 0,
                        device: str = 'cpu') -> tuple[list[Matches], Tensor]:
        """Computes all pairs between this image and all images in the image list

        :param image_list: list of images to pair.
        :param min_cover: minimum cover in percentile to keep the matches
        :param num_workers: number of thread to load the images and depth maps.
        :param device: device on which to compute the matches.
        :return: list of all matches and a count map with the same shape as the image.
        """
        u1, v1, wP1 = self.unproject_depth_map(self.depth_map().to(device))
        image_dataset = ImageDataset(image_list, load_image=False)
        image_loader = DataLoader(image_dataset, num_workers=num_workers, collate_fn=lambda x: x[0])
        count = torch.zeros(self.camera.height, self.camera.width, dtype=torch.int32, device=device)
        matches_dict = {}
        for other_name, other_depth_map in tqdm.tqdm(image_loader):
            other = self.colmap_model[other_name]
            other_depth_map = other_depth_map.to(device)
            u2, v2, wP2 = other.unproject_depth_map(other_depth_map)
            other_matches = self.match_dense(other, u1=u1, v1=v1, wP1=wP1, u2=u2, v2=v2, wP2=wP2)
            if len(other_matches) / (self.camera.width * self.camera.height) > min_cover:
                count[other_matches.v1, other_matches.u1] += 1
                other_distance_map = other.distance_map(other_depth_map)
                other_matches.z = other_distance_map[other_matches.v2, other_matches.u2]
                other_matches.to('cpu')
                matches_dict[other] = other_matches
        image_dataset = ImageDataset(list(matches_dict), load_image=True)
        image_loader = DataLoader(image_dataset, num_workers=num_workers, collate_fn=lambda x: x[0])
        for other_name, other_image in tqdm.tqdm(image_loader):
            other_matches = matches_dict[self.colmap_model[other_name]]
            other_matches.I = other_image.to(device)[other_matches.v2.to(device), other_matches.u2.to(device)].cpu()
        return list(matches_dict.values()), count

    def __repr__(self) -> str:
        """String representation"""
        return f'SfMImage({repr(self.name)})'


class Matches:
    def __init__(self, image1: SfMImage, image2: SfMImage, u1: Tensor, v1: Tensor, u2: Tensor, v2: Tensor,
                 I: Tensor = None, z: Tensor = None):
        """Matches object that embeds pairs between two images

        :param image1: first image.
        :param image2: second image.
        :param u1: u pixels coordinates of image1, shape (n,).
        :param v1: v pixels coordinates of image1, shape (n,).
        :param u2: u pixels coordinates of image2, shape (n,).
        :param v2: v pixels coordinates of image2, shape (n,).
        :param I: intensity of image2 at pixels coordinates (u2, v2), shape (n, 3).
        :param z: distance of image2 at pixels coordinates (u2, v2), shape (n,).
        """
        self.image1 = image1
        self.image2 = image2
        self.u1 = u1
        self.v1 = v1
        self.u2 = u2
        self.v2 = v2
        self.I = I
        self.z = z

    def to(self, device: str):
        """Transfers matches to specified device

        :param device: device on which to transfer the matches.
        """
        self.u1 = self.u1.to(device)
        self.v1 = self.v1.to(device)
        self.u2 = self.u2.to(device)
        self.v2 = self.v2.to(device)
        if self.I is not None:
            self.I = self.I.to(device)
        if self.z is not None:
            self.z = self.z.to(device)

    def filter(self, args_valid: Tensor):
        """Filters matches according to the index given in argument

        :param args_valid: index to filter the matches.
        """
        args = args_valid[self.v1, self.u1]
        self.u1 = self.u1[args]
        self.v1 = self.v1[args]
        self.u2 = self.u2[args]
        self.v2 = self.v2[args]
        self.I = self.I[args]
        self.z = self.z[args]

    def plot(self, step: int = 10000, color: tuple[float, float, float] = None) -> Image:
        """Creates an image that illustrates a subsample of the matches. Mostly used for manual inspection

        :param step: sample one pair every `step` pairs.
        :param color: color of the matches.
        :return: image with matches.
        """
        image1 = self.image1.image()
        image2 = self.image2.image()
        imatch = Image.fromarray(np.uint8(torch.concat([image1, image2], dim=1) * 255))
        draw = ImageDraw.Draw(imatch)
        for u1, v1, u2, v2 in zip(self.u1[::step], self.v1[::step], self.u2[::step], self.v2[::step]):
            fill = tuple(np.random.randint(0, 256, 3)) if color is None else color
            draw.line([(u1, v1), (u2 + image1.shape[1], v2)], fill=fill, width=3)
        return imatch

    def __len__(self) -> int:
        """Returns the number of matches"""
        return self.u1.shape[0]

    def __repr__(self) -> str:
        """String representation"""
        return f'Matches(image1={repr(self.image1)}, image2={repr(self.image2)}, {len(self)} matches)'


class ImageDataset(Dataset):
    def __init__(self, image_list: list[SfMImage], load_image: bool = True):
        """PyTorch Dataset to load images with multiple threads

        :param image_list: list of images to load.
        :param load_image: wheter to load images (True) or depth maps (False).
        """
        self.images = image_list
        self.load_image = load_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, x) -> tuple[str, Tensor]:
        if self.load_image:
            return self.images[x].name, self.images[x].image()
        else:
            return self.images[x].name, self.images[x].depth_map()


class COLMAPModel:
    def __init__(self, data_dir: Path, voxel_size: float):
        """COLMAP dataset object

        :param data_dir: path to directory with undistorted images, depth maps and corresponding COLMAP model.
        :param voxel_size: size of voxels for fast image retrieval.
        """
        colmap_cameras = read_cameras_binary(data_dir / 'sparse' / 'cameras.bin')
        colmap_images = read_images_binary(data_dir / 'sparse' / 'images.bin')

        self.voxels = Voxels(colmap_model=self, voxel_size=voxel_size)

        self.cameras = {}
        for camera in colmap_cameras.values():
            fx, fy, u0, v0 = camera.params
            self.cameras[camera.id] = Camera(
                camera.id,
                camera.width,
                camera.height,
                torch.tensor([
                    [fx, 0, u0],
                    [0, fy, v0],
                    [0, 0, 1]
                ], dtype=torch.float32)
            )

        self.images = {}
        for image in colmap_images.values():
            self.images[image.id] = SfMImage(
                image.id,
                data_dir / 'images' / image.name,
                Pose(
                    torch.tensor(image.qvec2rotmat(), dtype=torch.float32),
                    torch.tensor(image.tvec, dtype=torch.float32).view(3, 1)
                ).inverse(),
                self.cameras[image.camera_id],
                self
            )

        self.imagename2id = {image.name: image.id for image in self.images.values()}

    def __getitem__(self, image_name: str) -> SfMImage:
        """Returns the image with name `image_name`

        :param image_name: name of the image.
        :return: image object.
        """
        return self.images[self.imagename2id[image_name]]

    def compute_voxels(self, image_list: list[SfMImage], num_workers: int = 0, device: str = 'cpu'):
        """Computes voxels observed by all images in the dataset

        :param image_list: list of images to compute voxels.
        :param num_workers: number of threads for depth maps loading.
        :param device: device on which to compute the voxels.
        """
        image_dataset = ImageDataset(image_list, load_image=False)
        image_loader = DataLoader(image_dataset, num_workers=num_workers, collate_fn=lambda x: x[0])
        for image_name, image_depth_map in tqdm.tqdm(image_loader):
            image = self[image_name]
            image.compute_voxel_list(image_depth_map.to(device))
            self.voxels.append(image.id, image.voxel_list)


class Voxels:
    def __init__(self, colmap_model: COLMAPModel, voxel_size: float):
        """Voxel grid embedding the scene

        :param colmap_model: COLMAP dataset object.
        :param voxel_size: size of voxels.
        """
        self.colmap_model = colmap_model
        self.voxel_size = voxel_size
        self.voxels_dict = {}

    def voxelize(self, points: Tensor) -> Tensor:
        """Transforms points into voxels

        :param points: points for which to compute voxels, shape (n, 3).
        :return: voxels, shape (n, 3).
        """
        voxels = points / self.voxel_size
        return voxels.floor().int()

    def append(self, image_id: int, voxels: list[tuple[int, int, int]]):
        """Append list of voxels to voxels dict

        :param image_id: id of image embedding the `voxels`.
        :param voxels: voxels list of image with `image_id`.
        """
        for voxel in voxels:
            if voxel not in self.voxels_dict:
                self.voxels_dict[voxel] = [image_id]
            else:
                self.voxels_dict[voxel].append(image_id)

    def pairs(self, voxels: list[tuple[int, int, int]], min_cover: float = 0.01) -> list[SfMImage]:
        """Find all images observing the `voxels`

        :param voxels: list of voxels for which to retrieve observing images.
        :param min_cover: only retrieves images with at least `min_cover` percentile shared voxels observations.
        :return: list of images that observe the same voxels.
        """
        pair_ids = []
        for voxel in voxels:
            pair_ids += self.voxels_dict[voxel]
        pair_ids, counts = np.unique(pair_ids, return_counts=True)
        pair_ids = pair_ids[counts > int(min_cover * len(voxels))]
        pairs = [self.colmap_model.images[pair_id] for pair_id in pair_ids]
        return pairs
