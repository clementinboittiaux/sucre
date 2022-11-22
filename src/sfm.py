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

import numpy as np
import torch
import tqdm
import PIL
from torch import Tensor
import loader


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
        return f'Camera(id={self.id}, width={self.width}, height={self.height}, K={repr(self.K)})'


class Image:
    def __init__(self, image_id: int, image_path: Path, depth_path: Path, pose: Pose, camera: Camera):
        self.id = image_id
        self.name = image_path.name
        self.image_path = image_path
        self.depth_path = depth_path
        self.pose = pose
        self.camera = camera

    def unproject_depth_map(self, depth_map: Tensor, transform: bool = True) -> tuple[Tensor, Tensor, Tensor]:
        v, u = torch.where(depth_map > 0)
        uvw = torch.stack([u + 0.5, v + 0.5, torch.ones_like(u)])
        cp = uvw * depth_map[v, u]
        cP = self.camera.K_inv.to(cp.device) @ cp
        if not transform:
            return u, v, cP
        wP = self.pose.transform(cP)
        return u, v, wP

    def distance_map(self, depth_map: Tensor) -> Tensor:
        u, v, cP = self.unproject_depth_map(depth_map, transform=False)
        distance_map = torch.zeros_like(depth_map)
        distance_map[v, u] = cP.norm(dim=0)
        return distance_map

    def project(self, wP: Tensor) -> Tensor:
        cP = self.pose.inverse().transform(wP)
        cp = self.camera.K.to(cP.device) @ cP
        px = cp[:2] / cp[2]
        return px

    def match_one_way(self, other: Image, u1: Tensor, v1: Tensor, wP1: Tensor) -> Matches:
        u2, v2 = other.project(wP1).long()
        args = (0 <= u2) & (u2 < other.camera.width) & (0 <= v2) & (v2 < other.camera.height)
        u1, v1, u2, v2 = u1[args], v1[args], u2[args], v2[args]
        return Matches(image1=self, image2=other, u1=u1, v1=v1, u2=u2, v2=v2)

    def match_two_way(self, other: Image, u1: Tensor, v1: Tensor, wP1: Tensor, u2: Tensor, v2: Tensor,
                      wP2: Tensor) -> Matches:
        matches1 = self.match_one_way(other, u1=u1, v1=v1, wP1=wP1)
        matches2 = other.match_one_way(self, u1=u2, v1=v2, wP1=wP2)
        return matches1 & matches2

    def match_images(self, image_list: list[Image], min_cover: float = 0.01, num_workers: int = 0,
                        device: str = 'cpu') -> tuple[list[Matches], Tensor]:
        u1, v1, wP1 = self.unproject_depth_map(loader.load_depth(self.depth_path).to(device))
        matches_dict = {}
        for other_idx, other_depth_map in tqdm.tqdm(
                loader.loader(image_list, return_image=False, num_workers=num_workers)
        ):
            other = image_list[other_idx]
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
    def __init__(self, image1: Image, image2: Image, u1: Tensor, v1: Tensor, u2: Tensor, v2: Tensor):
        self.image1 = image1
        self.image2 = image2
        self.u1 = u1
        self.v1 = v1
        self.u2 = u2
        self.v2 = v2

    def map(self):
        match_map = torch.zeros(
            (self.image1.camera.height, self.image1.camera.width, 2),
            device=self.u1.device,
            dtype=self.u1.dtype
        )
        match_map[self.v1, self.u1, 0] = self.v2
        match_map[self.v1, self.u1, 1] = self.u2
        return match_map

    def to(self, device: str):
        self.u1 = self.u1.to(device)
        self.v1 = self.v1.to(device)
        self.u2 = self.u2.to(device)
        self.v2 = self.v2.to(device)

    def plot(self, step: int = 10000, color: tuple[float, float, float] = None) -> Image:
        image1 = loader.load_image(self.image1.image_path)
        image2 = loader.load_image(self.image2.image_path)
        imatch = PIL.Image.fromarray(np.uint8(torch.concat([image1, image2], dim=1) * 255))
        draw = PIL.ImageDraw.Draw(imatch)
        for u1, v1, u2, v2 in zip(self.u1[::step], self.v1[::step], self.u2[::step], self.v2[::step]):
            fill = tuple(np.random.randint(0, 256, 3)) if color is None else color
            draw.line([(u1, v1), (u2 + image1.shape[1], v2)], fill=fill, width=3)
        return imatch

    def __and__(self, other: Matches) -> Matches:
        match_map = other.map()
        args = torch.all(match_map[self.v2, self.u2] == torch.stack([self.v1, self.u1]).T, dim=1)
        return Matches(
            image1=self.image1,
            image2=self.image2,
            u1=self.u1[args],
            v1=self.v1[args],
            u2=self.u2[args],
            v2=self.v2[args]
        )


    def __len__(self) -> int:
        """Returns the number of matches"""
        return self.u1.shape[0]

    def __repr__(self) -> str:
        """String representation"""
        return f'Matches(image1={repr(self.image1)}, image2={repr(self.image2)}, {len(self)} matches)'


class COLMAPModel:
    def __init__(self, model_dir: Path, image_dir: Path, depth_dir: Path):
        self.cameras = {}
        for camera in loader.load_cameras(model_dir).values():
            fx, fy, u0, v0 = camera.params
            self.cameras[camera.id] = Camera(
                camera_id=camera.id,
                width=camera.width,
                height=camera.height,
                K=torch.tensor([
                    [fx, 0, u0],
                    [0, fy, v0],
                    [0, 0, 1]
                ], dtype=torch.float32)
            )

        self.images = {}
        for image in loader.load_images(model_dir).values():
            image_path = image_dir / image.name
            depth_path = (depth_dir / image.name).with_stem('depth_' + image_path.stem).with_suffix('.png')
            self.images[image.id] = Image(
                image_id=image.id,
                image_path=image_path,
                depth_path=depth_path,
                pose=Pose(
                    torch.tensor(image.qvec2rotmat(), dtype=torch.float32),
                    torch.tensor(image.tvec.reshape(3, 1), dtype=torch.float32)
                ).inverse(),
                camera=self.cameras[image.camera_id]
            )

        self.imagename2id = {image.name: image.id for image in self.images.values()}

    def __getitem__(self, image_name: str) -> Image:
        """Returns Image with name `image_name`

        :param image_name: Image name.
        :return: Image.
        """
        return self.images[self.imagename2id[image_name]]
