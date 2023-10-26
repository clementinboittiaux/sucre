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

import torch
from torch import Tensor


def exp(pose: Tensor) -> tuple[Tensor, Tensor]:
    w1, w2, w3, p1, p2, p3 = pose
    zero = torch.zeros_like(w1)
    pose = torch.stack([zero, -w3, w2, p1, w3, zero, -w1, p2, -w2, w1, zero, p3, zero, zero, zero, zero])
    pose = torch.matrix_exp(pose.view(4, 4))
    return pose[:3, :3], pose[:3, 3:4]
