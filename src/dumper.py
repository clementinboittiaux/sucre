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

import numpy as np
import sfm
import h5py


def save_matches(stream: h5py.File, matches: sfm.Matches):
    group = stream.create_group(matches.image2.name)
    group.create_dataset('u1', data=matches.u1.short().cpu().numpy())
    group.create_dataset('v1', data=matches.v1.short().cpu().numpy())
    group.create_dataset('u2', data=matches.u2.short().cpu().numpy())
    group.create_dataset('v2', data=matches.v2.short().cpu().numpy())
    group.create_dataset('z', data=np.full(len(matches), np.nan, dtype=np.float32))
    group.create_dataset('I', data=np.full((3, len(matches)), np.nan, dtype=np.float32))
