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

"""Place Blender cameras at the COLMAP poses of the SUCRe views.

This script wipes the current scene, then reads a COLMAP model (``.bin`` or
``.txt``) and, for every saved view, creates a Blender camera with intrinsics
(focal length, aspect ratio, principal point) consistent with the COLMAP
cameras, plus an image plane -- coinciding with the camera's view frame --
textured with the view's reconstructed image.

The reconstructed images are loaded as an image sequence, so scrubbing the
timeline shows the optimization evolving: scene frame ``f`` displays the
reconstruction at optimization step ``f``.

The optimized light is added as a camera parented to the target camera: its
pose is keyframed per optimization step and its own image plane shows the
illumination pattern, also as an image sequence.

It can be run two ways:

1. From Blender's Text Editor: edit the ``CONFIG`` dictionary below, then `Run Script`.
2. From the command line (headless):
       blender --background --python blender/place_cameras.py -- \\
           --model-dir /path/to/sparse \\
           --views-dir outputs/<target>_views

If ``--views-dir`` points at a SUCRe ``*_views`` directory, only the images that
have a sub-folder there get a camera (and the scene render settings are taken
from the restored target image). Otherwise every image of the model is placed.

The resulting scene is always saved to a ``.blend`` file (see ``--output-blend``).
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

# --------------------------------------------------------------------------- #
# Defaults used when the script is run from Blender's Text Editor (no CLI args).
# --------------------------------------------------------------------------- #
CONFIG = {
    'model_dir': '/home/ubuntu/Data/EiffelTower/sparse',
    'views_dir': '/home/ubuntu/Dev/sucre/outputs/20150418T014353.000Z_views',
    'collection': 'SUCRe Cameras',
    'sensor_width': 36.0,         # arbitrary virtual sensor size, in mm
    'camera_display_size': 1.6,   # viewport size of the camera gizmos (and image planes)
    'point_radius': 0.01,         # viewport radius of the unprojected point-cloud points
    'output_blend': '',           # .blend save path; empty = derive from views_dir
    'export_web': '',             # if set, also export an interactive web viewer here
}

# COLMAP camera models: id -> (name, number of parameters).
CAMERA_MODELS = {
    0: ('SIMPLE_PINHOLE', 3), 1: ('PINHOLE', 4), 2: ('SIMPLE_RADIAL', 4),
    3: ('RADIAL', 5), 4: ('OPENCV', 8), 5: ('OPENCV_FISHEYE', 8),
    6: ('FULL_OPENCV', 12), 7: ('FOV', 5), 8: ('SIMPLE_RADIAL_FISHEYE', 4),
    9: ('RADIAL_FISHEYE', 5), 10: ('THIN_PRISM_FISHEYE', 12),
}


# --------------------------------------------------------------------------- #
# COLMAP model readers (pure Python, no pycolmap dependency).
# --------------------------------------------------------------------------- #
def read_cameras_bin(path: Path) -> dict:
    cameras = {}
    with open(path, 'rb') as f:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            camera_id, model_id = struct.unpack('<ii', f.read(8))
            width, height = struct.unpack('<QQ', f.read(16))
            model, num_params = CAMERA_MODELS[model_id]
            params = struct.unpack('<' + 'd' * num_params, f.read(8 * num_params))
            cameras[camera_id] = {'model': model, 'width': width, 'height': height, 'params': params}
    return cameras


def read_images_bin(path: Path) -> dict:
    images = {}
    with open(path, 'rb') as f:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            image_id = struct.unpack('<i', f.read(4))[0]
            qvec = struct.unpack('<dddd', f.read(32))
            tvec = struct.unpack('<ddd', f.read(24))
            camera_id = struct.unpack('<i', f.read(4))[0]
            name = b''
            while (char := f.read(1)) not in (b'\x00', b''):
                name += char
            num_points2D = struct.unpack('<Q', f.read(8))[0]
            f.seek(24 * num_points2D, 1)  # skip the 2D points (x, y, point3D_id)
            images[image_id] = {'qvec': qvec, 'tvec': tvec,
                                'camera_id': camera_id, 'name': name.decode('utf-8')}
    return images


def read_cameras_txt(path: Path) -> dict:
    cameras = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        e = line.split()
        cameras[int(e[0])] = {'model': e[1], 'width': int(e[2]), 'height': int(e[3]),
                              'params': tuple(float(x) for x in e[4:])}
    return cameras


def read_images_txt(path: Path) -> dict:
    images = {}
    data = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith('#')]
    for line in data[::2]:  # COLMAP writes 2 lines per image, the 2nd holds 2D points
        e = line.split()
        images[int(e[0])] = {
            'qvec': tuple(float(x) for x in e[1:5]),
            'tvec': tuple(float(x) for x in e[5:8]),
            'camera_id': int(e[8]),
            'name': ' '.join(e[9:]),
        }
    return images


def read_colmap_model(model_dir: str) -> tuple[dict, dict]:
    """Read COLMAP cameras and images, supporting both the ``.bin`` and ``.txt`` formats."""
    model_dir = Path(model_dir)
    if (model_dir / 'cameras.bin').exists():
        return read_cameras_bin(model_dir / 'cameras.bin'), read_images_bin(model_dir / 'images.bin')
    if (model_dir / 'cameras.txt').exists():
        return read_cameras_txt(model_dir / 'cameras.txt'), read_images_txt(model_dir / 'images.txt')
    raise FileNotFoundError(f'No COLMAP cameras.bin/.txt found in {model_dir}.')


def pinhole_params(camera: dict) -> tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) for any COLMAP camera model."""
    model, p = camera['model'], camera['params']
    if model == 'PINHOLE':
        return p[0], p[1], p[2], p[3]
    if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL', 'FOV',
                 'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE'):
        return p[0], p[0], p[1], p[2]  # single shared focal length
    return p[0], p[1], p[2], p[3]      # OPENCV family: fx, fy, cx, cy come first


# --------------------------------------------------------------------------- #
# COLMAP <-> Blender conversions.
# --------------------------------------------------------------------------- #
def camera_to_world(qvec: tuple, tvec: tuple) -> Matrix:
    """Build the Blender ``matrix_world`` of a camera from a COLMAP (qvec, tvec) pose.

    COLMAP stores the world-to-camera pose and uses the computer-vision frame
    (x right, y down, z forward). Blender cameras use (x right, y up, z backward),
    hence the diag(1, -1, -1) axis flip.
    """
    rotation = Quaternion(qvec).to_matrix()         # world-to-camera rotation
    rotation_cw = rotation.transposed()             # camera-to-world rotation
    center = -(rotation_cw @ Vector(tvec))          # camera center in world
    matrix = rotation_cw.to_4x4()
    matrix.translation = center
    return matrix @ Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))


def apply_intrinsics(cam_data: bpy.types.Camera, fx: float, fy: float, cx: float, cy: float,
                     width: int, height: int, sensor_width: float):
    """Set a Blender camera's focal length and lens shift to match a COLMAP camera.

    ``fx != fy`` (non-square pixels) is handled at render time through the scene
    pixel aspect ratio (see :func:`apply_render_settings`).
    """
    cam_data.type = 'PERSP'
    if width >= height:
        cam_data.sensor_fit = 'HORIZONTAL'
        cam_data.sensor_width = sensor_width
        cam_data.lens = fx * sensor_width / width
    else:
        cam_data.sensor_fit = 'VERTICAL'
        cam_data.sensor_height = sensor_width
        cam_data.lens = fy * sensor_width / height
    # Lens shift for the principal point, in units of the largest image dimension.
    cam_data.shift_x = (width / 2 - cx) / max(width, height)
    cam_data.shift_y = (cy - height / 2) / max(width, height)
    # Keep the raw COLMAP intrinsics so later scripts can render at the right resolution.
    cam_data['colmap_fx'], cam_data['colmap_fy'] = fx, fy
    cam_data['colmap_cx'], cam_data['colmap_cy'] = cx, cy
    cam_data['colmap_width'], cam_data['colmap_height'] = width, height


def apply_render_settings(camera: dict):
    """Set the scene resolution / pixel aspect from a COLMAP camera (used for the target view)."""
    fx, fy, _, _ = pinhole_params(camera)
    render = bpy.context.scene.render
    render.resolution_x = camera['width']
    render.resolution_y = camera['height']
    render.resolution_percentage = 100
    # Non-square pixels: bias the pixel aspect so both fx and fy are honoured.
    render.pixel_aspect_x, render.pixel_aspect_y = (1.0, fx / fy) if fx > fy else (fy / fx, 1.0)


# --------------------------------------------------------------------------- #
# Blender scene helpers.
# --------------------------------------------------------------------------- #
def clear_scene():
    """Wipe every object, collection and orphan datablock from the file."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)
    for blocks in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights,
                   bpy.data.materials, bpy.data.images, bpy.data.node_groups):
        for block in list(blocks):
            try:
                blocks.remove(block)
            except (RuntimeError, ReferenceError):
                pass


def get_clean_collection(name: str) -> bpy.types.Collection:
    """Return a collection with the given name, emptied of any previous cameras."""
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
        return collection
    for obj in list(collection.objects):
        data = obj.data
        bpy.data.objects.remove(obj)
        if isinstance(data, bpy.types.Camera) and data.users == 0:
            bpy.data.cameras.remove(data)
    return collection


def create_camera(name: str, collection: bpy.types.Collection, config: dict) -> bpy.types.Object:
    cam_data = bpy.data.cameras.new(name)
    cam_data.display_size = config['camera_display_size']
    obj = bpy.data.objects.new(name, cam_data)
    collection.objects.link(obj)
    return obj


def find_image_sequence(directory: Path, prefix: str) -> list:
    """Return the sorted ``[(iteration, path), ...]`` of ``<prefix>_*.jpg`` files in a directory.

    Paths are resolved to absolute: Blender resolves a relative path given to
    ``images.load`` against the wrong base, which mangles the stored filepath.
    """
    sequence = []
    for path in directory.glob(f'{prefix}_*.jpg'):
        sequence.append((int(path.stem.split('_')[-1]), path.resolve()))
    return sorted(sequence)


def _add_sequence_texture(material: bpy.types.Material, sequence: list, num_frames: int,
                          frame_start: int = 0, cyclic: bool = False, uv_attribute: str = None,
                          location: tuple = (-400, 0)) -> bpy.types.Node:
    """Add (and wire) an image-sequence Image Texture node to ``material``; return the node.

    The sequence spans ``num_frames`` steps starting at scene frame ``frame_start``; with
    ``cyclic`` it loops. With ``uv_attribute`` it is sampled at that named geometry attribute
    (used by the point cloud) instead of the mesh UV map.
    """
    nodes, links = material.node_tree.nodes, material.node_tree.links
    texture = nodes.new('ShaderNodeTexImage')
    texture.location = location
    texture.image = bpy.data.images.load(str(sequence[0][1]), check_existing=True)
    if len(sequence) > 1:
        texture.image.source = 'SEQUENCE'
        texture.image_user.frame_duration = num_frames
        texture.image_user.frame_start = frame_start
        texture.image_user.frame_offset = -1
        texture.image_user.use_cyclic = cyclic
        texture.image_user.use_auto_refresh = True
    if uv_attribute is not None:
        attribute = nodes.new('ShaderNodeAttribute')
        attribute.location = (location[0] - 300, location[1])
        attribute.attribute_name = uv_attribute
        links.new(attribute.outputs['Vector'], texture.inputs['Vector'])
    return texture


def make_image_material(name: str, sequence: list, num_frames: int, uv_attribute: str = None,
                        cyclic: bool = False) -> bpy.types.Material:
    """Create an unlit (emission) material textured with a single image sequence.

    ``sequence`` is the sorted ``[(iteration, path), ...]`` list; scene frame ``f`` shows
    optimization step ``f``. With ``cyclic`` the sequence loops every ``num_frames`` frames
    (used by the light pattern, which replays over the second timeline phase).
    """
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    texture = _add_sequence_texture(material, sequence, num_frames, cyclic=cyclic,
                                    uv_attribute=uv_attribute)
    emission = nodes.new('ShaderNodeEmission')
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(texture.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    return material


def make_phased_material(name: str, sequence_a: list, sequence_b: list, num_frames: int,
                         uv_attribute: str = None) -> bpy.types.Material:
    """Create an unlit material for the two-phase timeline.

    ``sequence_a`` is shown over frames ``0..num_frames-1`` and ``sequence_b`` over the next
    ``num_frames`` frames (``sequence_b`` is offset to start at frame ``num_frames``). A Mix
    node switches between them; its factor is keyframed 0 -> 1 one frame apart at the phase
    boundary, so every integer (rendered) frame samples an exact 0 or 1.
    """
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    texture_a = _add_sequence_texture(material, sequence_a, num_frames, frame_start=0,
                                      uv_attribute=uv_attribute, location=(-400, 180))
    texture_b = _add_sequence_texture(material, sequence_b, num_frames, frame_start=num_frames,
                                      uv_attribute=uv_attribute, location=(-400, -180))
    mix = nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    mix.location = (-80, 0)
    color_inputs = [s for s in mix.inputs if s.type == 'RGBA']
    color_result = next(s for s in mix.outputs if s.type == 'RGBA')
    links.new(texture_a.outputs['Color'], color_inputs[0])
    links.new(texture_b.outputs['Color'], color_inputs[1])
    mix.inputs[0].default_value = 0.0
    mix.inputs[0].keyframe_insert('default_value', frame=num_frames - 1)
    mix.inputs[0].default_value = 1.0
    mix.inputs[0].keyframe_insert('default_value', frame=num_frames)
    emission = nodes.new('ShaderNodeEmission')
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(color_result, emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    return material


def make_image_plane(name: str, camera: bpy.types.Object, material: bpy.types.Material,
                     display_size: float) -> bpy.types.Object:
    """Create a plane that coincides with the camera's drawn view frame (its image plane).

    ``Camera.view_frame`` returns the four corners of the camera frame -- computed by
    Blender from the very same intrinsics (lens, sensor, lens shift, aspect ratio) -- for
    a unit display size. Scaling those corners by the camera display size makes the
    textured plane land exactly inside the camera gizmo's image plane.
    """
    corners = camera.data.view_frame(scene=bpy.context.scene)  # local space: TR, BR, BL, TL
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(corner * display_size) for corner in corners], [], [(0, 1, 2, 3)])
    mesh.update()
    uv_layer = mesh.uv_layers.new(name='UVMap')
    for loop_index, uv in enumerate([(1.0, 1.0), (1.0, 0.0), (0.0, 0.0), (0.0, 1.0)]):
        uv_layer.data[loop_index].uv = uv  # TR, BR, BL, TL
    mesh.materials.append(material)
    plane = bpy.data.objects.new(name, mesh)
    plane.parent = camera
    return plane


def create_light(name: str, collection: bpy.types.Collection, target_camera: bpy.types.Object,
                 target_colmap_camera: dict, light_dir: Path, num_frames: int, config: dict):
    """Add the optimized light as a camera parented to the target camera.

    The light shares the target camera's intrinsics. Its pose, given per optimization step
    relative to the camera by ``poses.json``, is keyframed on the timeline; its image plane
    shows the illumination pattern as an image sequence.
    """
    poses = json.loads((light_dir / 'poses.json').read_text())['light_to_camera']
    light_data = bpy.data.cameras.new(name)
    light_data.display_size = config['camera_display_size']
    fx, fy, cx, cy = pinhole_params(target_colmap_camera)
    apply_intrinsics(light_data, fx, fy, cx, cy, target_colmap_camera['width'],
                     target_colmap_camera['height'], config['sensor_width'])
    light = bpy.data.objects.new(name, light_data)
    light.parent = target_camera
    light.rotation_mode = 'QUATERNION'
    collection.objects.link(light)

    # Keyframe the light pose: light-to-camera (computer-vision) -> Blender local transform.
    # It is keyframed in both timeline phases so the light retraces the same trajectory
    # while the cameras switch from reconstruction to projected illumination.
    flip = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))
    for iteration, light_to_camera in poses.items():
        light.matrix_basis = flip @ Matrix(light_to_camera) @ flip
        for phase_start in (0, num_frames):
            frame = int(iteration) + phase_start
            light.keyframe_insert('location', frame=frame)
            light.keyframe_insert('rotation_quaternion', frame=frame)

    # Image plane showing the illumination pattern; cyclic so it replays in the second phase.
    patterns = find_image_sequence(light_dir, 'pattern')
    if patterns:
        material = make_image_material(f'{name}_pattern', patterns, num_frames, cyclic=True)
        collection.objects.link(make_image_plane(
            f'{name}_plane', light, material, config['camera_display_size']))
    return light


def make_point_cloud_modifier(obj: bpy.types.Object, name: str, radius: float,
                              material: bpy.types.Material) -> bpy.types.Modifier:
    """Add a Geometry Nodes modifier turning the mesh vertices into a renderable point cloud.

    ``Mesh to Points`` keeps the per-point attributes (each point's source pixel coordinate)
    and ``Set Material`` applies the restored-RGB image-sequence material to the points.
    """
    group = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    group.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nodes, links = group.nodes, group.links
    group_input = nodes.new('NodeGroupInput')
    group_input.location = (-300, 0)
    group_output = nodes.new('NodeGroupOutput')
    group_output.location = (300, 0)
    to_points = nodes.new('GeometryNodeMeshToPoints')
    to_points.inputs['Radius'].default_value = radius
    set_material = nodes.new('GeometryNodeSetMaterial')
    set_material.location = (150, 0)
    set_material.inputs['Material'].default_value = material
    links.new(group_input.outputs['Geometry'], to_points.inputs['Mesh'])
    links.new(to_points.outputs['Points'], set_material.inputs['Geometry'])
    links.new(set_material.outputs['Geometry'], group_output.inputs['Geometry'])
    modifier = obj.modifiers.new(name, 'NODES')
    modifier.node_group = group
    return modifier


def create_point_cloud(name: str, collection: bpy.types.Collection, pointcloud_dir: Path,
                       target_view_dir: Path, num_frames: int, config: dict) -> bpy.types.Object:
    """Add the unprojected restored RGB image as a 3D point cloud.

    ``geometry.npz`` gives one 3D world point per target pixel (fixed across optimization
    steps) plus its texture coordinate; the vertices are rendered as points through a
    Geometry Nodes modifier, each sampling a color sequence at its coordinate. Over the two
    timeline phases the color is the restored RGB image (``rgb_*``), then the target view's
    projected light pattern (``vignetting_*``).
    """
    import numpy as np  # bundled with Blender
    geometry = np.load(pointcloud_dir / 'geometry.npz')
    xyz, uv = geometry['xyz'], geometry['uv']
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(xyz.tolist(), [], [])
    mesh.update()
    # Per-point texture coordinate into the restored RGB image (third component unused).
    attribute = mesh.attributes.new('rgb_uv', 'FLOAT_VECTOR', 'POINT')
    uvw = np.concatenate([uv, np.zeros((len(uv), 1))], axis=1)
    attribute.data.foreach_set('vector', uvw.reshape(-1))
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    rgbs = find_image_sequence(pointcloud_dir, 'rgb')
    target_vignettings = find_image_sequence(target_view_dir, 'vignetting')
    if target_vignettings:
        material = make_phased_material(f'{name}_color', rgbs, target_vignettings,
                                        num_frames, uv_attribute='rgb_uv')
    else:
        material = make_image_material(f'{name}_color', rgbs, num_frames, uv_attribute='rgb_uv')
    mesh.materials.append(material)
    make_point_cloud_modifier(obj, f'{name}_nodes', config['point_radius'], material)
    return obj


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #
def parse_config() -> dict:
    """Read configuration from CLI args (after a ``--``) falling back to ``CONFIG``."""
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', default=CONFIG['model_dir'],
                        help='path to the COLMAP model directory.')
    parser.add_argument('--views-dir', default=CONFIG['views_dir'],
                        help='path to a SUCRe <target>_views directory. Pass an empty '
                             'string to place every image of the model instead.')
    parser.add_argument('--collection', default=CONFIG['collection'],
                        help='name of the Blender collection holding the cameras.')
    parser.add_argument('--sensor-width', type=float, default=CONFIG['sensor_width'],
                        help='virtual sensor size in mm (only affects the lens value, not the FOV).')
    parser.add_argument('--camera-display-size', type=float, default=CONFIG['camera_display_size'],
                        help='viewport display size of the camera gizmos and their image planes.')
    parser.add_argument('--point-radius', type=float, default=CONFIG['point_radius'],
                        help='viewport radius of the unprojected point-cloud points.')
    parser.add_argument('--output-blend', default=CONFIG['output_blend'],
                        help='path to save the resulting .blend file. If empty, it is derived '
                             'from --views-dir as <views-dir>/../<target>.blend.')
    parser.add_argument('--export-web', default=CONFIG['export_web'],
                        help='if set, also export a self-contained interactive Three.js '
                             'viewer of the scene to this directory.')
    return vars(parser.parse_args(argv))


WEB_FPS = 24  # playback / encoding frame rate for the web viewer's video sequences


def encode_video(image_dir: Path, prefix: str, output: Path, fps: int = WEB_FPS,
                 scale: float = 1.0):
    """Encode ``<prefix>_NNNN.jpg`` in ``image_dir`` into an H.264 .mp4 at ``output``.

    Short GOP (I-frame every 5 frames) keeps seeking responsive; ``+faststart`` puts
    the moov atom up front so the browser can play the file as it downloads. The glob
    pattern is used (rather than ``%04d``) so non-contiguous frame indices -- e.g.
    from ``--save-interval N`` -- still encode in lexicographic order. ``scale`` < 1
    downsamples with Lanczos for the web (small image planes don't need full res);
    output dimensions are rounded to even values, required by yuv420p.
    """
    import subprocess
    vf = f'scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2:flags=lanczos'
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-framerate', str(fps),
        '-pattern_type', 'glob', '-i', str(image_dir / f'{prefix}_*.jpg'),
        '-c:v', 'libx265', '-preset', 'slow', '-crf', '24', '-tag:v', 'hvc1',
        '-g', '5', '-keyint_min', '5',
        '-pix_fmt', 'yuv420p',
        '-vf', vf,
        '-movflags', '+faststart',
        str(output),
    ]
    subprocess.run(cmd, check=True)


def export_web_bundle(collection: bpy.types.Collection, config: dict, target_stem: str,
                      num_frames: int, phased: bool):
    """Export a self-contained Three.js viewer of the built scene to ``config['export_web']``.

    Geometry is read straight from the built Blender objects (so the placement logic is
    shared, not reimplemented): ``scene.json`` holds each camera's apex and image-plane
    corners in world coordinates plus the light's per-step transforms; ``pointcloud.bin``
    holds the point geometry. The per-step JPEG sequences are encoded to H.264 .mp4
    videos (one per sequence, ~10-20x smaller than the raw frames), since consecutive
    optimization steps are highly redundant. The static viewer files (the repository
    ``web/`` directory) are copied in, making the output directory deployable.
    """
    import shutil
    import numpy as np

    scene = bpy.context.scene
    out = Path(config['export_web']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'videos').mkdir(exist_ok=True)
    views_dir = Path(config['views_dir'])
    base = views_dir.parent

    # Saved iterations may be sparse (e.g. with --save-interval 2: 0, 2, 4, ...).
    # The web bundle's numFrames is the count of saved frames; the per-step light
    # data is sampled at each saved iteration's Blender frame.
    saved_iterations = [it for it, _ in find_image_sequence(views_dir / target_stem,
                                                            'reconstruction')]
    n_saved = len(saved_iterations) or num_frames

    def world_corners(plane):
        return [list(plane.matrix_world @ vertex.co) for vertex in plane.data.vertices]

    # View cameras and their image planes; encode the JPEG sequences to MP4.
    views = []
    for camera in [o for o in collection.objects
                   if o.type == 'CAMERA' and not o.name.endswith('_light')]:
        plane = collection.objects.get(f'{camera.name}_plane')
        if plane is None:
            continue
        view = {
            'name': camera.name,
            'isTarget': camera.name == target_stem,
            'apex': list(camera.matrix_world.translation),
            'corners': world_corners(plane),
        }
        view_dir = views_dir / camera.name
        recon_out = out / 'videos' / f'{camera.name}_reconstruction.mp4'
        encode_video(view_dir, 'reconstruction', recon_out, scale=0.5)
        view['reconstructionVideo'] = f'videos/{camera.name}_reconstruction.mp4'
        if phased:
            vign_out = out / 'videos' / f'{camera.name}_vignetting.mp4'
            # The target view's vignetting doubles as the point cloud's phase-2 color,
            # so it stays at full resolution; the other views are tiny image planes.
            vign_scale = 1.0 if camera.name == target_stem else 0.5
            encode_video(view_dir, 'vignetting', vign_out, scale=vign_scale)
            view['vignettingVideo'] = f'videos/{camera.name}_vignetting.mp4'
        views.append(view)

    # Light: world-space frustum + pattern plane, keyframed per optimization step.
    light_data = None
    light = collection.objects.get(f'{target_stem}_light')
    if light is not None:
        light_plane = collection.objects.get(f'{target_stem}_light_plane')
        apexes, corners = [], []
        for iteration in saved_iterations:
            scene.frame_set(iteration)
            apexes.append(list(light.matrix_world.translation))
            corners.append(world_corners(light_plane) if light_plane else [])
        scene.frame_set(0)
        light_dir = base / f'{target_stem}_light'
        encode_video(light_dir, 'pattern', out / 'videos' / 'light_pattern.mp4', scale=0.5)
        light_data = {'apex': apexes, 'corners': corners,
                      'patternVideo': 'videos/light_pattern.mp4'}

    # Point cloud: Draco-compressed positions + per-point UV (tex_coord).
    # quantization_bits=14 -> ~0.6 mm position precision over a 10 m scene,
    # which is well below the per-point pixel-quad footprint at our point sizes.
    import DracoPy
    point_cloud = None
    pc_dir = base / f'{target_stem}_pointcloud'
    if (pc_dir / 'geometry.npz').exists():
        geometry = np.load(pc_dir / 'geometry.npz')
        xyz = geometry['xyz'].astype(np.float32)
        # DracoPy.encode requires tex_coord as float64 (asserts dtype == float / float64).
        uv = geometry['uv'].astype(np.float64)
        drc = DracoPy.encode(xyz, tex_coord=uv, quantization_bits=14, compression_level=7)
        (out / 'pointcloud.drc').write_bytes(drc)
        encode_video(pc_dir, 'rgb', out / 'videos' / 'pointcloud_rgb.mp4')
        point_cloud = {'count': int(len(xyz)), 'data': 'pointcloud.drc',
                       'rgbVideo': 'videos/pointcloud_rgb.mp4'}
        if phased:
            # The target view's vignetting video doubles as the point cloud's phase 2 color.
            point_cloud['vignettingVideo'] = f'videos/{target_stem}_vignetting.mp4'

    # Optional Draco-compressed surface mesh of the reconstruction.
    mesh_src = Path(__file__).resolve().parent.parent / 'mesh.ply'
    mesh_rel = None
    if mesh_src.exists():
        from plyfile import PlyData
        ply = PlyData.read(str(mesh_src))
        mverts = np.column_stack([ply['vertex']['x'], ply['vertex']['y'],
                                  ply['vertex']['z']]).astype(np.float32)
        mfaces = np.vstack(ply['face']['vertex_indices']).astype(np.uint32)
        (out / 'mesh.drc').write_bytes(
            DracoPy.encode(mverts, mfaces, quantization_bits=14, compression_level=7))
        mesh_rel = 'mesh.drc'

    manifest = {
        'fps': WEB_FPS,
        'numFrames': n_saved,
        'phased': phased,
        'totalFrames': scene.frame_end + 1,
        'views': views,
        'light': light_data,
        'pointCloud': point_cloud,
        'mesh': mesh_rel,
    }
    (out / 'scene.json').write_text(json.dumps(manifest, indent=1))

    # Copy the static viewer files from the repository web/ directory.
    web_src = Path(__file__).resolve().parent.parent / 'web'
    for fname in ('index.html', 'viewer.js'):
        shutil.copy2(web_src / fname, out / fname)

    print(f'Exported web viewer to {out} '
          f'({len(views)} view(s), point cloud: {"yes" if point_cloud else "no"}).')


def main():
    config = parse_config()
    clear_scene()
    print('Cleared the scene.')
    cameras, images = read_colmap_model(config['model_dir'])
    print(f'Loaded COLMAP model: {len(cameras)} camera(s), {len(images)} image(s).')

    # Select which images to place: the saved views, or the whole model.
    target_stem = None
    if config['views_dir']:
        views_dir = Path(config['views_dir'])
        target_stem = views_dir.name[:-len('_views')] if views_dir.name.endswith('_views') else None
        view_stems = sorted(d.name for d in views_dir.iterdir() if d.is_dir())
        stem_to_image = {Path(image['name']).stem: image for image in images.values()}
        selected, missing = [], []
        for stem in view_stems:
            (selected if stem in stem_to_image else missing).append(stem_to_image.get(stem, stem))
        if missing:
            print(f'WARNING: {len(missing)} view(s) had no matching COLMAP image: {missing}')
        print(f'Placing cameras for {len(selected)} saved view(s) from {views_dir}.')
    else:
        selected = sorted(images.values(), key=lambda im: im['name'])
        print(f'Placing cameras for all {len(selected)} model image(s).')

    # Resolve the target image and apply the scene render settings up front, so the
    # camera view frames (hence the image planes) are built with the right aspect ratio.
    target_image = next((im for im in images.values()
                         if Path(im['name']).stem == target_stem), None)
    if target_image is None and selected:
        target_image = selected[0]
    if target_image is not None:
        apply_render_settings(cameras[target_image['camera_id']])

    # Map optimization steps to timeline frames from the target view's reconstructions.
    # When illumination data is present the timeline has two phases of num_frames steps
    # each: reconstruction, then projected illumination (replaying the same optimization).
    num_frames = 1
    phased = False
    if config['views_dir'] and target_image is not None:
        target_view_dir = Path(config['views_dir']) / Path(target_image['name']).stem
        iterations = [it for it, _ in find_image_sequence(target_view_dir, 'reconstruction')]
        if iterations:
            num_frames = max(iterations) + 1
            phased = bool(find_image_sequence(target_view_dir, 'vignetting'))
            last_frame = (2 * num_frames if phased else num_frames) - 1
            scene = bpy.context.scene
            scene.frame_start, scene.frame_end, scene.frame_current = 0, last_frame, 0
            if len(iterations) != num_frames:
                print(f'WARNING: reconstruction steps are not contiguous '
                      f'({len(iterations)} of {num_frames}); the image sequence will have gaps.')
            phase_desc = ('two phases (reconstruction, then projected illumination)'
                          if phased else 'one phase (reconstruction)')
            print(f'Timeline: {phase_desc}, frames 0..{last_frame}.')

    collection = get_clean_collection(config['collection'])
    fx_fy_warned = False
    for image in selected:
        camera = cameras[image['camera_id']]
        fx, fy, cx, cy = pinhole_params(camera)
        if not fx_fy_warned and abs(fx - fy) / max(fx, fy) > 0.01:
            print('WARNING: fx != fy; non-square pixels are only exact for the target view.')
            fx_fy_warned = True
        name = Path(image['name']).stem
        obj = create_camera(name, collection, config)
        apply_intrinsics(obj.data, fx, fy, cx, cy, camera['width'], camera['height'],
                         config['sensor_width'])
        obj.matrix_world = camera_to_world(image['qvec'], image['tvec'])
        # Image plane coinciding with the camera's view frame. Over the two timeline phases
        # it shows this view's reconstruction, then its projected illumination pattern.
        if config['views_dir']:
            view_dir = Path(config['views_dir']) / name
            reconstructions = find_image_sequence(view_dir, 'reconstruction')
            vignettings = find_image_sequence(view_dir, 'vignetting')
            if not reconstructions:
                print(f'WARNING: no reconstruction image for view "{name}".')
            else:
                if vignettings:
                    material = make_phased_material(f'{name}_plane', reconstructions,
                                                    vignettings, num_frames)
                else:
                    material = make_image_material(f'{name}_plane', reconstructions, num_frames)
                collection.objects.link(make_image_plane(
                    f'{name}_plane', obj, material, config['camera_display_size']))

    # Optimized light, modeled as a camera parented to the target camera.
    if config['views_dir'] and target_image is not None:
        target_stem_name = Path(target_image['name']).stem
        light_dir = Path(config['views_dir']).parent / f'{target_stem_name}_light'
        target_camera = collection.objects.get(target_stem_name)
        if (light_dir / 'poses.json').exists() and target_camera is not None:
            create_light(f'{target_stem_name}_light', collection, target_camera,
                         cameras[target_image['camera_id']], light_dir, num_frames, config)
            print(f'Added light camera from {light_dir}.')
        else:
            print(f'No light data at {light_dir}; skipping light.')

    # Unprojected restored RGB image, as a 3D point cloud; phase 2 recolors it with the
    # target view's projected light pattern.
    if config['views_dir'] and target_image is not None:
        target_stem_name = Path(target_image['name']).stem
        pointcloud_dir = Path(config['views_dir']).parent / f'{target_stem_name}_pointcloud'
        target_view_dir = Path(config['views_dir']) / target_stem_name
        if (pointcloud_dir / 'geometry.npz').exists():
            create_point_cloud(f'{target_stem_name}_pointcloud', collection,
                               pointcloud_dir, target_view_dir, num_frames, config)
            print(f'Added point cloud from {pointcloud_dir}.')
        else:
            print(f'No point-cloud data at {pointcloud_dir}; skipping point cloud.')

    cameras_in_scene = [obj for obj in collection.objects if obj.type == 'CAMERA']
    if target_image is not None:
        target_name = Path(target_image['name']).stem
        bpy.context.scene.camera = collection.objects.get(target_name)
        print(f'Active camera and render resolution set from "{target_name}".')

    n_planes = len([obj for obj in collection.objects
                    if obj.type == 'MESH' and obj.name.endswith('_plane')])
    print(f'Done: {len(cameras_in_scene)} camera(s) and {n_planes} image plane(s) '
          f'in collection "{config["collection"]}".')

    # Save the resulting scene to a .blend file.
    output_blend = config['output_blend']
    if not output_blend:
        if config['views_dir']:
            views_dir = Path(config['views_dir'])
            output_blend = views_dir.parent / f'{target_stem or views_dir.name}.blend'
        else:
            output_blend = Path.cwd() / 'sucre_cameras.blend'
    output_blend = Path(output_blend).resolve()
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    # relative_remap stores image paths relative to the .blend, so the textures resolve
    # on any machine as long as the whole outputs/ folder is kept together.
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend), check_existing=False,
                                relative_remap=True)
    print(f'Saved {output_blend}.')

    # Optionally export a self-contained interactive web viewer of the same scene.
    if config['export_web'] and config['views_dir']:
        export_web_bundle(collection, config, target_stem, num_frames, phased)


if __name__ == '__main__':
    main()
