"""Build a self-contained Three.js viewer of a SUCRe optimization run.

Reads SUCRe outputs (a ``<target>_views`` directory and its sibling ``_light``
and ``_pointcloud`` directories) and a COLMAP model, encodes per-step JPEGs to
H.265 .mp4s, compresses the mesh and point cloud with Draco, and writes a
deployable bundle (``scene.json`` + ``*.drc`` + ``videos/*.mp4`` + ``index.html``
+ ``viewer.js``) to the given output directory.

Run from the sucre conda env (needs DracoPy, plyfile, and ffmpeg on PATH):

    python sucre/export_web.py \\
        --model-dir /path/to/colmap/sparse \\
        --views-dir outputs/<target>_views \\
        --export-web outputs/web
"""
import argparse
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


WEB_FPS = 24  # playback / encoding frame rate for the per-step video sequences
# Local keep-rate at the last frame when progressive frame-dropping is enabled
# (see --frame-drop-start); the rate falls linearly from 1.0 to this value.
FRAME_DROP_END_RATE = 1 / 6

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
def read_cameras_bin(path):
    cameras = {}
    with open(path, 'rb') as f:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            camera_id, model_id = struct.unpack('<ii', f.read(8))
            width, height = struct.unpack('<QQ', f.read(16))
            model, num_params = CAMERA_MODELS[model_id]
            params = struct.unpack('<' + 'd' * num_params, f.read(8 * num_params))
            cameras[camera_id] = {'model': model, 'width': width, 'height': height,
                                  'params': params}
    return cameras


def read_images_bin(path):
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


def read_cameras_txt(path):
    cameras = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        e = line.split()
        cameras[int(e[0])] = {'model': e[1], 'width': int(e[2]), 'height': int(e[3]),
                              'params': tuple(float(x) for x in e[4:])}
    return cameras


def read_images_txt(path):
    images = {}
    data = [ln for ln in Path(path).read_text().splitlines()
            if ln.strip() and not ln.startswith('#')]
    for line in data[::2]:  # COLMAP writes 2 lines per image, the 2nd holds 2D points
        e = line.split()
        images[int(e[0])] = {
            'qvec': tuple(float(x) for x in e[1:5]),
            'tvec': tuple(float(x) for x in e[5:8]),
            'camera_id': int(e[8]),
            'name': ' '.join(e[9:]),
        }
    return images


def read_colmap_model(model_dir):
    model_dir = Path(model_dir)
    if (model_dir / 'cameras.bin').exists():
        return read_cameras_bin(model_dir / 'cameras.bin'), read_images_bin(model_dir / 'images.bin')
    if (model_dir / 'cameras.txt').exists():
        return read_cameras_txt(model_dir / 'cameras.txt'), read_images_txt(model_dir / 'images.txt')
    raise FileNotFoundError(f'No COLMAP cameras.bin/.txt found in {model_dir}.')


def pinhole_params(camera):
    """Return (fx, fy, cx, cy) for any COLMAP camera model."""
    model, p = camera['model'], camera['params']
    if model == 'PINHOLE':
        return p[0], p[1], p[2], p[3]
    if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL', 'FOV',
                 'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE'):
        return p[0], p[0], p[1], p[2]
    return p[0], p[1], p[2], p[3]


# --------------------------------------------------------------------------- #
# Geometry helpers (OpenCV camera convention: +X right, +Y down, +Z forward).
# --------------------------------------------------------------------------- #
def quat_to_mat(qvec):
    """COLMAP qvec (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def camera_to_world(qvec, tvec):
    """COLMAP (qvec, tvec) world-to-camera pose -> 4x4 camera-to-world matrix."""
    R_wc = quat_to_mat(qvec)
    R_cw = R_wc.T
    C = -R_cw @ np.asarray(tvec)
    T = np.eye(4)
    T[:3, :3] = R_cw
    T[:3, 3] = C
    return T


def image_plane_corners(fx, fy, cx, cy, width, height, distance):
    """Four corners of the image plane in camera frame at +Z = distance.

    Order matches the viewer's QUAD_UV: TR, BR, BL, TL.
    """
    xl = (0 - cx) / fx * distance
    xr = (width - cx) / fx * distance
    yt = (0 - cy) / fy * distance
    yb = (height - cy) / fy * distance
    return np.array([[xr, yt, distance],   # top-right
                     [xr, yb, distance],   # bottom-right
                     [xl, yb, distance],   # bottom-left
                     [xl, yt, distance]])  # top-left


def transform_points(T, points):
    """Apply 4x4 homogeneous T to (N,3) points."""
    points = np.asarray(points)
    return (T[:3, :3] @ points.T).T + T[:3, 3]


# --------------------------------------------------------------------------- #
# Misc helpers.
# --------------------------------------------------------------------------- #
def find_image_sequence(directory, prefix):
    """Return sorted [(iteration, path), ...] for ``<prefix>_*.jpg`` files."""
    sequence = []
    for path in Path(directory).glob(f'{prefix}_*.jpg'):
        sequence.append((int(path.stem.split('_')[-1]), path.resolve()))
    return sorted(sequence)


def select_kept_frames(n_saved, drop_start, end_rate=FRAME_DROP_END_RATE, power=1.0):
    """Frame positions to keep after smooth, progressive decimation.

    Every frame before ``drop_start`` is kept. From ``drop_start`` to the last
    frame the local keep-rate falls from 1.0 to ``end_rate`` along
    ``(1 - t)**power`` (``t`` is the normalized position past the start), so the
    animation thins out toward the end with no density jump at the boundary.
    ``power=1`` is a linear ramp, which keeps ~half the post-start frames almost
    regardless of ``end_rate``; larger powers drop frames far more aggressively
    (the kept fraction past the start is ``(1 - end_rate)/(power + 1) +
    end_rate``). A deterministic accumulator spreads the kept frames evenly at
    the local rate; the boundary frame is always kept, but the final frame may be
    dropped to keep that spacing consistent. Returns sorted positions into the
    saved sequence.
    """
    last = n_saved - 1
    if drop_start is None or drop_start >= last:
        return list(range(n_saved))
    span = last - drop_start
    kept = list(range(drop_start))
    acc = 0.0
    for i in range(drop_start, n_saved):
        t = (i - drop_start) / span
        acc += end_rate + (1.0 - end_rate) * (1.0 - t) ** power
        if acc >= 1.0:
            kept.append(i)
            acc -= 1.0
    return kept


def encode_video(image_dir, prefix, output, fps=WEB_FPS, scale=1.0, frames=None):
    """Encode ``<prefix>_NNNN.jpg`` in ``image_dir`` to an AV1 .mp4.

    AV1 (libsvtav1) keeps the bundle small while playing in both Firefox and
    Chromium (HEVC does not decode in Chromium on Linux). A 1-second GOP
    (keyint=24) balances seek latency against size: AV1 leans on long-range
    prediction, so a very short GOP bloats the file (keyint=5 was ~3.5x larger
    here) while a 1s GOP still seeks within a few decoded frames at these small
    resolutions. ``+faststart`` puts the moov atom up front so the browser plays
    during download. The glob pattern accepts non-contiguous indices (e.g. from
    ``--save-interval N``). ``scale < 1`` Lanczos-downsamples for the web; output
    dims are rounded even for yuv420p.

    ``frames``, when given, is an ordered list of source frame paths to encode
    instead of every ``<prefix>_*.jpg``; they are staged as contiguously named
    symlinks so progressive frame-dropping plays back as one continuous video.
    """
    image_dir = Path(image_dir)
    vf = f'scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2:flags=lanczos'
    staging = None
    if frames is None:
        glob_path = image_dir / f'{prefix}_*.jpg'
    else:
        staging = Path(tempfile.mkdtemp(prefix='sucre_frames_'))
        for k, src in enumerate(frames):
            (staging / f'frame_{k:05d}.jpg').symlink_to(Path(src).resolve())
        glob_path = staging / 'frame_*.jpg'
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-framerate', str(fps),
        '-pattern_type', 'glob', '-i', str(glob_path),
        '-c:v', 'libsvtav1', '-crf', '32', '-preset', '6',
        '-svtav1-params', 'keyint=24',
        '-pix_fmt', 'yuv420p',
        '-vf', vf,
        '-movflags', '+faststart',
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Bundle builder.
# --------------------------------------------------------------------------- #
def build_bundle(args):
    import DracoPy

    views_dir = Path(args.views_dir).resolve()
    if not views_dir.name.endswith('_views'):
        raise SystemExit(f'--views-dir must end with "_views" (got {views_dir.name}).')
    target_stem = views_dir.name[:-len('_views')]
    base = views_dir.parent
    out = Path(args.export_web).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'videos').mkdir(exist_ok=True)

    cameras, images = read_colmap_model(args.model_dir)
    stem_to_image = {Path(im['name']).stem: im for im in images.values()}
    if target_stem not in stem_to_image:
        raise SystemExit(f'Target "{target_stem}" not found in the COLMAP model.')
    target_image = stem_to_image[target_stem]
    target_camera = cameras[target_image['camera_id']]
    target_T = camera_to_world(target_image['qvec'], target_image['tvec'])

    # Saved iterations come from the target's reconstruction sequence; the rest
    # of the optimization (lights, point cloud) is sampled at the same indices.
    saved_iterations = [it for it, _ in find_image_sequence(views_dir / target_stem,
                                                            'reconstruction')]
    if not saved_iterations:
        raise SystemExit(f'No reconstruction frames in {views_dir / target_stem}.')
    n_saved = len(saved_iterations)
    kept_positions = select_kept_frames(n_saved, args.frame_drop_start,
                                        args.frame_drop_end_rate, args.frame_drop_power)
    kept_iterations = [saved_iterations[i] for i in kept_positions]
    n_frames = len(kept_iterations)
    decimating = n_frames != n_saved
    phased = bool(find_image_sequence(views_dir / target_stem, 'vignetting'))
    print(f'Target "{target_stem}": {n_saved} saved iteration(s), '
          f'phased={phased}.')
    if decimating:
        print(f'Frame-dropping from frame {args.frame_drop_start}: '
              f'{n_saved} -> {n_frames} animation frames '
              f'(keep-rate 1.0 -> {args.frame_drop_end_rate:.3f}, '
              f'power {args.frame_drop_power:g}).')

    def kept_paths(directory, prefix):
        """Source paths for the kept frames of a sequence, or None if not decimating."""
        if not decimating:
            return None
        sequence = find_image_sequence(directory, prefix)
        return [sequence[i][1] for i in kept_positions]

    # ------------------------------------------------------------------- #
    # View cameras: world-space apex + image-plane corners; per-view video.
    # ------------------------------------------------------------------- #
    view_stems = sorted(d.name for d in views_dir.iterdir() if d.is_dir())
    views = []
    fx_fy_warned = False
    for stem in view_stems:
        if stem not in stem_to_image:
            print(f'WARNING: view "{stem}" has no matching COLMAP image; skipping.')
            continue
        image = stem_to_image[stem]
        camera = cameras[image['camera_id']]
        fx, fy, cx, cy = pinhole_params(camera)
        if not fx_fy_warned and abs(fx - fy) / max(fx, fy) > 0.01:
            print('WARNING: fx != fy on at least one camera; image-plane is still '
                  'drawn as a flat quad.')
            fx_fy_warned = True
        T = camera_to_world(image['qvec'], image['tvec'])
        corners_cam = image_plane_corners(fx, fy, cx, cy, camera['width'],
                                          camera['height'], args.plane_distance)
        corners_world = transform_points(T, corners_cam)

        view_dir = views_dir / stem
        recon_out = out / 'videos' / f'{stem}_reconstruction.mp4'
        encode_video(view_dir, 'reconstruction', recon_out, scale=0.5,
                     frames=kept_paths(view_dir, 'reconstruction'))
        view = {
            'name': stem,
            'isTarget': stem == target_stem,
            'apex': T[:3, 3].tolist(),
            'corners': corners_world.tolist(),
            'reconstructionVideo': f'videos/{stem}_reconstruction.mp4',
        }
        if phased and find_image_sequence(view_dir, 'vignetting'):
            # The target's vignetting doubles as the point cloud's phase-2 color,
            # so it stays at full image resolution; the rest go to half scale.
            vign_scale = 1.0 if stem == target_stem else 0.5
            vign_out = out / 'videos' / f'{stem}_vignetting.mp4'
            encode_video(view_dir, 'vignetting', vign_out, scale=vign_scale,
                         frames=kept_paths(view_dir, 'vignetting'))
            view['vignettingVideo'] = f'videos/{stem}_vignetting.mp4'
        views.append(view)
    print(f'Encoded {len(views)} view(s).')

    # ------------------------------------------------------------------- #
    # Light: parented to the target camera; one apex + corners per saved step.
    # ------------------------------------------------------------------- #
    light_data = None
    light_dir = base / f'{target_stem}_light'
    light_poses_path = light_dir / 'poses.json'
    if light_poses_path.exists():
        poses = json.loads(light_poses_path.read_text())['light_to_camera']
        fx, fy, cx, cy = pinhole_params(target_camera)
        light_corners_cam = image_plane_corners(fx, fy, cx, cy, target_camera['width'],
                                                target_camera['height'],
                                                args.plane_distance)
        apexes, corners = [], []
        for it in kept_iterations:
            key = str(it)
            if key not in poses:
                raise SystemExit(f'Light pose missing for iteration {it}.')
            light_to_camera = np.array(poses[key])
            light_T = target_T @ light_to_camera
            apexes.append(light_T[:3, 3].tolist())
            corners.append(transform_points(light_T, light_corners_cam).tolist())
        encode_video(light_dir, 'pattern', out / 'videos' / 'light_pattern.mp4',
                     scale=0.5, frames=kept_paths(light_dir, 'pattern'))
        light_data = {'apex': apexes, 'corners': corners,
                      'patternVideo': 'videos/light_pattern.mp4'}
        print(f'Encoded light pattern; {len(apexes)} pose sample(s).')
    else:
        print(f'No light data at {light_dir}; skipping.')

    # ------------------------------------------------------------------- #
    # Point cloud: Draco-compressed positions + per-point UV.
    # ------------------------------------------------------------------- #
    point_cloud = None
    pc_dir = base / f'{target_stem}_pointcloud'
    if (pc_dir / 'geometry.npz').exists():
        geometry = np.load(pc_dir / 'geometry.npz')
        xyz = geometry['xyz'].astype(np.float32)
        # Point cloud UVs go in via a uint16 generic attribute. DracoPy's
        # `tex_coord` is silently dropped when there are no faces, and float32
        # generics don't compress well; pre-quantizing to uint16 gives ~5x
        # smaller output and a UV precision of 1/65535 (sub-pixel on any
        # realistic video). The string key makes DracoPy auto-assign unique IDs
        # (position=0, uv=1); the JS side reads unique_id=1 into the geometry's
        # `uv` slot and sets `normalized=true` so the shader sees uv in [0, 1].
        uv = geometry['uv']
        uv16 = (uv * 65535.0).round().clip(0, 65535).astype(np.uint16)
        drc = DracoPy.encode(xyz, generic_attributes={'uv': uv16},
                             quantization_bits=args.draco_quantization_bits,
                             compression_level=7)
        (out / 'pointcloud.drc').write_bytes(drc)
        encode_video(pc_dir, 'rgb', out / 'videos' / 'pointcloud_rgb.mp4',
                     frames=kept_paths(pc_dir, 'rgb'))
        point_cloud = {'count': int(len(xyz)), 'data': 'pointcloud.drc',
                       'rgbVideo': 'videos/pointcloud_rgb.mp4'}
        if phased:
            # Target view's vignetting doubles as the point cloud's phase-2 color.
            point_cloud['vignettingVideo'] = f'videos/{target_stem}_vignetting.mp4'
        print(f'Encoded point cloud: {len(xyz)} pts, '
              f'{len(drc) / 1024:.1f} KB at qp={args.draco_quantization_bits}.')
    else:
        print(f'No point-cloud data at {pc_dir}; skipping.')

    # ------------------------------------------------------------------- #
    # Optional Draco-compressed surface mesh.
    # ------------------------------------------------------------------- #
    mesh_rel = None
    mesh_src = Path(args.mesh_ply).resolve() if args.mesh_ply else None
    if mesh_src and mesh_src.exists():
        from plyfile import PlyData
        ply = PlyData.read(str(mesh_src))
        v = ply['vertex']
        mverts = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32)
        mfaces = np.vstack(ply['face']['vertex_indices']).astype(np.uint32)
        drc = DracoPy.encode(mverts, mfaces,
                             quantization_bits=args.draco_quantization_bits,
                             compression_level=7)
        (out / 'mesh.drc').write_bytes(drc)
        mesh_rel = 'mesh.drc'
        print(f'Encoded mesh: {len(mverts)} verts, {len(mfaces)} faces, '
              f'{len(drc) / 1024:.1f} KB.')
    elif args.mesh_ply:
        print(f'Mesh file {mesh_src} not found; skipping mesh.')

    # ------------------------------------------------------------------- #
    # Manifest + static viewer files.
    # ------------------------------------------------------------------- #
    manifest = {
        'fps': WEB_FPS,
        'numFrames': n_frames,
        'phased': phased,
        'views': views,
        'light': light_data,
        'pointCloud': point_cloud,
        'mesh': mesh_rel,
    }
    (out / 'scene.json').write_text(json.dumps(manifest, indent=1))

    web_src = Path(__file__).resolve().parent.parent / 'web'
    for fname in ('index.html', 'viewer.js'):
        shutil.copy2(web_src / fname, out / fname)

    print(f'Exported web viewer to {out}.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-dir', required=True,
                        help='path to the COLMAP model directory (cameras.bin/txt).')
    parser.add_argument('--views-dir', required=True,
                        help='path to a SUCRe <target>_views directory.')
    parser.add_argument('--export-web', required=True,
                        help='output directory for the deployable web bundle.')
    parser.add_argument('--mesh-ply',
                        default=str(Path(__file__).resolve().parent.parent / 'mesh.ply'),
                        help='optional surface mesh .ply (default: <repo>/mesh.ply).')
    parser.add_argument('--plane-distance', type=float, default=1.16,
                        help='distance (in world units) at which to place each '
                             'camera/light image-plane quad in front of its apex.')
    parser.add_argument('--draco-quantization-bits', type=int, default=14,
                        help='Draco quantization bits for positions and UVs '
                             '(14 ≈ 0.6 mm precision over a 10 m scene).')
    parser.add_argument('--frame-drop-start', type=int, default=None,
                        help='animation frame index at which to start progressively '
                             'dropping frames. Before it every frame is kept; from it '
                             'to the last frame the keep-rate falls linearly from 1.0 '
                             'to --frame-drop-end-rate, so the animation speeds up '
                             'toward the end. Default: keep all frames.')
    parser.add_argument('--frame-drop-end-rate', type=float, default=FRAME_DROP_END_RATE,
                        help='keep-rate at the last frame when --frame-drop-start is set '
                             f'(default {FRAME_DROP_END_RATE:.3f}, i.e. keep ~1 frame in 6).')
    parser.add_argument('--frame-drop-power', type=float, default=1.0,
                        help='steepness of the keep-rate falloff after --frame-drop-start. '
                             '1.0 is linear and keeps ~half the post-start frames whatever '
                             'the end-rate; higher values drop frames much faster (kept '
                             'fraction past the start ≈ (1-end_rate)/(power+1) + end_rate).')
    build_bundle(parser.parse_args(argv))


if __name__ == '__main__':
    main()
