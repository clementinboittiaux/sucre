# SUCRe: Leveraging Scene Structure for Underwater Color Restoration

[Paper](https://arxiv.org/abs/2212.09129) • [Project page](https://clementinboittiaux.github.io/sucre/)

This repository implements [SUCRe](https://arxiv.org/abs/2212.09129), a multi-view method that uses a dense scene
reconstruction for underwater color restoration.

<img src="https://user-images.githubusercontent.com/74063264/208756825-6c9ca9db-e75e-48ee-8ae3-6fb632ad3d02.gif" width="100%" alt=""/>

This implementation requires undistorted underwater images along with their corresponding depth maps and a
[COLMAP](https://colmap.github.io/) model of the scene.

## Installation
SUCRe requires Python >=3.10 and [PyTorch](https://pytorch.org/)>=2.1.0. [PyTorch](https://pytorch.org/) should be
installed independently for GPU compatibility. All other modules are listed in [requirements.txt](requirements.txt) and
can be installed using pip: 
```bash
pip install -r requirements.txt
```

## Input data
The [sucre.py](sucre/sucre.py) script takes as input three directories:
- A directory containing the undistorted underwater images.
- A directory containing the depth maps.
- A directory containing the [COLMAP](https://colmap.github.io/format.html) model of the scene.

### Data format
The directory containing undistorted underwater images can be arbitrarily organized:
```
images
├─ image001.jpg
├─ image002.jpg
└─ ...
```

Depth maps should be encoded in 16-bits single channel PNG format. The depth information should be stored in
millimeters. Depth maps should have the same name as their corresponding images preceded by `depth_`. For example, the
depth map of `image001.jpg` should be named `depth_image001.png`:
```
depths
├─ depth_image001.png
├─ depth_image002.png
└─ ...
```

The [COLMAP](https://colmap.github.io/format.html) model consists of three files in either `.bin` or `.txt` format. Because underwater images are undistorted,
all intrinsic parameters are expected to use a `PINHOLE` camera model:
```
sparse
├─ cameras.bin
├─ images.bin
└─ points3D.bin
```

### Data processing pipeline
All the data required to run SUCRe **can be retrieved from you own set of underwater images** by following this pipeline:
1. Run [COLMAP](https://colmap.github.io/) on your set of underwater images.
2. Run `colmap image_undistorter` to retrieve the undistorted images and
[COLMAP](https://colmap.github.io/format.html) model.  
3. Build a dense 3D mesh from the undistorted [COLMAP](https://colmap.github.io/format.html) model using
[OpenMVS](https://github.com/cdcseacave/openMVS/).  
4. Use [Maxime Ferrera's script](https://github.com/ferreram/depth_map_2_mesh_ray_tracer/) to compute depth maps from the
undistorted [COLMAP](https://colmap.github.io/format.html) model and the 3D mesh.

Following these steps will ensure you have all the necessary information in the correct format. 

## Usage
To run SUCRe on your own set of images, simply run the [sucre.py](sucre/sucre.py) script with the following arguments:
```bash
python sucre.py
    --image-dir <path to the directory containing the undistorted underwater images>
    --depth-dir <path to the directory containing the depth maps>
    --model-dir <path to the directory containing the undistorted COLMAP model>
    --output-dir <path to output directory>
    (
        --image-name <name of image to restore>
        or
        --image-list <path to a .txt file with names of images to restore>
        or
        --image-ids <min image id> <max image id>
    )
```

The `--image-list` flag takes as input a `.txt` file with the following format:
```text
image001.jpg
image012.jpg
...
```

Other options are available with documentation by running the script with the `-h` flag:
```bash
python sucre.py -h
```

## BibTeX citation
Please consider citing our work if you use any code from this repository or ideas presented in the paper:

```
@inproceedings{boittiaux2024sucre,
    author={Boittiaux, Cl\'ementin and Marxer, Ricard and Dune, Claire and Arnaubec, Aur\'elien and Ferrera, Maxime and Hugel, Vincent},
    title={{SUCRe}: Leveraging Scene Structure for Underwater Color Restoration},
    booktitle={3DV},
    year={2024}
}
```
