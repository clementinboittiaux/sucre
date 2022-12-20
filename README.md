# SUCRe

<p align="center">
    <img src="https://user-images.githubusercontent.com/74063264/208756825-6c9ca9db-e75e-48ee-8ae3-6fb632ad3d02.gif" width="100%"/>
</p>

This code implements both Gaussian *Sea-thru* and SUCRe methods.

## Disclaimer
**This README is not up-to-date**, however the `main` branch always contains the latest unstable release.
Run scripts with `-h` command to see available options.

## Requirements
SUCRe requires Python >=3.10 with modules specified in [environment.yml](environment.yml).
With [Anaconda](https://www.anaconda.com/) installed, one can build the environment by simply running:
```bash
conda env create -f environment.yml
conda activate sucre
```

## Datasets
Both Gaussian *Sea-thru* and SUCRe take a dataset directory as input. A dataset directory must include three
directories: `depth_maps`, `images` and `sparse`. The structure of the dataset directory should look like this:
```
dataset
├── images
│   ├── someimage.ext
│   ├── anotherimage.ext
│   └── ...
├── depth_maps
│   ├── depth_someimage.png
│   ├── depth_anotherimage.png
│   └── ...
└── sparse
    ├── cameras.bin
    └── images.bin
```
where
- `images` contains undistorted original underwater images.
- `depth_maps` contains the corresponding depth maps in single channel PNG format. Depth maps should be encoded on 
16 bits and depth information should be in millimeters. They should have the same name as their corresponding
images preceded by *"depth_"*.
- `sparse` contains a COLMAP model with all cameras expressed in a **PINHOLE** camera model.

All of this information can be retrieved with the following pipeline:
1. Run [COLMAP](https://colmap.github.io/) on your set of images
2. Run COLMAP's `image_undistorter` to retrieve `images` and `sparse` directories.  
3. Build a 3D mesh from the undistorted COLMAP model using [OpenMVS](https://github.com/cdcseacave/openMVS/).  
4. Use [Maxime Ferrera's raycasting repository](https://github.com/ferreram/depth_map_2_mesh_ray_tracer/) to recover the depth maps from the undistorted COLMAP model and the 3D mesh.

## Gaussian *Sea-thru*
To run Gaussian *Sea-thru* on your dataset:
```bash
python gaussian_seathru.py
    --data-dir <path to dataset directory>
    --output-dir <path to output directory>
    --image-name <name of image to restore>
```
Optional flags:
- `--linear-beta`, `--no-linear-beta` switch between SUCRe model (`--linear-beta`)
and *Sea-thru* model (`--no-linear-beta`). By default, *Sea-thru* model is used.
- `device` sets the device on which the optimization is performed. It defaults to 'cpu'
but we suggest using 'cuda' if available to speed the process.

## SUCRe
To run SUCRe on your dataset:
```bash
python sucre.py
    --data-dir <path to dataset directory>
    --output-dir <path to output directory>
    (
        --image-name <name of image to restore>
        or
        --image-list <path to a .txt file with names of images to restore>
        or
        --image-ids <min image id> <max image id>
    )
```
Optional flags:
- `--voxel-size` <voxel size> sets the size of voxels for faster image matching. A smaller value means faster matching
but requires more memory.
- `--min-cover` sets the minimum percentile of shared observations to keep the pairs of an image.
- `--filter-images-path` is a path to a .txt file with names of images to discard during image matching.
- `--num-workers` sets the number of threads to load images.
It defaults to 0, which means images are loaded in the main thread.
We highly suggest you set this parameter to half the number of threads you have available.
- `--device` sets the device on which to compute voxels, matches and image restoration.
It defaults to 'cpu'. **We highly recommend using 'cuda'** if available.
On CPU, restoring one image can take up to several hours.

Example of .txt files for flags `--image-list` and `--filter-images-path`:
```text
someimage.png
anotherimage.png
...
```
