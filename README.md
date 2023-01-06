# SUCRe

This repository implements **SUCRe** and __Gaussian *Sea-thru*__ presented in
[SUCRe: Leveraging Scene Structure for Underwater Color Restoration](https://arxiv.org/abs/2212.09129).

<img src="https://user-images.githubusercontent.com/74063264/208756825-6c9ca9db-e75e-48ee-8ae3-6fb632ad3d02.gif" width="100%" alt=""/>

Both methods make use of the scene's structure to restore colors of underwater images.
- __Gaussian *Sea-thru*__ relies on a single image and its corresponding distance map to estimate the parameters
of an underwater image formation model and then revert the model to restore the image.
- **SUCRe** uses multiple images alongside their 6-DoF pose and intrinsics parameters to simultaneously estimate
the parameters of an underwater image formation model and the restored image.

## Requirements
SUCRe requires Python >=3.10 with modules specified in [environment.yml](environment.yml).
With [Anaconda](https://www.anaconda.com/) installed, one can build the environment by simply running:
```bash
conda env create -f environment.yml
conda activate sucre
```
SUCRe also requires [PyTorch](https://pytorch.org/) which must be installed manually for GPU compatibility.

## Inputs
Both scripts take as inputs the paths of three directories:
- A directory containing the undistorted underwater images.
- A directory containing the depth maps of the images in single channel PNG format. Depth maps should be encoded on 
16 bits and depth information should be in millimeters. They should have the same name as their corresponding
images preceded by `depth_`. For example, the depth map of `image001.jpg` should be named `depth_image001.png`. 
- A directory containing the [COLMAP model](https://colmap.github.io/format.html) of the scene with all cameras
expressed in a `PINHOLE` camera model.

All of this information about the scene can be retrieved with the following pipeline:
1. Run [COLMAP](https://colmap.github.io/) on your set of underwater images.
2. Run COLMAP's `image_undistorter` to retrieve the undistorted images as well as the COLMAP
model with PINHOLE camera model.  
3. Build a 3D mesh from the undistorted COLMAP model using [OpenMVS](https://github.com/cdcseacave/openMVS/).  
4. Use [Maxime Ferrera's raycasting repository](https://github.com/ferreram/depth_map_2_mesh_ray_tracer/) to recover the depth maps from the undistorted COLMAP model and the 3D mesh.

## Gaussian *Sea-thru*
Run Gaussian *Sea-thru* on images of your dataset:
```bash
python gaussian_seathru.py
    --image-dir <path to directory containing underwater images>
    --depth-dir <path to directory containing depth maps>
    --model-dir <path to directory containing COLMAP model>
    --output-dir <path to output directory>
    --image-name <name of image to restore>
```
Optional flags:
- `--linear-beta` switches to SUCRe image formation model.
- `--device` sets the device on which the optimization is performed. It defaults to `cpu`
but we suggest using `cuda` if available to speed the process.

## SUCRe
Run SUCRe on images of your dataset:
```bash
python sucre.py
    --image-dir <path to directory containing underwater images>
    --depth-dir <path to directory containing depth maps>
    --model-dir <path to directory containing COLMAP model>
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
- `--min-cover` sets the minimum percentile of shared observations to keep the pairs of an image.
- `--filter-images-path` is a path to a .txt file with names of images to discard during image matching.
- `--initialization` defines how the parameters of the underwater image formation model and the restored image are
initialized. It defaults to `global`, which uses dark and bright channel priors on all images of the dataset.
Parameters can also be initialized with Gaussian *Sea-thru* using only the image to be restored (`single-view`)
or all matched images (`multi-view`).
- `--solver` specifies which optimization method to use for solving SUCRe's least square.
It defaults to Adam optimizer (`adam`). Other available options are Levenberg-Marquardt (`lm`) and
Nelder-Mead simplex algorithm (`simplex`).
- `--max-iter` sets the maximum number of optimization iterations.
- `--function-tolerance` defines the absolute cost change threshold under which to stop the optimization.
This option has no effects for Adam optimizer.
- `--batch-size` defines the batch size for Adam optimizer.
A larger batch size leads to a faster optimization but requires more memory.
This option has no effects on the resulting image or image formation model parameters.
- `--outliers` specifies the method to filter outliers before normalizing the restored image.
Low intensity bounds of the restored image are retrieved with dark channel prior using either
a single image (`single-view`, default) or all matched images (`multi-view`).
If `--initialization`  is `global`, these bounds were also computed using all images (`global`).
- `--force-compute-matches` forces SUCRe to recompute matches even if the matches file of the image already exists.
- `--keep-matches` prevents SUCRe from deleting the matches file at the end of the optimization.
**Warning:** this can take a lot of space.
- `--num-workers` sets the number of threads to load images.
It defaults to 0, which means images are loaded in the main thread.
We highly suggest you set this parameter to half the number of threads you have available.
- `--device` sets the device for heavy computation.
It defaults to `cpu`. **We highly recommend** using `cuda` if available.
Restoring one image on CPU can take up to several hours.

Example of .txt files for flags `--image-list` and `--filter-images-path`:
```text
someimage.png
anotherimage.png
...
```

## BibTex citation
Please consider citing our work if you use any code from this repository or ideas presented in the paper:

```
@misc{boittiaux2022sucre,
  author={Boittiaux, Cl\'ementin and
          Marxer, Ricard and
          Dune, Claire and
          Arnaubec, Aur\'elien and
          Ferrera, Maxime and
          Hugel, Vincent},
  title={SUCRe: Leveraging Scene Structure for Underwater Color Restoration},
  publisher={arXiv},
  year={2022}
}
```