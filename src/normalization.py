import numpy as np
from torch import Tensor


def histogram_stretching(image: Tensor):
    image = image.numpy()
    valid = np.all(~np.isnan(image), axis=2)
    image_valid = image[valid]
    image_valid = np.clip(image_valid, np.percentile(image_valid, 1, axis=0), np.percentile(image_valid, 99, axis=0))
    image_valid = image_valid - np.min(image_valid, axis=0)
    image_valid = image_valid / np.max(image_valid, axis=0)
    image[~valid] = 0
    image[valid] = image_valid
    return image
