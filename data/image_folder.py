"""
A modified image folder class.

This module modifies the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both the current directory and its subdirectories.
"""

import os
from glob import glob

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
    ".nii",
]


def is_image_file(filename):
    """
    Check if a file is an allowed image extension.

    Args:
        filename (str): Path to a file.

    Returns:
        bool: True if the file is an image, False otherwise.
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    """
    Create a dataset by collecting image file paths from a directory and its subdirectories.

    Args:
        dir (str): Root directory path.
        max_dataset_size (int or float): Maximum number of images to include.

    Returns:
        list: List of image file paths.
    """
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[: min(max_dataset_size, len(images))]


def make_zarr_dataset(dir, max_dataset_size=float("inf")):
    """
    Create a dataset by collecting .zarr file paths from a directory.

    Args:
        dir (str): Root directory path.
        max_dataset_size (int or float): Maximum number of .zarr files to include.

    Returns:
        list: List of .zarr file paths.
    """
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    images = glob(os.path.join(dir, "*.zarr"))
    return images[: min(max_dataset_size, len(images))]
