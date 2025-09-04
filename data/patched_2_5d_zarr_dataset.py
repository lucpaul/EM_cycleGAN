import math
import numpy as np
import tifffile
import logging
from .SliceBuilder import build_slices, _gen_indices
from .base_dataset_2d import BaseDataset2D, get_transform
from .image_folder import make_dataset, make_zarr_dataset
import os
import zarr
import torch
from util.util import calculate_padding


class Patched25dZarrDataset(BaseDataset2D):
    """
    Dataset for loading and patching images in Zarr format for 2.5D inference.

    Loads images from a specified path, applies patching in three dimensions, and is used during inference for the test_2_5d.py script.
    This class loads the dataset lazily (batch-by-batch), reducing memory usage, but may be slower than the 2.5D dataset for .tif files.
    If you have tif files and want lazy loading, you can also use this dataset, as it can convert tif files to Zarr.
    """

    def __init__(self, opt):
        """
        Initialize the Patched25dZarrDataset.

        Parameters
        ----------
        opt : Option class
            Stores all the experiment flags; needs to be a subclass of BaseOptions.
        """
        super().__init__(opt)
        self.transform = get_transform(opt)
        self.patch_size = np.asarray([opt.patch_size, opt.patch_size, opt.patch_size])
        self.stride = self.patch_size
        if opt.stitch_mode == "tile-and-stitch":
            difference = 0
            for i in range(2, int(math.log(int(opt.netG[5:]), 2)) + 2):
                difference += 2**i
            stride = opt.patch_size - difference - 2
            self.stride = np.asarray([stride, stride, stride])
        elif opt.stitch_mode.startswith("overlap-averaging"):
            self.stride = self.patch_size - opt.patch_overlap

        assert self.patch_size.all() >= self.stride.all(), (
            f"Images can only be stitched if patch size is at least equal to stride, but not smaller. "
            f"Given patch size is {self.patch_size} and stride {self.stride}. That won't work."
        )
        if opt.stitch_mode == "tile-and-stitch":
            self.init_padding = ((self.patch_size - self.stride) / 2).astype(int)
        else:
            # Init padding should be 0 for resnet, or padded unet.
            self.init_padding = np.asarray([0, 0, 0])

        if os.listdir(opt.dataroot)[0].endswith(".tif"):
            self.is_tiff = True
            self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
            self.tif_file_shape = convert_tifs_to_zarrs(self.A_paths, self.init_padding, self.patch_size, self.stride)
        else:
            self.is_tiff = False
            self.A_paths = sorted(make_zarr_dataset(opt.dataroot, opt.max_dataset_size))

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Parameters
        ----------
        index : int
            Index for data access.

        Returns
        -------
        dict
            Dictionary containing:
            - 'A': Path to the image (Zarr or Tiff converted to Zarr)
            - 'A_full_size_raw': Dimensions of the raw dataset (if Tiff)
            - 'A_full_size_pad': Dimensions of the padded image (if Tiff)
            - 'IsTiff': Boolean indicating if the original file was a Tiff
        """
        A_path = self.A_paths[index]
        if self.is_tiff:
            A_path = os.path.splitext(A_path)[0] + ".zarr"
            A_size_raw = self.tif_file_shape[A_path][0]
            A_size_pad = self.tif_file_shape[A_path][1]
            return {
                "A": A_path,
                "A_full_size_raw": A_size_raw,
                "A_full_size_pad": A_size_pad,
                "IsTiff": self.is_tiff,
            }
        else:
            return {"A": A_path, "IsTiff": self.is_tiff}

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.A_paths)


def compute_patch_indices(dataset, padding, patch_size, stride, direction):
    """
    Compute patch indices after applying padding.

    Parameters
    ----------
    dataset : zarr.Array
        The Zarr dataset to extract patches from.
    padding : tuple
        Padding to apply ((before, after), ... for each dimension).
    patch_size : tuple
        Size of the patch (height, width).
    stride : tuple
        Stride for patch extraction (height, width).
    direction : str
        Direction of slicing ('xy', 'zx', or 'zy').

    Returns
    -------
    tuple
        (patch_indices, patches_per_slice, raw_shape, padded_shape)
    """
    logging.info("Padding zarr dataset: %s", padding)
    S, H, W = dataset.shape
    if direction == "xy":
        padded_S = S  # + padding[0][0] + padding[0][1]
        padded_H = H + padding[1][0] + padding[1][1]
        padded_W = W + padding[2][0] + padding[2][1]
    elif direction == "zx":
        padded_S = W  # + padding[2][0] + padding[2][1]
        padded_H = S + padding[0][0] + padding[0][1]
        padded_W = H + padding[1][0] + padding[1][1]
    elif direction == "zy":
        padded_S = H  # + padding[1][0] + padding[1][1]
        padded_H = S + padding[0][0] + padding[0][1]
        padded_W = W + padding[2][0] + padding[2][1]

    patch_indices = []
    logging.info("dimensions: %s %s", direction, dataset.shape)
    # for s in range(padded_S):
    for s in range(padded_S):
        h_indices = list(_gen_indices(padded_H, patch_size[0], stride[0]))
        for h in h_indices:
            w_indices = list(_gen_indices(padded_W, patch_size[1], stride[1]))
            for w in w_indices:
                patch_indices.append((s, h, w))
    logging.info("padded: %s", (padded_S, padded_H, padded_W))
    return (
        patch_indices,
        len(h_indices) * len(w_indices),
        (S, H, W),
        (padded_S, padded_H, padded_W),
    )


def convert_tifs_to_zarrs(
    tif_files,
    init_padding,
    patch_size,
    stride,
    chunk_size=(256, 256, 256),
    compression_level=5,
):
    """
    Convert a list of 3D .tif files into Zarr format with balanced chunk sizes.

    Parameters
    ----------
    tif_files : list of str
        Paths to .tif files.
    init_padding : tuple
        Initial padding to apply.
    patch_size : tuple
        Patch size for chunking.
    stride : tuple
        Stride for patching.
    chunk_size : tuple, optional
        Chunk size for storage optimization (default: (96, 96, 96)).
    compression_level : int, optional
        Compression level for Blosc (default: 5).

    Returns
    -------
    dict
        Metadata mapping zarr_path to (original_shape, padded_shape).
    """
    metadata = {}  # Store original shape information
    for tif_file in tif_files:
        filename = os.path.splitext(os.path.basename(tif_file))[0]
        zarr_path = os.path.join(os.path.dirname(tif_file), filename + ".zarr")
        with tifffile.TiffFile(tif_file) as tif:
            img = tif.asarray()  # Load full 3D image
        padding_d, padding_h, padding_w = calculate_padding(img.shape, init_padding, patch_size, stride, dim=None)
        logging.info("Tiff shape: %s", img.shape)
        pad_img_shape = (
            img.shape[0] + init_padding[2] + padding_d,
            img.shape[1] + init_padding[1] + padding_h,
            img.shape[2] + init_padding[0] + padding_w,
        )
        metadata[zarr_path] = (img.shape, pad_img_shape)
        # Create a Zarr dataset with balanced chunk sizes
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.open(
            store,
            mode="w",
            shape=img.shape,
            dtype=img.dtype,
            chunks=chunk_size,
            compressor=zarr.Blosc(cname="zstd", clevel=compression_level),
        )
        root[:] = img  # Store full volume
    return metadata  # Return original dimensions for later use
