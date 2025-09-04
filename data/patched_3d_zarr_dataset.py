import math
import numpy as np
import tifffile
import logging
from .SliceBuilder import _gen_indices
from .base_dataset_3d import BaseDataset3D, get_transform
from .image_folder import make_dataset, make_zarr_dataset
import os
import zarr
from util.util import calculate_padding


class Patched3dZarrDataset(BaseDataset3D):
    """
    Dataset for loading and patching 3D images in Zarr format for inference.

    Loads images from a specified path, applies patching for each z-slice of a stack with stride depending on the stitching mode and network depth.
    Used during inference for the test_3D.py and test_3D_resnet.py scripts.
    Can be called during inference using the flag --test_mode 3d.
    """

    def __init__(self, opt):
        """
        Initialize the Patched3dZarrDataset.

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
            self.init_padding = np.asarray([0, 0, 0])

        if os.listdir(opt.dataroot)[0].endswith(".tif"):
            self.is_tiff = True
            self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
            logging.info("Found TIFF files. Converting to zarr")
            self.tif_file_shape = convert_tifs_to_zarrs(
                self.A_paths,
                self.init_padding,
                self.patch_size,
                self.stride,
                chunk_size=(self.patch_size[0], self.patch_size[1], self.patch_size[2]),
            )
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


def compute_patch_indices(dataset, padding, patch_size, stride):
    """
    Computes patch indices after applying padding.
    """
    patch_indices = []
    D, H, W = dataset.shape  # (slices, height, width)
    padded_D = D + padding[0] + padding[1]
    padded_H = H + padding[2] + padding[3]  # Account for padding
    padded_W = W + padding[4] + padding[5]

    d_indices = _gen_indices(padded_D, patch_size[0], stride[0])
    for d in d_indices:  # Iterate over slices
        h_indices = _gen_indices(padded_H, patch_size[1], stride[1])
        for h in h_indices:
            w_indices = _gen_indices(padded_W, patch_size[2], stride[2])
            for w in w_indices:
                patch_indices.append((d, h, w))

    return patch_indices, (D, H, W), (padded_D, padded_H, padded_W)


def convert_tifs_to_zarrs(
    tif_files,
    init_padding,
    patch_size,
    stride,
    chunk_size=(256, 256, 256),
    compression_level=5,
):
    """
    Converts a list of 3D .tif files into Zarr format.

    Args:
        tif_files (list of str): Paths to .tif files.
        zarr_dir (str): Directory to save the Zarr datasets.
        chunk_size (tuple): Chunk size for storage optimization.
        compression_level (int): Compression level for Blosc.
    """

    metadata = {}  # Store original shape information

    for tif_file in tif_files:
        filename = os.path.splitext(os.path.basename(tif_file))[0]
        zarr_path = os.path.join(os.path.dirname(tif_file), filename + ".zarr")

        with tifffile.TiffFile(tif_file) as tif:
            img = tif.asarray()  # Load full 3D image

        padding_d, padding_h, padding_w = calculate_padding(img.shape, init_padding, patch_size, stride, dim=None)

        pad_img_shape = (
            img.shape[0] + init_padding[0] + padding_d,
            img.shape[1] + init_padding[1] + padding_h,
            img.shape[2] + init_padding[2] + padding_w,
        )
        # Save original shape
        metadata[zarr_path] = (img.shape, pad_img_shape)

        # Create a Zarr dataset
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
