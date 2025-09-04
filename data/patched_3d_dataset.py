import numpy as np
import tifffile
import logging
from .SliceBuilder import build_slices_3d
from .base_dataset_3d import BaseDataset3D, get_transform
from .image_folder import make_dataset
import math
from util.util import calculate_padding


class Patched3dDataset(BaseDataset3D):
    """
    Dataset for loading and patching 3D images for inference.

    Loads images from a specified path, applies patching into 3D patches with stride depending on the stitching mode and network depth.
    Used during inference for the test_3D.py and test_3D_resnet.py scripts.
    Can be called during inference using the flag --test_mode 3d.
    """

    def __init__(self, opt):
        """
        Initialize the Patched3dDataset.

        Parameters
        ----------
        opt : Option class
            Stores all the experiment flags; needs to be a subclass of BaseOptions.
        """
        super().__init__(opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        self.transform = get_transform(opt)
        self.patch_size = np.asarray([opt.patch_size, opt.patch_size, opt.patch_size])
        self.stride = self.patch_size
        if opt.stitch_mode == "tile-and-stitch":
            difference = 0
            for i in range(2, int(math.log(int(opt.netG[5:]), 2)) + 2):
                difference += 2**i
            stride = opt.patch_size - difference - 2
            self.stride = np.asarray([stride, stride, stride])
        elif opt.stitch_mode == "overlap-averaging":
            self.stride = self.patch_size - opt.patch_overlap
        assert self.patch_size.all() >= self.stride.all(), (
            f"Images can only be stitched if patch size is at least equal to stride, but not smaller. "
            f"Given patch size is {self.patch_size} and stride {self.stride}. That won't work."
        )
        if opt.stitch_mode == "tile-and-stitch":
            self.init_padding = ((self.patch_size - self.stride) / 2).astype(int)
        else:
            self.init_padding = np.asarray([0, 0, 0])

    def build_patches(self, image_path, patch_size, stride):
        """
        Convert a 3D volume into blocks (patches) for inference.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        patch_size : tuple
            Size of the patch (depth, height, width).
        stride : tuple
            Stride for patch extraction (depth, height, width).

        Returns
        -------
        patches : list
            List of patches extracted from the image.
        img_sizes : tuple
            Tuple of (raw image size, padded image size).
        """
        A_img_full = tifffile.imread(image_path)
        if len(A_img_full.shape) > 3:
            A_img_full = np.squeeze(A_img_full)
        A_img_size_raw = A_img_full.shape
        if self.opt.netG.startswith("unet"):
            z1, y1, x1 = calculate_padding(
                A_img_size_raw,
                init_padding=self.init_padding,
                input_patch_size=patch_size,
                stride=stride,
                dim=None,
            )
            init_padding_param = int(self.init_padding[0])
            A_img_full = np.pad(
                A_img_full,
                pad_width=(
                    (init_padding_param, z1),
                    (init_padding_param, y1),
                    (init_padding_param, x1),
                ),
                mode="reflect",
            )
        slices = build_slices_3d(A_img_full, patch_size, stride)
        A_img_size_pad = A_img_full.shape
        img_sizes = (A_img_size_raw, A_img_size_pad)
        patches = []
        for slice_ in slices:
            A_img_patch = A_img_full[slice_]
            A_img_patch = np.expand_dims(A_img_patch, 0)
            patches.append(A_img_patch)
        return patches, img_sizes

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
            - 'A': List of patches for the image
            - 'A_paths': Path to the image
            - 'A_full_size_raw': Raw image size
            - 'A_full_size_pad': Padded image size
        """
        patches, img_sizes = self.build_patches(self.A_paths[index], self.patch_size, self.stride)
        A_path = self.A_paths[index]
        A_size_raw = img_sizes[0]
        A_size_pad = img_sizes[1]
        return {
            "A": patches,
            "A_paths": A_path,
            "A_full_size_raw": A_size_raw,
            "A_full_size_pad": A_size_pad,
        }

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.A_paths)
