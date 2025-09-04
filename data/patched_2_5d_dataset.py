import logging
from .base_dataset_2d import BaseDataset2D, get_transform
from .image_folder import make_dataset
import torchvision.transforms as transforms
import tifffile
import torch
from .SliceBuilder import build_slices
import numpy as np
import math
from util.util import calculate_padding


class Patched25dDataset(BaseDataset2D):
    """
    Dataset for loading and patching images, in .tif format, in 2.5D for inference.

    Loads images from a specified path, applies patching in three dimensions,
    and is used during inference for the test_2_5d.py script. This class loads the dataset in full,
    which can incur high memory usage. For lazy loading, look at the 2.5D zarr dataset.
    """

    def __init__(self, opt):
        """
        Initialize the Patched25dDataset.

        Parameters
        ----------
        opt : Option class
            Stores all the experiment flags; needs to be a subclass of BaseOptions.
        """
        super().__init__(opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        self.transform = get_transform(opt)
        self.patch_size = np.asarray([opt.patch_size, opt.patch_size, opt.patch_size])
        self.init_padding = np.asarray([0, 0, 0])
        self.stride = self.patch_size

        if opt.netG.startswith("unet") and opt.stitch_mode == "tile-and-stitch":
            difference = 0
            for i in range(2, int(math.log(int(opt.netG[5:]), 2)) + 2):
                difference += 2**i
            stride = opt.patch_size - difference - 2
            self.stride = np.asarray([stride, stride, stride])
            self.init_padding = ((self.patch_size - self.stride) / 2).astype(int)
        elif opt.netG.startswith("resnet") or opt.netG.startswith("swinunetr"):
            opt.netG = opt.netG[:14]
            self.stride = self.patch_size - opt.patch_overlap
            self.init_padding = np.asarray([0, 0, 0])
        elif opt.stitch_mode.startswith("overlap-averaging"):
            self.stride = self.patch_size - opt.patch_overlap

    def __getitem__(self, index):
        """
        Return patches from a single image and its metadata information.

        Parameters
        ----------
        index : int
            Index for dataset access.

        Returns
        -------
        dict
            Dictionary containing:
            - 'xy', 'zy', 'zx': patches from three planes of the stack
            - 'A_paths': Path to the original image
            - 'A_full_size_raw': Dimensions of the raw dataset
            - 'A_full_size_pad': Dimensions of the padded image
            - 'patches_per_slice_xy', 'patches_per_slice_zy', 'patches_per_slice_zx': Number of patches per slice for each direction
        """
        A_path = self.A_paths[index]

        A_img_full = tifffile.imread(A_path)
        if len(A_img_full.shape) > 2:
            A_img_full = np.squeeze(A_img_full)

        A_img_size_raw = A_img_full.shape
        logging.info("Full Tif %s", A_img_full.shape)
        z1, y1, x1 = 1, 1, 1
        if self.opt.netG.startswith("unet"):
            z1, y1, x1 = calculate_padding(
                A_img_size_raw,
                init_padding=self.init_padding,
                input_patch_size=self.patch_size,
                stride=self.stride,
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
        logging.info("Padded numpy %s", A_img_full.shape)

        A_img_full = torch.from_numpy(A_img_full)
        logging.info("Padded torch %s", A_img_full.shape)

        A_img_size_pad = A_img_full.shape
        logging.info("Padded %s", A_img_size_pad)
        patches = []

        A_img_full_2 = torch.permute(A_img_full, (2, 0, 1))
        logging.info("zy shape: %s", A_img_full_2.shape)
        A_img_full_3 = torch.permute(A_img_full, (1, 0, 2))
        logging.info("zx shape: %s", A_img_full_3.shape)
        logging.info("Creating orthopatches xy")
        for i in range(0, A_img_size_pad[0]):  # for naive unet stitching
            A_img_slice = A_img_full[i]
            slices = build_slices(
                A_img_slice,
                [self.patch_size[1], self.patch_size[2]],
                [self.stride[1], self.stride[2]],
                use_shape_only=False,
            )
            num_patches_per_slice = len(slices)
            for slice_ in slices:
                A_img_patch = A_img_slice[slice_]
                A_img_patch = torch.unsqueeze(A_img_patch, 0)
                patches.append(A_img_patch)

        logging.info("Creating orthopatches zy")
        patches_2 = []
        for j in range(0, A_img_size_pad[2]):
            A_img_slice = A_img_full_2[j]
            slices = build_slices(
                A_img_slice,
                [self.patch_size[2], self.patch_size[2]],
                [self.stride[2], self.stride[2]],
                use_shape_only=False,
            )
            num_patches_per_slice_2 = len(slices)
            for slice_ in slices:
                A_img_patch = A_img_slice[slice_]
                A_img_patch = torch.unsqueeze(A_img_patch, 0)
                patches_2.append(A_img_patch)

        logging.info("Creating orthopatches zx")
        patches_3 = []
        for k in range(0, A_img_size_pad[1]):
            A_img_slice = A_img_full_3[k]
            slices = build_slices(
                A_img_slice,
                [self.patch_size[1], self.patch_size[1]],
                [self.stride[1], self.stride[1]],
                use_shape_only=False,
            )
            num_patches_per_slice_3 = len(slices)
            for slice_ in slices:
                A_img_patch = A_img_slice[slice_]
                A_img_patch = torch.unsqueeze(A_img_patch, 0)
                patches_3.append(A_img_patch)

        return {
            "xy": patches,
            "zy": patches_2,
            "zx": patches_3,
            "A_paths": A_path,
            "A_full_size_raw": A_img_size_raw,
            "A_full_size_pad": A_img_size_pad,
            "patches_per_slice_xy": num_patches_per_slice,
            "patches_per_slice_zy": num_patches_per_slice_2,
            "patches_per_slice_zx": num_patches_per_slice_3,
        }

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.A_paths)
