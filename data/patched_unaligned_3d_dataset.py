import os
import numpy as np
import logging
import random
import torch
import tifffile
import torchvision.transforms as transforms
from .base_dataset_3d import BaseDataset3D, get_transform
from .image_folder import make_dataset
from .SliceBuilder import build_slices_3d


class PatchedUnaligned3dDataset(BaseDataset3D):
    """
    Dataset for loading unaligned/unpaired 3D datasets for training.

    Requires two directories to host training images from domain A ('/path/to/data/trainA')
    and from domain B ('/path/to/data/trainB').
    Used during training of a 3D model.
    """

    def __init__(self, opt):
        """
        Initialize the PatchedUnaligned3dDataset.

        Parameters
        ----------
        opt : Option class
            Stores all the experiment flags; needs to be a subclass of BaseOptions.
        """
        super().__init__(opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.transform_A = get_transform(self.opt)
        self.transform_B = get_transform(self.opt)
        self.patch_size = [opt.patch_size, opt.patch_size, opt.patch_size]
        self.stride_A = [opt.stride_A, opt.stride_A, opt.stride_A]
        self.stride_B = [opt.stride_B, opt.stride_B, opt.stride_B]
        self.filter_A = 0.1
        self.filter_B = 0.1
        self.all_patches_A = self.build_patches(self.A_paths, self.stride_A, self.filter_A)
        self.all_patches_B = self.build_patches(self.B_paths, self.stride_B, self.filter_B)
        self.current_index = 0

    def build_patches(self, image_paths, stride, filter_):
        """
        Build patches from a list of image paths.

        Parameters
        ----------
        image_paths : list
            List of image file paths.
        stride : list or tuple
            Stride for patch extraction.
        filter_ : float
            Fraction of lowest-variance patches to filter out.

        Returns
        -------
        list
            Filtered list of patches.
        """
        all_patches = []
        stdevs = []
        transform = transforms.Compose([transforms.ToTensor()])
        for image_path in image_paths:
            img = tifffile.imread(image_path)
            img = transform(img)
            img = torch.permute(img, (1, 2, 0))
            logging.info("Building patches for %s", image_path)
            slices = build_slices_3d(img, self.patch_size, stride)
            for slice_ in slices:
                img_patch = img[slice_]
                img_patch = torch.unsqueeze(img_patch, 0)
                stdevs.append(torch.std(img_patch, dim=[1, 2, 3]).item())
                all_patches.append(img_patch)
        all_patches_sorted = [x for _, x in sorted(zip(stdevs, all_patches), key=lambda pair: pair[0])]
        first_index = int(filter_ * len(all_patches_sorted))
        all_patches_filtered = all_patches_sorted[first_index:]
        return all_patches_filtered

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
            - 'A': Patch from domain A
            - 'B': Patch from domain B
        """
        if len(self.all_patches_A) <= len(self.all_patches_B):
            A_patch = self.all_patches_A[index]
            B_patch = random.choice(self.all_patches_B)
        else:
            B_patch = self.all_patches_B[index]
            A_patch = random.choice(self.all_patches_A)
        A_patch = self.transform_A(A_patch)
        B_patch = self.transform_B(B_patch)
        return {"A": A_patch, "B": B_patch}

    def __len__(self):
        """
        Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take the minimum of the two.
        """
        return min(len(self.all_patches_A), len(self.all_patches_B))

    def normalize(self, input_, lower_percentile, upper_percentile):
        """
        Normalize input array to the given percentiles.

        Parameters
        ----------
        input_ : np.ndarray
            Input array to normalize.
        lower_percentile : float
            Lower percentile for normalization.
        upper_percentile : float
            Upper percentile for normalization.

        Returns
        -------
        np.ndarray
            Normalized input array.
        """
        u_p_input = np.percentile(input_, upper_percentile)
        l_p_input = np.percentile(input_, lower_percentile)
        normalized_input = (input_ - l_p_input) / (u_p_input - l_p_input)
        return normalized_input
