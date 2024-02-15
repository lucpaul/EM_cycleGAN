import os
from .base_dataset_3d import BaseDataset3D, get_transform#, normalize
from .image_folder_3d import make_dataset
from .SliceBuilder import build_slices_3d
import torchvision.transforms as transforms
import tifffile
import random
import torch
import numpy as np
import time


class patchedunaligned3ddataset(BaseDataset3D):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset3D.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        self.transform_A = get_transform(self.opt)
        self.transform_B = get_transform(self.opt)

        self.patch_size = [opt.patch_size, opt.patch_size, opt.patch_size]  # [190, 190, 190]
        self.stride_A = [opt.stride_A, opt.stride_A, opt.stride_A]  # [180, 180, 180]
        self.stride_B = [opt.stride_B, opt.stride_B, opt.stride_B]  # [200, 200, 200]

        self.all_patches_A = self.build_patches(self.A_paths, self.stride_A)

        # Build patches for domain B during initialization
        self.all_patches_B = self.build_patches(self.B_paths, self.stride_B)

        # Initialize index to track progress during training
        self.current_index = 0

    def build_patches(self, image_paths, stride):
        all_patches = []
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        start = time.time()
        for image_path in image_paths:
            img = tifffile.imread(image_path)#, out='memmap')

            img = transform(img)

            img = torch.permute(img, (1, 2, 0))
            print("Building patches for", image_path)

            # This is the tried and tested version of the slicer

            slices = build_slices_3d(img, self.patch_size, stride)

            for slice in slices:
                img_patch = img[slice]
                img_patch = torch.unsqueeze(img_patch, 0)
                img_patch = self.transform_A(img_patch)
                all_patches.append(img_patch)

        all_patches = torch.stack(all_patches)

        stop = time.time()
        print(stop-start)
        return all_patches

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        if len(self.all_patches_A) <= len(self.all_patches_B):
            A_patch = self.all_patches_A[index]
            B_patch = random.choice(self.all_patches_B)

        elif len(self.all_patches_B) < len(self.all_patches_A):
            B_patch = self.all_patches_B[index]
            A_patch = random.choice(self.all_patches_A)

        A_patch = self.transform_A(A_patch)
        B_patch = self.transform_B(B_patch)
        return {'A': A_patch, 'B': B_patch}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        len_A = 0
        len_B = 0

        for image_A_path in self.A_paths:
            A_img = tifffile.TiffFile(image_A_path)
            A_img_shape = (len(A_img.pages), A_img.pages[0].shape[0], A_img.pages[0].shape[1])
            z0, y0, x0 = np.ma.ceil(((np.asarray(A_img_shape) - np.asarray(self.patch_size)) / np.asarray(self.stride_A)) + 1)
            number_of_patches_A = z0*y0*x0
            len_A += number_of_patches_A

        for image_B_path in self.B_paths:
            B_img = tifffile.TiffFile(image_B_path)
            B_img_shape = (len(B_img.pages), B_img.pages[0].shape[0], B_img.pages[0].shape[1])
            z1, y1, x1 = np.ma.ceil(((np.asarray(B_img_shape) - np.asarray(self.patch_size)) / np.asarray(self.stride_B)) + 1)
            number_of_patches_B = z1*y1*x1
            len_B += number_of_patches_B

        return min(int(len_A), int(len_B))


    def normalize(self, input, lower_percentile, upper_percentile):
        #print("input max:", input.max(), "input min: ", input.min())
        u_p_input = np.percentile(input, upper_percentile)
        l_p_input = np.percentile(input, lower_percentile)

        # Normalize between -1 and 1
        # normalized_input = (2 * (input - l_p_input) / (u_p_input - l_p_input)) - 1

        # Normalize between 0 and 1
        normalized_input = (input - l_p_input) / (u_p_input - l_p_input)
        return normalized_input

