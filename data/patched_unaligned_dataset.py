import os

import numpy as np
import os
from .base_dataset import BaseDataset, get_transform
from .image_folder import make_dataset
from PIL import Image
from .SliceBuilder import build_slices
import torchvision.transforms as transforms
import tifffile
import random
import torch


class patchedunaligneddataset(BaseDataset):
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
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.transform_A = get_transform(self.opt)#, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt)#, grayscale=(output_nc == 1))

        self.patch_size = [opt.patch_size, opt.patch_size] #[254, 254]
        self.stride_A = [opt.stride_A, opt.stride_A] #[222, 222]
        self.stride_B = [opt.stride_B, opt.stride_B] #[254, 254]

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
        A_path = self.A_paths[index % self.A_size]  # make sure index is within the  range
        if self.opt.serial_batches:   # make sure index is within the range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        if A_path.endswith(".tif"):
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            A_img = tifffile.imread(A_path)

            B_img = tifffile.imread(B_path)

        A_img = transform(A_img)
        A_img = self.normalize(A_img, 0.1, 99.8)
        patches_A = []
        for i in list(range(0, A_img.shape[0]))[0::10]:
            A_img_2D_slice = A_img[i, :, :]
            slices_A = build_slices(A_img_2D_slice, self.patch_size, self.stride_A)

            for sliceA in slices_A:
                A_img_patch = A_img_2D_slice[sliceA]
                A_img_patch = torch.unsqueeze(A_img_patch, 0)
                patches_A.append(A_img_patch)

        B_img = transform(B_img)
        B_img = self.normalize(B_img, 0.1, 99.8)
        patches_B = []

        for i in list(range(0, B_img.shape[0]))[0::10]:
            B_img_2D_slice = B_img[i, :, :]
            slices_B = build_slices(B_img_2D_slice, self.patch_size, self.stride_B)

            for sliceB in slices_B:
                B_img_patch = B_img_2D_slice[sliceB]
                B_img_patch = torch.unsqueeze(B_img_patch, 0)
                patches_B.append(B_img_patch)

        patches_A = random.sample(patches_A, k=len(patches_A))

        return {'A': patches_A, 'B': patches_B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


    def normalize(self, input, lower_percentile, upper_percentile):
        #print("input max:", input.max(), "input min: ", input.min())
        u_p_input = np.percentile(input, upper_percentile)
        l_p_input = np.percentile(input, lower_percentile)

        normalized_input = (input - l_p_input) / (u_p_input - l_p_input)

        return normalized_input