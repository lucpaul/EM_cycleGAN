import os

import numpy as np
import os
from .base_dataset import BaseDataset, get_transform
from .image_folder import make_dataset
from PIL import Image
from .SliceBuilder import build_slices, build_slices_fast
import torchvision.transforms as transforms
import tifffile
import random
import torch


class patchedunalignednewdataset(BaseDataset):
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

        self.max_samples = opt.max_dataset_size
        #self.A_size = len(self.A_paths)  # get the size of dataset A
        #self.B_size = len(self.B_paths)  # get the size of dataset B
        self.transform_A = get_transform(self.opt)#, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt)#, grayscale=(output_nc == 1))

        self.patch_size = [opt.patch_size, opt.patch_size] #[254, 254]
        self.stride_A = [opt.stride_A, opt.stride_A] #[222, 222]
        self.stride_B = [opt.stride_B, opt.stride_B] #[254, 254]

        self.patches_A = self.build_patches(self.A_paths, self.stride_A)
        self.patches_B = self.build_patches(self.B_paths, self.stride_B)

    def build_patches(self, image_paths, stride):
        all_patches = []
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        for image_path in image_paths:
            img = tifffile.imread(image_path)#, out='memmap')
            #print(img.min(), img.max())
            #img = self.normalize(img, 0.1, 99.8)
            #print(img.min(), img.max())
            img = transform(img)

            #img = self.normalize(img, 0.1, 99.8)

            img = torch.permute(img, (1, 2, 0))
            print("Building patches for", image_path)
            # print(img.shape, self.patch_size, stride)

            # This is the tried and tested version of the slicer
            #print(img.shape)
            if all([i % j == 0 for i, j in zip(torch.tensor((img.shape[0]*img.shape[1], img.shape[2])), self.patch_size)]):
                #print(torch.tensor((img.shape[0]*img.shape[1], img.shape[2])), self.patch_size)
                img_patches = build_slices_fast(img, self.patch_size, n_samples=self.max_samples)
                #img_patches = list(img_patches)
                all_patches += img_patches

                print('All image dimensions evenly divisible by patch size')
                #print(len(all_patches), all_patches[0:10])
            else:
                #print(img.shape)
                for z in range(0, img.shape[0]):
                    img_slice = img[z]
                    #print("shape: ", img.shape)
                    slices = build_slices(img_slice, self.patch_size, stride)

                    for slice in slices:
                        img_patch = img_slice[slice]
                        img_patch = torch.unsqueeze(img_patch, 0)
                        img_patch = self.transform_A(img_patch)
                        all_patches.append(img_patch)

                #print(len(all_patches), all_patches[0].shape)

        all_patches = torch.stack(all_patches)
        #all_patches = torch.concat(all_patches)
        #print(all_patches.shape)
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

        if len(self.patches_A) <= len(self.patches_B):
            A_patch = self.patches_A[index]
            B_patch = random.choice(self.patches_B)

        elif len(self.patches_B) < len(self.patches_A):
            B_patch = self.patches_B[index]
            A_patch = random.choice(self.patches_A)

        A_patch = self.transform_A(A_patch)
        B_patch = self.transform_B(B_patch)

        return {'A': A_patch, 'B': B_patch}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return min(len(self.patches_A), len(self.patches_B))


    def normalize(self, input, lower_percentile, upper_percentile):
        #print("input max:", input.max(), "input min: ", input.min())
        u_p_input = np.percentile(input, upper_percentile)
        l_p_input = np.percentile(input, lower_percentile)

        normalized_input = (input - l_p_input) / (u_p_input - l_p_input)

        return normalized_input