import os

import numpy as np
import os
from .base_dataset_2d import BaseDataset2D, get_transform
from .image_folder import make_dataset
from .SliceBuilder import build_slices, build_slices_fast
import torchvision.transforms as transforms
import tifffile
import random
import torch


class patchedunaligned2ddataset(BaseDataset2D):
    """
    This dataset class can load unaligned/unpaired datasets and is used during training of a 2d model.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset2D.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.max_samples = opt.max_dataset_size
        self.transform_A = get_transform(self.opt)#, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt)#, grayscale=(output_nc == 1))

        self.patch_size = [opt.patch_size, opt.patch_size]
        self.stride_A = [opt.stride_A, opt.stride_A]
        self.stride_B = [opt.stride_B, opt.stride_B]

        self.filter_A = 0.1
        self.filter_B = 0.1

        self.patches_A = self.build_patches(self.A_paths, self.stride_A, self.filter_A)
        self.patches_B = self.build_patches(self.B_paths, self.stride_B, self.filter_B)

    def build_patches(self, image_paths, stride, filter):
        all_patches = []
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        for image_path in image_paths:
            img = tifffile.imread(image_path)
            img = transform(img)
            img = torch.permute(img, (1, 2, 0))
            print("Building patches for", image_path)
            # This option is quite a bit faster, but currently should only be used if the image is evenly divisible by the patch size
            # This one also currently will not augment the extracted patches.
            if stride == self.patch_size and all([i % j == 0 for i, j in zip(torch.tensor((img.shape[0]*img.shape[1], img.shape[2])), self.patch_size)]):
                img_patches = build_slices_fast(img, self.patch_size, n_samples=self.max_samples)
                all_patches += img_patches

                print('All image dimensions evenly divisible by patch size')

            # This is the tried and trusted version of the slicer
            else:
                for z in range(0, img.shape[0]):
                    img_slice = img[z]
                    slices = build_slices(img_slice, self.patch_size, stride)

                    for slice in slices:
                        img_patch = img_slice[slice]
                        img_patch = torch.unsqueeze(img_patch, 0)
                        img_patch = self.transform_A(img_patch)
                        all_patches.append(img_patch)

        all_patches = torch.stack(all_patches)

        # Here I will test an option to filter out those patches that are mostly background.
        # For now, by choosing the 5% (?) of patches with the lowest standard deviation in pixel values,
        # which presumably contain the least insightful structures.

        print("filtering out the worst patches")

        stdevs = torch.squeeze(torch.std(all_patches, dim=[2, 3]), dim=1)
        index = torch.arange(stdevs.shape[0])
        index[stdevs < torch.quantile(stdevs, filter, dim=None, keepdim=True, interpolation="nearest")] = -1

        all_patches = torch.squeeze(all_patches, dim=0)[index != -1]
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
        u_p_input = np.percentile(input, upper_percentile)
        l_p_input = np.percentile(input, lower_percentile)

        normalized_input = (input - l_p_input) / (u_p_input - l_p_input)

        return normalized_input