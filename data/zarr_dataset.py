
import os
from .base_dataset_3d import BaseDataset3D, get_transform#, normalize
from .image_folder import make_zarr_dataset
from .SliceBuilder import build_slices_3d
import torchvision.transforms as transforms
import tifffile
import random
import torch
import numpy as np
import zarr

class zarrdataset(BaseDataset3D):
    """
    This dataset class can load unaligned/unpaired datasets and is used during training of a 3d model.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset3D.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.A_paths = sorted(make_zarr_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_zarr_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.zarr_key = opt.zarr_key
        self.transform_A = get_transform(self.opt)
        self.transform_B = get_transform(self.opt)

        self.patch_size = [opt.patch_size, opt.patch_size, opt.patch_size]
        #self.stride_A = [opt.stride_A, opt.stride_A, opt.stride_A]
        #self.stride_B = [opt.stride_B, opt.stride_B, opt.stride_B]

        self.filter_A = 0.1
        self.filter_B = 0.1

        #self.all_patches_A = self.build_patches(self.A_paths, self.stride_A, self.filter_A)

        # Build patches for domain B during initialization
        #self.all_patches_B = self.build_patches(self.B_paths, self.stride_B, self.filter_B)

        # Initialize index to track progress during training
        self.current_index = 0

    def build_patches(self, image_paths, stride, filter):
        all_patches = []
        stdevs = []
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        for image_path in image_paths:
            print(image_path)
            img = np.array(zarr.open(image_path, "r", path="volumes/raw"))
            # TODO: take out normalization code, only for dev
            img = img.astype(np.float32)
            img = np.clip(img, 0, 255) / 255
            print(img.shape, img.dtype, np.min(img), np.max(img), np.mean(img))
            #img = tifffile.imread(image_path)
            img = transform(img)

            img = torch.permute(img, (1, 2, 0))
            print("Building patches for", image_path)

            # This is the tried and tested version of the slicer

            slices = build_slices_3d(img, self.patch_size, stride)

            for slice in slices:
                img_patch = img[slice]
                img_patch = torch.unsqueeze(img_patch, 0)
                stdevs.append(torch.std(img_patch, dim=[1, 2, 3]).item())
                #img_patch = self.transform_A(img_patch)
                all_patches.append(img_patch)

        #all_patches = torch.stack(all_patches)

        # Here I will test an option to filter out those patches that are mostly background.
        # For now, by choosing the 5% (?) of patches with the lowest standard deviation in pixel values,
        # which presumably contain the least insightful structures.

        #print("filtering out the worst patches")
        #stdevs = torch.squeeze(torch.std(all_patches, dim=[2, 3, 4]), dim=1)
        #index = torch.arange(stdevs.shape[0])
        #index[stdevs < torch.quantile(stdevs, filter, dim=None, keepdim=True, interpolation="nearest")] = -1

        #all_patches = torch.squeeze(all_patches, dim=0)[index != -1]

        all_patches_sorted = [x for _, x in sorted(zip(stdevs, all_patches), key=lambda pair:pair[0])]

        first_index = int(filter * len(all_patches_sorted))
        all_patches_filtered = all_patches_sorted[first_index:]
        return all_patches_filtered

    def __len__(self):
        return min(len(self.A_paths), len(self.B_paths))

    def load_patch(self, img_path):
        img = zarr.open(img_path, "r", path=self.zarr_key)
        # get random crop
        d, h, w = img.shape[-3:]
        new_d, new_h, new_w = self.patch_size
        d_start = np.random.randint(0, d - new_d + 1)
        h_start = np.random.randint(0, h - new_h + 1)
        w_start = np.random.randint(0, w - new_w + 1)

        img = np.array(img[
                       d_start:d_start + new_d,
                       h_start:h_start + new_h,
                       w_start:w_start + new_w])

        # TODO: take out normalization code, only for dev
        img = img.astype(np.float32)
        img = np.clip(img, 0, 255) / 255
        # img = np.expand_dims(img, axis=0)
        return img

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
        if len(self.A_paths) <= len(self.B_paths):
            A_path = self.A_paths[index]
            B_path = random.choice(self.B_paths)
        else:
            B_path = self.B_paths[index]
            A_path = random.choice(self.A_paths)

        # read in patches
        A_patch = self.load_patch(A_path)
        B_patch = self.load_patch(B_path)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        A_patch = torch.unsqueeze(transform(A_patch), 0)
        B_patch = torch.unsqueeze(transform(B_patch), 0)

        A_patch = self.transform_A(A_patch)
        B_patch = self.transform_B(B_patch)


        return {'A': A_patch, 'B': B_patch}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return min(len(self.A_paths), len(self.B_paths))


    def normalize(self, input, lower_percentile, upper_percentile):
        u_p_input = np.percentile(input, upper_percentile)
        l_p_input = np.percentile(input, lower_percentile)

        # Normalize between -1 and 1
        # normalized_input = (2 * (input - l_p_input) / (u_p_input - l_p_input)) - 1

        # Normalize between 0 and 1
        normalized_input = (input - l_p_input) / (u_p_input - l_p_input)
        return normalized_input

