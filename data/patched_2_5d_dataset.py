from .base_dataset_2d import BaseDataset2D, get_transform
from .image_folder import make_dataset
import torchvision.transforms as transforms
import tifffile
import torch
from .SliceBuilder import build_slices
import numpy as np
import math


def _calc_padding(volume_shape, init_padding, input_patch_size, stride):
    number_of_patches = np.ma.ceil(((volume_shape + init_padding - input_patch_size) / stride) + 1)
    volume_new = ((np.asarray(number_of_patches))*stride) + input_patch_size
    new_padding = volume_new - volume_shape - init_padding
    return new_padding.astype(int)


class patched25ddataset(BaseDataset2D):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.
    It patches the dataset sequentially in 3 dimensions with a hard-coded stride that is equal to or smaller than the patch size.

    This dataset is used during inference for the test_2_5D.py and test_2_5D_resnet.py scripts.

    It can be called during inference using the flag --test_mode 2.5d
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset2D.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        self.transform = get_transform(opt)
        self.patch_size = np.asarray([opt.patch_size, opt.patch_size, opt.patch_size])

        if opt.netG.startswith('unet'):
            difference = 0
            for i in range(2, int(math.log(int(opt.netG[5:]), 2)) + 2):
                difference += 2 ** i
            stride = opt.patch_size - difference - 2
            self.stride = np.asarray([stride, stride, stride])
            self.init_padding = ((self.patch_size - self.stride) / 2).astype(int)
        elif opt.netG.startswith('resnet') or opt.netG.startswith('swinunetr'):
            opt.netG = opt.netG[:14]
            self.stride = self.patch_size - 8
            self.init_padding = np.asarray([0, 0, 0])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        A_img_full = tifffile.imread(A_path)
        A_img_size_raw = A_img_full.shape

        z1, y1, x1 = 1, 1, 1
        if self.opt.netG.startswith('unet'):
            z1, y1, x1 = _calc_padding(A_img_size_raw, init_padding=self.init_padding, input_patch_size=self.patch_size, stride=self.stride)
            init_padding_param = int(self.init_padding[0])
            A_img_full = np.pad(A_img_full, pad_width=((init_padding_param, z1), (init_padding_param, y1), (init_padding_param, x1)), mode="reflect")

        A_img_full = transform(A_img_full)
        A_img_full = torch.permute(A_img_full, (1, 2, 0))

        A_img_size_pad = A_img_full.shape
        patches = []

        A_img_full_2 = torch.permute(A_img_full, (2, 0, 1))
        A_img_full_3 = torch.permute(A_img_full, (1, 2, 0))
        print("Creating orthopatches xy")
        for i in range(self.init_padding[0], A_img_size_pad[0]-z1+1):#range(0, A_img_size_pad[0]): # for naive unet stitching
            A_img_slice = A_img_full[i]
            slices = build_slices(A_img_slice, [self.patch_size[1], self.patch_size[2]], [self.stride[1], self.stride[2]])
            num_patches_per_slice = len(slices)
            for slice in slices:
                A_img_patch = A_img_slice[slice]
                A_img_patch = torch.unsqueeze(A_img_patch, 0)
                patches.append(A_img_patch)

        print("Creating orthopatches zy")
        patches_2 = []
        for j in range(self.init_padding[2], A_img_size_pad[2]-x1+1):#range(0, A_img_size_pad[2])
            A_img_slice = A_img_full_2[j]
            slices = build_slices(A_img_slice, [self.patch_size[2], self.patch_size[2]], [self.stride[2], self.stride[2]])
            num_patches_per_slice_2 = len(slices)
            for slice in slices:
                A_img_patch = A_img_slice[slice]
                A_img_patch = torch.unsqueeze(A_img_patch, 0)
                patches_2.append(A_img_patch)

        print("Creating orthopatches zx")
        patches_3 = []
        for k in range(self.init_padding[1], A_img_size_pad[1]-y1+1):#range(0, A_img_size_pad[1]):
            A_img_slice = A_img_full_3[k]
            slices = build_slices(A_img_slice, [self.patch_size[1], self.patch_size[1]], [self.stride[1], self.stride[1]])
            num_patches_per_slice_3 = len(slices)
            for slice in slices:
                A_img_patch = A_img_slice[slice]
                A_img_patch = torch.unsqueeze(A_img_patch, 0)
                patches_3.append(A_img_patch)

        return {'xy': patches, 'zy': patches_2, 'zx': patches_3,
                'A_paths': A_path, 'A_full_size_raw': A_img_size_raw,
                'A_full_size_pad': A_img_size_pad,
                'patches_per_slice_xy': num_patches_per_slice,
                'patches_per_slice_zy': num_patches_per_slice_2,
                'patches_per_slice_zx': num_patches_per_slice_3}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)