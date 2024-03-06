from .base_dataset_2d import BaseDataset2D, get_transform
from .image_folder import make_dataset
import tifffile
import math
from .SliceBuilder import build_slices
import numpy as np

def _calc_padding(volume_shape, init_padding, input_patch_size, stride):
    number_of_patches = np.ma.ceil(((volume_shape[1:] + init_padding - input_patch_size) / stride) + 1)
    volume_new = ((np.asarray(number_of_patches))*stride) + input_patch_size
    new_padding = volume_new - volume_shape[1:] - init_padding
    return new_padding.astype(int)

class patched2ddataset(BaseDataset2D):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.
    When the test_script for the unet model is run, it patches the dataset for each z-slice of a stack with a hard-coded
    stride that is exactly equal to the output shape of the unet which allows the results to be tiled-and-stitched without artefacts.

    If run with a different backbone, the stride is hardcoded to be smaller than or equal to the patch size for each z-slice
    and results are stitched in a standard approach by averaging overlapping patches.

    This dataset is used during inference for the test_2D.py and test_2D_resnet.py scripts.

    It can be called during inference using the flag --test_mode 2d
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset2D.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        self.transform = get_transform(opt)
        self.patch_size = np.asarray([opt.patch_size, opt.patch_size])

        if opt.netG.startswith('unet'):
            difference = 0
            for i in range(2, int(math.log(int(opt.netG[5:]), 2)) + 2):
                difference += 2 ** i
            stride = opt.patch_size - difference - 2
            self.stride = np.asarray([stride, stride])
        else:
            self.stride = self.patch_size - 16

        assert self.patch_size.all() >= self.stride.all(), f"Images can only be stitched if patch size is at least equal to stride, but not smaller. " \
                                                        f"Given patch size is {self.patch_size} and stride {self.stride}. That won't work."
        self.init_padding = ((self.patch_size - self.stride) / 2).astype(int)


    def build_patches(self, image_path, patch_size, stride):
        """We create a function which converts a volume into blocks using """

        A_img_full = tifffile.imread(image_path) # Read image
        #A_img_full = normalize(A_img_full, 0.1, 99.8) #Not tested the results for this yet

        A_img_size_raw = A_img_full.shape # Get the raw image size

        if self.opt.netG.startswith('unet'):
            y1, x1 = _calc_padding(A_img_size_raw, init_padding=self.init_padding, input_patch_size=patch_size, stride=stride)
            init_padding_param = int(self.init_padding[0])
            A_img_full = np.pad(A_img_full, pad_width=((0, 0), (init_padding_param, y1), (init_padding_param, x1)), mode="reflect")

        A_img_size_pad = A_img_full.shape
        img_sizes = (A_img_size_raw, A_img_size_pad)

        patches = []
        for z in range(0, A_img_size_raw[0]):
            img_slice = A_img_full[z]
            slices = build_slices(img_slice, patch_size, stride)
            for slice in slices:
                A_img_patch = img_slice[slice]
                A_img_patch = np.expand_dims(A_img_patch, 0)
                patches.append(A_img_patch)

        #Converting to np.ndarray is a bit mysterious in terms of RAM use. Sometimes useful, sometimes catastrophic.
        #Leaving it here in case it's needed again.

        #patches = np.array(patches)

        patches_per_slice = len(slices)

        return patches, img_sizes, patches_per_slice

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        #print("getItem: ", self.patch_size, self.stride)
        patches, img_sizes, patches_per_slice = self.build_patches(self.A_paths[index], self.patch_size, self.stride)

        A_path = self.A_paths[index]

        A_size_raw = img_sizes[0]

        A_size_pad = img_sizes[1]

        return {'A': patches, 'A_paths': A_path, 'A_full_size_raw': A_size_raw, 'A_full_size_pad': A_size_pad, 'patches_per_slice': patches_per_slice}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)