import numpy as np
import tifffile

from .SliceBuilder import build_slices_3d
from .base_dataset_3d import BaseDataset3D, get_transform
from .image_folder import make_dataset


def _calc_padding(volume_shape, init_padding, input_patch_size, stride):
    number_of_patches = (np.ma.ceil(((volume_shape + init_padding - input_patch_size) / stride) + 1)).astype(int)
    volume_new = ((np.asarray(number_of_patches))*stride) + input_patch_size
    new_padding = volume_new - volume_shape - init_padding

    # The below statement ensures that the padded volume will remain bigger than the input volume, even when
    # the stride, and thus the cropped patch size after inference, is very small.
    # If the calculated padding is sufficient, then nothing changes:

    new_padding = np.where(new_padding > ((input_patch_size - stride) / 2), new_padding, new_padding + (input_patch_size - stride))

    return new_padding.astype(int)

class patcheddataset_3d(BaseDataset3D):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset3D.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))

        self.transform = get_transform(opt)#, grayscale=(input_nc == 1))

        self.stride = np.asarray([opt.stride_A, opt.stride_A, opt.stride_A])


        self.patch_size = np.asarray([opt.patch_size, opt.patch_size, opt.patch_size])

        self.init_padding = ((self.patch_size - self.stride) / 2).astype(int)

    def build_patches(self, image_path, patch_size, stride):
        """We create a function which converts a volume into blocks using """

        A_img_full = tifffile.imread(image_path) # Read image
        #A_img_full = normalize(A_img_full, 0.1, 99.8) #Not tested the results for this yet

        A_img_size_raw = A_img_full.shape # Get the raw image size

        z1, y1, x1 = _calc_padding(A_img_size_raw, init_padding=self.init_padding, input_patch_size=patch_size, stride=stride)

        init_padding_param = int(self.init_padding[0])

        #A_img_full = np.pad(A_img_full, pad_width=((init_padding_param, z1), (init_padding_param, y1), (init_padding_param, x1)), mode="reflect")

        slices = build_slices_3d(A_img_full, patch_size, stride)

        A_img_size_pad = A_img_full.shape
        img_sizes = (A_img_size_raw, A_img_size_pad)

        patches = []
        for slice in slices:
            A_img_patch = A_img_full[slice]
            A_img_patch = np.expand_dims(A_img_patch, 0)
            patches.append(A_img_patch)

        #Converting to np.ndarray is a bit mysterious in terms of RAM use. Sometimes useful, sometimes catastrophic.
        #Leaving it here in case it's needed again.

        #patches = np.array(patches)

        return patches, img_sizes

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        patches, img_sizes = self.build_patches(self.A_paths[index], self.patch_size, self.stride)

        A_path = self.A_paths[index]

        A_size_raw = img_sizes[0]

        A_size_pad = img_sizes[1]

        return {'A':patches, 'A_paths':A_path, 'A_full_size_raw':A_size_raw, 'A_full_size_pad':A_size_pad}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
