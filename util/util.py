"""This module contains simple helper functions."""

import os
import numpy as np
import torch


def calculate_padding(volume_shape, init_padding, input_patch_size, stride, dim=None):
    """
    Calculate the required padding for patch-based inference to ensure full coverage of the input volume.

    If 'dim' is specified, only the last 'dim' dimensions are used (e.g., for 2D, dim=2 skips channel/z).
    Otherwise, all dimensions are used (for 2.5D/3D).

    Args:
        volume_shape (tuple or list): Shape of the input volume.
        init_padding (int or array-like): Initial padding applied to the volume.
        input_patch_size (int or array-like): Size of the patch to extract.
        stride (int or array-like): Stride between patches.
        dim (int, optional): Number of spatial dimensions to consider (e.g., 2 for 2D, 3 for 3D). If None, use all dimensions.

    Returns:
        np.ndarray: The required padding for each dimension as an integer array.
    """

    if dim is not None:
        shape = np.array(volume_shape[-dim:])
    else:
        shape = np.array(volume_shape)

    number_of_patches = np.ma.ceil(((shape + init_padding - input_patch_size) / stride) + 1)
    volume_new = (np.asarray(number_of_patches) * stride) + input_patch_size
    new_padding = volume_new - shape - init_padding
    new_padding = np.where(
        new_padding > ((input_patch_size - stride) / 2),
        new_padding,
        new_padding + (input_patch_size - stride),
    )
    return new_padding.astype(int)


def tensor2im(input_image):
    """ "Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy


def adjust_patch_size(opt):
    """
    Adjust the patch size in the options object to ensure compatibility with the chosen network architecture.

    For UNet architectures, ensures the patch size is divisible by the depth factor (number of downsampling steps),
    accounting for valid convolutions and optional patch halo. For other backbones, ensures divisibility by the depth factor.
    Modifies opt.patch_size in place if necessary.

    Args:
        opt: An options object with attributes patch_size, netG, stitch_mode, phase, and optionally patch_halo.
    """
    # The depth factor's default value should change depening on the network architecture used for generating images.
    depth_factor = 5

    old_patch_size = opt.patch_size
    if opt.netG.startswith("unet"):
        depth_factor = int(opt.netG[5:])
        if opt.stitch_mode == "tile-and-stitch" or opt.phase == "train":
            patch_size = opt.patch_size
            if (patch_size + 2) % depth_factor == 0:
                pass
            else:
                # In the valid unet, the patch sizes that can be evenly downsampled in the layers (i.e. without residual) are
                # limited to values which are divisible by 32 (2**5 for 5 downsampling steps), after adding the pixels lost in the valid conv layer, i.e.:
                # 158 (instead of 160), 190 (instead of 192), 222 (instead of 224), etc. Below, the nearest available patch size
                # selected to patch the image accordingly. (Choosing a smaller value than the given patch size, should ensure
                # that the patches are not bigger than any dimensions of the whole input image)
                new_patch_sizes = opt.patch_size - torch.arange(1, depth_factor)
                new_patch_size = int(new_patch_sizes[(new_patch_sizes + 2) % depth_factor == 0])
                opt.patch_size = new_patch_size
                print(
                    f"The provided patch size {old_patch_size} is not compatible with the chosen unet backbone with valid convolutions. Patch size was changed to {new_patch_size}"
                )

        elif opt.stitch_mode == "overlap-averaging":
            patch_size = opt.patch_size
            if (patch_size + opt.patch_halo * 2) % depth_factor == 0:
                print(
                    "Patch size is compatible with the unet backbone.",
                    patch_size + opt.patch_halo * 2,
                )
                pass
            else:
                new_patch_sizes = opt.patch_size + opt.patch_halo * 2 - torch.arange(1, depth_factor)
                new_patch_size = int(new_patch_sizes[new_patch_sizes % depth_factor == 0]) - opt.patch_halo * 2
                opt.patch_size = new_patch_size
                print(
                    f"The provided patch size {old_patch_size} is not compatible with the unet backbone. Patch size was changed to {new_patch_size}"
                )

    else:
        # This section has not been fully tested, but will be of interest if other backbones are used.
        patch_size = opt.patch_size
        if patch_size % depth_factor == 0:
            pass
        else:
            new_patch_sizes = opt.patch_size - torch.arange(1, depth_factor)
            new_patch_size = int(new_patch_sizes[(new_patch_sizes % depth_factor) == 0])
            opt.patch_size = new_patch_size
            print(f"The provided patch size {old_patch_size} is not compatible with the resnet backbone. Patch size was changed to {new_patch_size}")


def diagnose_network(net, name="network"):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
