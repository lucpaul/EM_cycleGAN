###from here: https://github.com/jinh0park/pytorch-ssim-3D/tree/master

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    """
    Create a 1D Gaussian kernel.

    Args:
        window_size (int): Size of the window.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        torch.Tensor: 1D tensor with Gaussian weights.
    """
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """
    Create a 2D Gaussian window for SSIM computation.

    Args:
        window_size (int): Size of the window.
        channel (int): Number of channels.

    Returns:
        torch.autograd.Variable: 2D window expanded to the number of channels.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_window_3D(window_size, channel):
    """
    Create a 3D Gaussian window for SSIM computation.

    Args:
        window_size (int): Size of the window.
        channel (int): Number of channels.

    Returns:
        torch.autograd.Variable: 3D window expanded to the number of channels.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    Compute the Structural Similarity Index (SSIM) between two images (2D).

    Args:
        img1 (Tensor): First image.
        img2 (Tensor): Second image.
        window (Tensor): Gaussian window.
        window_size (int): Size of the window.
        channel (int): Number of channels.
        size_average (bool): Whether to average the SSIM map.

    Returns:
        Tensor: SSIM value or map.
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    """
    Compute the Structural Similarity Index (SSIM) between two images (3D).

    Args:
        img1 (Tensor): First image.
        img2 (Tensor): Second image.
        window (Tensor): Gaussian window.
        window_size (int): Size of the window.
        channel (int): Number of channels.
        size_average (bool): Whether to average the SSIM map.

    Returns:
        Tensor: SSIM value or map.
    """
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    """
    Module for computing 2D Structural Similarity Index (SSIM) between two images.
    """

    def __init__(self, window_size=11, size_average=True):
        """
        Initialize SSIM module.

        Args:
            window_size (int): Size of the Gaussian window.
            size_average (bool): Whether to average the SSIM map.
        """
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """
        Compute SSIM between two images.

        Args:
            img1 (Tensor): First image.
            img2 (Tensor): Second image.

        Returns:
            Tensor: SSIM value.
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM3D(torch.nn.Module):
    """
    Module for computing 3D Structural Similarity Index (SSIM) between two images.
    """

    def __init__(self, window_size=11, size_average=True):
        """
        Initialize SSIM3D module.

        Args:
            window_size (int): Size of the Gaussian window.
            size_average (bool): Whether to average the SSIM map.
        """
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        """
        Compute 3D SSIM between two images.

        Args:
            img1 (Tensor): First image.
            img2 (Tensor): Second image.

        Returns:
            Tensor: SSIM value.
        """
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    """
    Compute SSIM between two images (convenience function for 2D).

    Args:
        img1 (Tensor): First image.
        img2 (Tensor): Second image.
        window_size (int): Size of the Gaussian window.
        size_average (bool): Whether to average the SSIM map.

    Returns:
        Tensor: SSIM value.
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim3D(img1, img2, window_size=11, size_average=True):
    """
    Compute SSIM between two images (convenience function for 3D).

    Args:
        img1 (Tensor): First image.
        img2 (Tensor): Second image.
        window_size (int): Size of the Gaussian window.
        size_average (bool): Whether to average the SSIM map.

    Returns:
        Tensor: SSIM value.
    """
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)
