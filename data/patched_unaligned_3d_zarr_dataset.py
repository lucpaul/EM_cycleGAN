import os
import numpy as np
import logging
from .base_dataset_3d import BaseDataset3D, get_transform
from .image_folder import make_dataset, make_zarr_dataset
from .SliceBuilder import build_slices_3d
import torch
import tifffile
import zarr
import torchvision.transforms as transforms
from numcodecs import Blosc
from functools import lru_cache


class PatchedUnaligned3dZarrDataset(BaseDataset3D):
    """
    Dataset for loading unaligned/unpaired 3D datasets from Zarr or TIFF for training.

    Requires two directories to host training images from domain A ('/path/to/data/trainA')
    and from domain B ('/path/to/data/trainB').
    Used during training of a 3D model.
    """

    def __init__(self, opt):
        """
        Initialize the PatchedUnaligned3dZarrDataset.

        Parameters
        ----------
        opt : Option class
            Stores all the experiment flags; needs to be a subclass of BaseOptions.
        """
        super().__init__(opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")
        if os.listdir(self.dir_A)[0].endswith(".tif"):
            assert os.listdir(self.dir_B)[0].endswith("tif"), "Files in trainA and trainB should both be in .tif format"
            self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        else:
            self.A_paths = sorted(make_zarr_dataset(self.dir_A, opt.max_dataset_size))
            self.B_paths = sorted(make_zarr_dataset(self.dir_B, opt.max_dataset_size))
        self.max_samples = opt.max_dataset_size
        self.transform_A = get_transform(self.opt)
        self.transform_B = get_transform(self.opt)
        self.patch_size = (opt.patch_size, opt.patch_size, opt.patch_size)
        self.stride_A = (opt.stride_A, opt.stride_A, opt.stride_A)
        self.stride_B = (opt.stride_B, opt.stride_B, opt.stride_B)
        self.filter_A = 0.1
        self.filter_B = 0.1
        self._cache_size = opt.cache_size
        self._get_patch_cached = lru_cache(maxsize=self._cache_size)(self._get_patch)
        self.use_caching = False
        if opt.use_zarr:
            if self.A_paths[0].endswith("tif"):
                logging.info("Converting tif files to zarr format.")
                self.data_is_zarr = False
                self.zarr_paths_A = save_as_zarr(self.A_paths, self.patch_size, self.stride_A, self.filter_A)
                self.zarr_paths_B = save_as_zarr(self.B_paths, self.patch_size, self.stride_B, self.filter_B)
                self.datasets_A = [zarr.open(zarr_file_A, mode="r") for zarr_file_A in self.zarr_paths_A]
                self.datasets_B = [zarr.open(zarr_file_B, mode="r") for zarr_file_B in self.zarr_paths_B]
                self.cumulative_sizes_A = compute_cumulative_sizes(self.datasets_A)
                self.cumulative_sizes_B = compute_cumulative_sizes(self.datasets_B)
            else:
                self.data_is_zarr = True
                self.datasets_A = [zarr.open(zarr_file_A, mode="r") for zarr_file_A in self.A_paths]
                self.datasets_B = [zarr.open(zarr_file_B, mode="r") for zarr_file_B in self.B_paths]
                self.is_pre_extracted_A = []
                for dataset_A in self.datasets_A:
                    if len(dataset_A.shape) == 4:
                        self.is_pre_extracted_A.append(True)
                    elif len(dataset_A.shape) == 3:
                        self.is_pre_extracted_A.append(False)
                    else:
                        raise ValueError(f"Unsupported dataset shape {dataset_A.shape}")
                self.is_pre_extracted_B = []
                for dataset_B in self.datasets_B:
                    if len(dataset_B.shape) == 4:
                        self.is_pre_extracted_B.append(True)
                    elif len(dataset_B.shape) == 3:
                        self.is_pre_extracted_B.append(False)
                    else:
                        raise ValueError(f"Unsupported dataset shape {dataset_B.shape}")
                self.patch_indices_A = extract_zarr_patches(
                    self.datasets_A,
                    self.patch_size,
                    self.stride_A,
                    self.is_pre_extracted_A,
                )
                self.patch_indices_B = extract_zarr_patches(
                    self.datasets_B,
                    self.patch_size,
                    self.stride_B,
                    self.is_pre_extracted_B,
                )

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Parameters
        ----------
        index : int
            Index for data access.

        Returns
        -------
        dict
            Dictionary containing:
            - 'A': Patch from domain A
            - 'B': Patch from domain B
        """
        if self.data_is_zarr:
            patch_info_A = self.patch_indices_A[index]
            dataset_idx_A = patch_info_A[0]
            dataset_A = self.datasets_A[dataset_idx_A]
            patch_info_B = self.patch_indices_B[index]
            dataset_idx_B = patch_info_B[0]
            dataset_B = self.datasets_B[dataset_idx_B]
            if self.is_pre_extracted_A[dataset_idx_A]:
                patch_idx_A = patch_info_A[1]
                patch_A = dataset_A[patch_idx_A]
            else:
                d, h, w = patch_info_A[1:]
                pd, ph, pw = self.patch_size
                if self.use_caching:
                    patch_A = self._get_patch_cached(self.datasets_A, dataset_idx_A, d, h, w)
                else:
                    patch_A = dataset_A[d : d + pd, h : h + ph, w : w + pw]
            if self.is_pre_extracted_B[dataset_idx_B]:
                patch_idx_B = patch_info_B[1]
                patch_B = dataset_B[patch_idx_B]
            else:
                d, h, w = patch_info_B[1:]
                pd, ph, pw = self.patch_size
                if self.use_caching:
                    patch_B = self._get_patch_cached(self.datasets_B, dataset_idx_B, d, h, w)
                else:
                    patch_B = dataset_B[d : d + pd, h : h + ph, w : w + pw]
        else:
            dataset_idx_A = np.searchsorted(self.cumulative_sizes_A, index, side="right") - 1
            local_index_A = index - self.cumulative_sizes_A[dataset_idx_A]
            dataset_idx_B = np.searchsorted(self.cumulative_sizes_B, index, side="right") - 1
            local_index_B = index - self.cumulative_sizes_B[dataset_idx_B]
            patch_A = self.datasets_A[dataset_idx_A][local_index_A]
            patch_B = self.datasets_B[dataset_idx_B][local_index_B]
        transform = transforms.Compose([transforms.ToTensor()])
        patch_A = transform(patch_A)
        patch_B = transform(patch_B)
        patch_A = torch.unsqueeze(self.transform_A(patch_A), dim=0)
        patch_B = torch.unsqueeze(self.transform_B(patch_B), dim=0)
        return {"A": patch_A, "B": patch_B}

    def __len__(self):
        """
        Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take the minimum of the two.
        """
        if self.data_is_zarr:
            return min(len(self.patch_indices_A), len(self.patch_indices_B))
        return min(self.cumulative_sizes_A[-1], self.cumulative_sizes_B[-1])

    def _get_patch(self, datasets, dataset_idx, d, h, w):
        """
        Extract a patch from a full 3D volume dataset. This function is cached to improve performance.

        Parameters
        ----------
        datasets : list
            List of zarr datasets.
        dataset_idx : int
            Index of the dataset.
        d, h, w : int
            Depth, height, and width indices.

        Returns
        -------
        np.ndarray
            Extracted patch.
        """
        dataset = datasets[dataset_idx]
        pd, ph, pw = self.patch_size
        patch = dataset[d : d + pd, h : h + ph, w : w + pw]
        return patch

    def clear_cache(self):
        """
        Clear the LRU cache to free memory.
        """
        self._get_patch_cached.cache_clear()


def compute_cumulative_sizes(datasets):
    """
    Compute the cumulative sizes of all datasets for global indexing.

    Parameters
    ----------
    datasets : list
        List of zarr datasets.

    Returns
    -------
    list
        Cumulative sizes of patches across datasets.
    """
    sizes = [dataset.shape[0] for dataset in datasets]
    return [0] + list(np.cumsum(sizes))


def build_patches(image_path, patch_size, stride, filter_):
    """
    Build patches from a single image path.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    patch_size : tuple
        Size of the patch.
    stride : tuple
        Stride for patch extraction.
    filter_ : float
        Fraction of lowest-variance patches to filter out.

    Returns
    -------
    list
        Filtered list of patches.
    """
    all_patches = []
    stdevs = []
    logging.info("Building patches for %s", image_path)
    img = tifffile.imread(image_path)
    if len(img.shape) > 3:
        img = np.squeeze(img)
    slices = build_slices_3d(img, patch_size, stride)
    for slice_ in slices:
        img_patch = img[slice_]
        stdevs.append(np.std(img_patch))
        all_patches.append(img_patch)
    all_patches_sorted = [x for _, x in sorted(zip(stdevs, all_patches), key=lambda pair: pair[0])]
    first_index = int(filter_ * len(all_patches_sorted))
    all_patches_filtered = all_patches_sorted[first_index:]
    return all_patches_filtered


def extract_zarr_patches(datasets, patch_size, stride, is_pre_extracted):
    """
    Extract patch indices for zarr datasets, handling both pre-extracted and full-volume cases.

    Parameters
    ----------
    datasets : list
        List of zarr datasets.
    patch_size : tuple
        Size of the patch.
    stride : tuple
        Stride for patch extraction.
    is_pre_extracted : list
        List of booleans indicating if each dataset is pre-extracted.

    Returns
    -------
    list
        List of patch indices.
    """
    patch_indices = []
    for dataset_idx, dataset in enumerate(datasets):
        if is_pre_extracted[dataset_idx]:
            num_patches = dataset.shape[0]
            for patch_idx in range(num_patches):
                patch_indices.append((dataset_idx, patch_idx))
        else:
            D, H, W = dataset.shape
            pd, ph, pw = patch_size
            sd, sh, sw = stride
            d_indices = range(0, D - pd + 1, sd)
            h_indices = range(0, H - ph + 1, sh)
            w_indices = range(0, W - pw + 1, sw)
            for d in d_indices:
                for h in h_indices:
                    for w in w_indices:
                        patch_indices.append((dataset_idx, d, h, w))
    return patch_indices


def save_patches_to_zarr(patch_list, output_path):
    """
    Save a list of filtered patches (NumPy arrays) to a Zarr dataset.

    Parameters
    ----------
    patch_list : list of np.ndarray
        List of 3D patches as NumPy arrays.
    output_path : str
        Path to save the Zarr dataset.

    Returns
    -------
    str
        Path to the created Zarr dataset.
    """
    if not patch_list:
        raise ValueError("The patch list is empty. No data to save.")
    patch_shape = patch_list[0].shape
    dtype = patch_list[0].dtype
    for patch in patch_list:
        if patch.shape != patch_shape:
            raise ValueError("All patches must have the same shape.")
    store = zarr.DirectoryStore(output_path)
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    zarr_dataset = zarr.open(
        store,
        mode="w",
        shape=(len(patch_list), *patch_shape),
        chunks=(1, *patch_shape),
        dtype=dtype,
        compressor=compressor,
    )
    for i, patch in enumerate(patch_list):
        zarr_dataset[i] = patch
    logging.info("Saved %d patches to Zarr dataset at %s", len(patch_list), output_path)
    return output_path


def save_as_zarr(image_paths, patch_size, stride, filter_):
    """
    Save patches from a list of image paths as Zarr datasets.

    Parameters
    ----------
    image_paths : list
        List of image file paths.
    patch_size : tuple
        Size of the patch.
    stride : tuple
        Stride for patch extraction.
    filter_ : float
        Fraction of lowest-variance patches to filter out.

    Returns
    -------
    list
        List of paths to created Zarr datasets.
    """
    zarr_paths = []
    for img_path in image_paths:
        file_name = os.path.basename(img_path)[:-4]
        save_dir = os.path.dirname(img_path)
        patches = build_patches(img_path, patch_size, stride, filter_)
        save_patches_to_zarr(patches, os.path.join(save_dir, file_name + ".zarr"))
        zarr_paths.append(os.path.join(save_dir, file_name + ".zarr"))
    return zarr_paths
