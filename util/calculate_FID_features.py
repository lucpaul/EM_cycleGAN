import os
import random
import logging
import torch
import tifffile
import zarr
from util.fid_score import save_fid_stats
from data.SliceBuilder import build_slices, build_slices_zarr_2d


def fid_features(input_dir, save_path, n_samples=20000):
    """
    Extract FID features from a directory of .tif/.tiff and .zarr images and save them for FID calculation.

    Args:
        input_dir (str): Directory containing .tif/.tiff and/or .zarr files.
        save_path (str): Directory to save the FID features.
        n_samples (int): Number of random patches to sample (default: 20000).

    Returns:
        None
    """
    patches = []
    # Build set of base names for .zarr folders
    files = os.listdir(input_dir)
    zarr_basenames = set()
    for f in files:
        if f.endswith(".zarr"):
            zarr_basenames.add(os.path.splitext(f)[0])

    for img in files:
        base, ext = os.path.splitext(img)
        # Skip .tif/.tiff if .zarr with same base exists
        if (ext in [".tif", ".tiff"]) and (base in zarr_basenames):
            continue
        if ext in [".tif", ".tiff"]:
            dataset = tifffile.imread(os.path.join(input_dir, img))
            dataset = torch.from_numpy(dataset)
        elif ext == ".zarr":
            dataset = zarr.open(os.path.join(input_dir, img), mode="r")
        else:
            continue

        logging.info("Dataset shape: %s", dataset.shape)
        for z in range(0, dataset.shape[0]):
            new_patches = []
            img_slice = dataset[z]
            if ext in [".tif", ".tiff"]:
                slices = build_slices(img_slice, [128, 128], [128, 128], use_shape_only=False)
                for slice in slices:
                    img_patch = img_slice[slice]
                    img_patch = torch.unsqueeze(img_patch, 0)
                    new_patches.append(img_patch)
                patches += new_patches
            elif ext == ".zarr":
                slice_indices = build_slices_zarr_2d(img_slice, [0, 0, 0, 0], [128, 128], [128, 128])
                for slice in slice_indices:
                    img_patch = img_slice[slice[0] : slice[0] + 128, slice[1] : slice[1] + 128]
                    img_patch = torch.tensor(img_patch)
                    img_patch = torch.unsqueeze(img_patch, 0)
                    new_patches.append(img_patch)
                patches += new_patches

    random.seed(42)

    # Default index sampler
    indices = random.sample(range(len(patches)), k=n_samples)

    # This should only be used for testing or if the dataset is too small to run a proper test,
    # as it allows duplicate patches:
    # indices = random.choices(range(len(patches)), k=n_samples)

    patches = torch.stack(patches)

    patches = patches[indices, :, :, :]

    if input_dir.endswith("/"):
        input_dir = input_dir[:-1]

    if save_path.endswith("/"):
        save_path = save_path[:-1]

    fingerprint_path = os.path.join(save_path, "FID_features_for_" + os.path.basename(input_dir))

    paths = [list(patches), fingerprint_path]

    # Here the paths argument is a list containing a list and a path
    save_fid_stats(paths, batch_size=50, device="cuda", dims=2048)
