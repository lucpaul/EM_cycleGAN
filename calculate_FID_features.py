from util.fid_score import save_fid_stats
import tifffile
import os
import torch
from data.SliceBuilder import build_slices
import random


def fid_features(input_dir, save_path, n_samples=2000):
    patches = []
    for img in os.listdir(input_dir):
        dataset = tifffile.imread(os.path.join(input_dir, img))
        dataset = torch.from_numpy(dataset)

        new_patches = []
        for z in range(0, dataset.shape[0]):
            img_slice = dataset[z]
            slices = build_slices(img_slice, [128, 128], [128, 128])
            for slice in slices:
                img_patch = img_slice[slice]
                img_patch = torch.unsqueeze(img_patch, 0)
                new_patches.append(img_patch)

        patches += new_patches

    random.seed(42)
    indices = random.sample(range(len(patches)), k=n_samples)

    patches = torch.stack(patches)
    patches = patches[indices, :, :, :]

    if input_dir.endswith('/'):
        input_dir = input_dir[:-1]

    if save_path.endswith('/'):
        save_path = save_path[:-1]

    fingerprint_path = os.path.join(save_path, "FID_features_for_"+os.path.basename(input_dir))

    paths = [list(patches), fingerprint_path]

    # Here the paths argument is a list containing a list and a path
    save_fid_stats(paths, batch_size=50, device='cuda', dims=2048)