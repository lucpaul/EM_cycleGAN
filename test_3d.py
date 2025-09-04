"""
Memory-efficient 3D inference script for image-to-image translation with 3D patches, using tile-and-stitch or overlap-averaging.

This version loads only the required patches for each batch, on demand, and pads only edge patches as needed, matching the indices from compute_patch_indices.
"""

import logging
import math
import os
import numpy as np
import tifffile
import torch
from torchvision import transforms
from tqdm import tqdm
import zarr
import shutil
from torch import nn
from data import create_dataset
from data.patched_3d_zarr_dataset import compute_patch_indices
from util.util import calculate_padding
from data.SliceBuilder import build_slices_3d, build_slices_zarr_3d
from models import create_model
from options.test_options import TestOptions
from util.util import adjust_patch_size


def load_patches_batch_3d(zarr_data, batch_indices, patch_size, padding):
    """
    For each patch index (d, h, w) in the padded array, extract the corresponding region from the raw array and pad as needed.
    The indices are assumed to be for the padded array as returned by compute_patch_indices.
    Ensures all returned patches are exactly patch_size.
    """
    pd, ph, pw = patch_size
    pad_d0, pad_d1, pad_h0, pad_h1, pad_w0, pad_w1 = padding
    D, H, W = zarr_data.shape
    patches = []
    # print("batch indices: ", batch_indices)
    for d, h, w in batch_indices:
        d0 = d - pad_d0
        h0 = h - pad_h0
        w0 = w - pad_w0
        d1 = d0 + pd
        h1 = h0 + ph
        w1 = w0 + pw

        pad_before = [0, 0, 0]
        pad_after = [0, 0, 0]
        if d0 < 0:
            pad_before[0] = -d0
            d0 = 0
        if d1 > D:
            pad_after[0] = d1 - D
            d1 = D
        if h0 < 0:
            pad_before[1] = -h0
            h0 = 0
        if h1 > H:
            pad_after[1] = h1 - H
            h1 = H
        if w0 < 0:
            pad_before[2] = -w0
            w0 = 0
        if w1 > W:
            pad_after[2] = w1 - W
            w1 = W

        patch = zarr_data[d0:d1, h0:h1, w0:w1]
        pad_width = ((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]), (pad_before[2], pad_after[2]))
        if any(pad_before) or any(pad_after):
            # If any axis is empty, use constant mode to avoid reflect error
            if 0 in patch.shape:
                patch = np.pad(patch, pad_width, mode="constant")
            else:
                patch = np.pad(patch, pad_width, mode="reflect")
        # Ensure patch is exactly (pd, ph, pw)
        pad_d = max(0, pd - patch.shape[0])
        pad_h = max(0, ph - patch.shape[1])
        pad_w = max(0, pw - patch.shape[2])
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            patch = np.pad(
                patch,
                ((0, pad_d), (0, pad_h), (0, pad_w)),
                mode="constant",
            )
        # Crop if patch is too large (should not happen, but for safety)
        patch = patch[:pd, :ph, :pw]
        patches.append(torch.tensor(patch).unsqueeze(0))
    return patches


def inference(opt):
    # --- Load model settings and check mode ---
    with open(os.path.join(opt.name, "train_opt.txt"), "r") as f:
        model_settings = f.read().splitlines()
    dicti = {line.split(":")[0].replace(" ", ""): line.split(":")[1].replace(" ", "") for line in model_settings[1:-1]}
    opt.netG = dicti["netG"]
    if opt.netG.startswith(("unet", "resnet")):
        opt.ngf = int(dicti["ngf"])
    assert dicti["train_mode"] == "3d", "For 3D predictions, the model needs to be a 3D model."

    # --- Set patch and stride parameters based on stitch mode ---
    patch_size = opt.patch_size
    stride = opt.patch_size
    init_padding = 0
    if opt.stitch_mode == "tile-and-stitch":
        adjust_patch_size(opt)
        patch_size = opt.patch_size
        stride = patch_size
        difference = sum(2**i for i in range(2, int(math.log(int(opt.netG[5:]), 2) + 2)))
        stride = patch_size - difference - 2
        init_padding = (patch_size - stride) // 2
    elif opt.stitch_mode == "overlap-averaging":
        adjust_patch_size(opt)
        patch_size = opt.patch_size
        stride = patch_size - opt.patch_overlap
        patch_halo = (opt.patch_halo, opt.patch_halo, opt.patch_halo)

    # --- Create dataset and model ---
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    # --- Set padding for Conv layers based on stitch mode ---
    for module in model.netG.modules():
        if isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
            module.padding = (0, 0, 0) if opt.stitch_mode in ("tile-and-stitch", "valid-no-crop") else (1, 1, 1)

    # --- Initialize wandb if requested ---
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo="CycleGAN-and-pix2pix")

    # --- Set model to eval mode and define transform ---
    model.eval()
    transform = transforms.Compose([transforms.ConvertImageDtype(dtype=torch.float32)])
    for data in dataset:
        # --- Prepare prediction map and input list ---
        prediction_map = None
        prediction_zarr = None
        normalization_zarr = None
        if isinstance(data["A"], np.ndarray):
            input_list = data["A"][0]
        elif isinstance(data["A"], list) and opt.use_zarr:
            zarr_path = data["A"][0]
            init_pad = np.asarray([init_padding, init_padding, init_padding])
            patch_sz = np.asarray([patch_size, patch_size, patch_size])
            stride_arr = np.asarray([stride, stride, stride])
            # Set a small chunk cache size for Zarr
            store = zarr.DirectoryStore(zarr_path)
            zarr_data = zarr.open(store, mode="r")
            zarr_str = "zarr"
            stitch_str = opt.stitch_mode.replace("-", "_")
            base = os.path.basename(data["A"][0][:-5])
            out_name = f"generated_{opt.test_mode}_{zarr_str}_{stitch_str}_{base}"
            pred_zarr_path = os.path.join(opt.results_dir, out_name + ".zarr")

            if hasattr(zarr_data, "chunk_store") and hasattr(zarr_data.chunk_store, "cache"):
                zarr_data.chunk_store.cache._max_size = 256 * 1024 * 1024  # 32MB
            calc_pad = calculate_padding(zarr_data.shape, init_pad, patch_sz, stride_arr, dim=None)
            padding = (init_pad[0], calc_pad[0], init_pad[1], calc_pad[1], init_pad[2], calc_pad[2])
            input_list, raw_size, padded_size = compute_patch_indices(zarr_data, padding, patch_sz, stride_arr)
            # print("Input List: ", input_list)

            if not data.get("IsTiff", False):
                data["A_full_size_raw"] = raw_size
                data["A_full_size_pad"] = padded_size

        else:
            input_list = data["A"]
            # print("Input List: ", input_list)

        prediction_slices = None
        for i in tqdm(range(0, math.ceil(len(input_list) / opt.batch_size)), desc="Inference progress"):
            # --- Prepare input tensor for this batch ---
            if opt.use_zarr:
                input_batch_indices = input_list[i * opt.batch_size : i * opt.batch_size + opt.batch_size]
                batch_patches = load_patches_batch_3d(zarr_data, input_batch_indices, (patch_size, patch_size, patch_size), padding)
                input_tensor = torch.stack(batch_patches)
            else:
                input_tensor = torch.cat(input_list[i * opt.batch_size : i * opt.batch_size + opt.batch_size], dim=0)
            input_tensor = transform(input_tensor)
            if isinstance(data["A"], np.ndarray):
                input_tensor = torch.unsqueeze(input_tensor, 0)
            if opt.stitch_mode == "overlap-averaging":
                input_tensor = _pad(input_tensor, patch_halo)

            model.set_input(input_tensor)
            model.test()
            img_batched = model.fake

            for b in range(opt.batch_size):
                if b + i * opt.batch_size >= len(input_list):
                    break

                img = torch.unsqueeze(img_batched[b], 0)

                # --- Initialize prediction/normalization maps and slices for new volume ---
                if prediction_map is None:
                    if opt.stitch_mode == "tile-and-stitch":
                        size_0 = stride * math.ceil(((data["A_full_size_pad"][0] - patch_size) / stride) + 1)
                        size_1 = stride * math.ceil(((data["A_full_size_pad"][1] - patch_size) / stride) + 1)
                        size_2 = stride * math.ceil(((data["A_full_size_pad"][2] - patch_size) / stride) + 1)
                    else:
                        size_0 = data["A_full_size_pad"][0]
                        size_1 = data["A_full_size_pad"][1]
                        size_2 = data["A_full_size_pad"][2]

                    # Create Zarr arrays for prediction and normalization

                    norm_zarr_path = os.path.join(opt.results_dir, "norm_zarr_tmp.zarr")
                    prediction_zarr = zarr.open(pred_zarr_path, mode="w", shape=(size_0, size_1, size_2), dtype=np.float32, chunks=(32, 32, 32))
                    prediction_map = prediction_zarr  # for compatibility
                    if opt.stitch_mode == "overlap-averaging":
                        normalization_zarr = zarr.open(norm_zarr_path, mode="w", shape=(size_0, size_1, size_2), dtype=np.uint8, chunks=(32, 32, 32))
                        normalization_map = normalization_zarr
                    # Build slices as before
                    if opt.stitch_mode == "tile-and-stitch":
                        if opt.use_zarr:
                            prediction_slices = build_slices_zarr_3d(
                                prediction_map,
                                [0, 0, 0, 0, 0, 0],
                                np.asarray([stride, stride, stride]),
                                np.asarray([stride, stride, stride]),
                            )
                        else:
                            prediction_slices = build_slices_3d(
                                prediction_map,
                                [stride, stride, stride],
                                [stride, stride, stride],
                            )
                    elif opt.stitch_mode in ("valid-no-crop", "overlap-averaging"):
                        if opt.stitch_mode == "overlap-averaging":
                            if opt.use_zarr:
                                prediction_slices = build_slices_zarr_3d(
                                    prediction_map,
                                    [0, 0, 0, 0, 0, 0],
                                    np.asarray([patch_size, patch_size, patch_size]),
                                    np.asarray([stride, stride, stride]),
                                )
                            else:
                                prediction_slices = build_slices_3d(
                                    prediction_map,
                                    [patch_size, patch_size, patch_size],
                                    [stride, stride, stride],
                                )
                        else:
                            if opt.use_zarr:
                                prediction_slices = build_slices_zarr_3d(
                                    prediction_map,
                                    [0, 0, 0, 0, 0, 0],
                                    np.asarray([patch_size, patch_size, patch_size]),
                                    np.asarray([patch_size, patch_size, patch_size]),
                                )
                            else:
                                prediction_slices = build_slices_3d(
                                    prediction_map,
                                    [patch_size, patch_size, patch_size],
                                    [patch_size, patch_size, patch_size],
                                )

                # --- Crop/unpad patch if needed and convert to numpy ---
                if opt.stitch_mode == "tile-and-stitch":
                    img = img[:, :, init_padding:-init_padding, init_padding:-init_padding, init_padding:-init_padding]
                elif opt.stitch_mode == "overlap-averaging":
                    img = _unpad(img, patch_halo)
                img = torch.squeeze(torch.squeeze(img, 0), 0).cpu().float().numpy()

                # --- Update normalization map if overlap-averaging ---
                if opt.stitch_mode == "overlap-averaging":
                    if opt.use_zarr:
                        pred_d, pred_h, pred_w = prediction_slices[i * opt.batch_size + b][:3]
                        normalization_map[pred_d : pred_d + patch_size, pred_h : pred_h + patch_size, pred_w : pred_w + patch_size] += 1
                    else:
                        normalization_map[prediction_slices[i * opt.batch_size + b]] += 1

                # --- Add patch prediction to prediction map ---
                if opt.use_zarr:
                    pred_d, pred_h, pred_w = prediction_slices[i * opt.batch_size + b][:3]
                    if pred_d < prediction_map.shape[0] and pred_h < prediction_map.shape[1] and pred_w < prediction_map.shape[2]:
                        if opt.stitch_mode == "tile-and-stitch":
                            prediction_map[pred_d : pred_d + stride, pred_h : pred_h + stride, pred_w : pred_w + stride] += img
                        else:
                            prediction_map[pred_d : pred_d + patch_size, pred_h : pred_h + patch_size, pred_w : pred_w + patch_size] += img

                else:
                    prediction_map[prediction_slices[i * opt.batch_size + b]] += img

        # --- Normalize and save prediction map ---
        if opt.use_zarr:
            if opt.stitch_mode == "overlap-averaging":
                prediction_zarr[:] = prediction_zarr[:] / normalization_zarr[:]
            # Write TIFF slice-by-slice to avoid memory spikes
            arr_shape = prediction_zarr.shape

            if not getattr(opt, "no_tiff", False):
                base = os.path.basename(data["A"][0][:-5])
                out_name = f"generated_{opt.test_mode}_{zarr_str}_{stitch_str}_{base}.tif"
                tiff_path = os.path.join(opt.results_dir, out_name)

                # Write in blocks of slices that fit in ~5GB RAM

                dtype_bytes = np.dtype(np.float32).itemsize
                z_dim = data["A_full_size_raw"][0]
                y_dim = data["A_full_size_raw"][1]
                x_dim = data["A_full_size_raw"][2]
                max_bytes = 5 * 1024**3  # 5GB
                slices_per_block = max(1, max_bytes // (y_dim * x_dim * dtype_bytes))
                # Compute global min/max for scaling
                global_min = float("inf")
                global_max = float("-inf")
                for z_start in range(0, z_dim, slices_per_block):
                    z_end = min(z_start + slices_per_block, z_dim)
                    block = prediction_zarr[z_start:z_end, :y_dim, :x_dim]
                    block_min = block.min()
                    block_max = block.max()
                    if block_min < global_min:
                        global_min = block_min
                    if block_max > global_max:
                        global_max = block_max
                # Write blocks with global scaling
                with tifffile.TiffWriter(tiff_path, bigtiff=True) as tif:
                    for z_start in range(0, z_dim, slices_per_block):
                        print(f"Writing block {z_start // slices_per_block + 1}")
                        z_end = min(z_start + slices_per_block, z_dim)
                        block = prediction_zarr[z_start:z_end, :y_dim, :x_dim]
                        if global_max > global_min:
                            block = (255 * (block - global_min) / (global_max - global_min)).astype(np.uint8)
                        else:
                            block = np.zeros_like(block, dtype=np.uint8)
                        tif.write(block, contiguous=True)
            # Clean up temp zarr files

            if normalization_zarr is not None:
                shutil.rmtree(normalization_zarr.store.path, ignore_errors=True)
            # --- Explicitly close the Zarr array to free resources ---

            zarr_data.store.close()
        else:
            # RAM-based logic for non-zarr mode
            if opt.stitch_mode == "overlap-averaging":
                prediction_map = prediction_map / normalization_map
            arr = (255 * (prediction_map - prediction_map.min()) / (prediction_map.max() - prediction_map.min())).astype(np.uint8)
            arr = arr[
                0 : data["A_full_size_raw"][0],
                0 : data["A_full_size_raw"][1],
                0 : data["A_full_size_raw"][2],
            ]
            mode = "3d"
            zarr_str = "tiff"
            stitch_str = opt.stitch_mode.replace("-", "_")
            base = os.path.basename(data["A_paths"][0])
            out_name = f"generated_{mode}_{zarr_str}_{stitch_str}_{base}.tif"
            tifffile.imwrite(os.path.join(opt.results_dir, out_name), arr)


def _pad(m, patch_halo):
    if patch_halo is not None:
        z, y, x = patch_halo
        return nn.functional.pad(m, (x, x, y, y, z, z), mode="reflect")
    return m


def _unpad(m, patch_halo):
    if patch_halo is not None:
        z, y, x = patch_halo
        if z == 0:
            return m[..., y:-y, x:-x]
        else:
            return m[..., z:-z, y:-y, x:-x]
    return m


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    opt = TestOptions().parse()
    opt.num_threads = 8 if opt.use_zarr else 0
    opt.dataset_mode = "patched_3d_zarr" if opt.use_zarr else "patched_3d"
    opt.test_mode = "3d"
    opt.input_nc = opt.output_nc = 1
    opt.serial_batches = opt.no_flip = True
    inference(opt)
    TestOptions().save_options(opt)
