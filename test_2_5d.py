"""Sample script for image-to-image translation with 2.5D patches, using optimized batch patch loading from Zarr datasets.

This version writes each direction's slices to Zarr on disk (if --use_zarr), then combines and saves as TIFF (unless --no_tiff). For non-Zarr, uses fast in-memory approach.
"""

import logging
import os
import math
import numpy as np
import tifffile
import torch
from tqdm import tqdm
from torchvision import transforms
import zarr
from options.test_options import TestOptions
import argparse
from data import create_dataset
from models import create_model
from util.util import adjust_patch_size
from data.SliceBuilder import build_slices, build_slices_zarr_2d
from data.patched_2_5d_zarr_dataset import compute_patch_indices
from util.util import calculate_padding
from collections import defaultdict

try:
    import wandb
except ImportError:
    logging.warning('wandb package cannot be found. The option "--use_wandb" will result in error.')


def load_patches_batch_2_5d(zarr_data, batch_indices, patch_size, padding, direction):
    ph, pw = patch_size
    patches_by_slice = defaultdict(list)
    for idx, (s, h, w) in enumerate(batch_indices):
        patches_by_slice[s].append((idx, h, w))
    patches = [None] * len(batch_indices)
    for s, patch_list in patches_by_slice.items():
        if direction == "xy":
            slice_data = zarr_data[s, :, :]
            slice_data = np.pad(
                slice_data,
                pad_width=((padding[1]), (padding[2])),
                mode="reflect",
            )
        elif direction == "zx":
            slice_data = zarr_data[:, :, s]
            slice_data = np.pad(
                slice_data,
                pad_width=((padding[0]), (padding[1])),
                mode="reflect",
            )
        elif direction == "zy":
            slice_data = zarr_data[:, s, :]
            slice_data = np.pad(
                slice_data,
                pad_width=((padding[0]), (padding[2])),
                mode="reflect",
            )
        for idx, h, w in patch_list:
            patch = slice_data[h : h + ph, w : w + pw]
            patches[idx] = torch.tensor(patch).unsqueeze(0)
    return patches


def inference_2_5D(opt):
    # ...existing code up to dataset/model creation...
    with open(os.path.join(opt.name, "train_opt.txt"), "r") as f:
        model_settings = f.read().splitlines()
    dicti = {line.split(":")[0].replace(" ", ""): line.split(":")[1].replace(" ", "") for line in model_settings[1:-1]}
    opt.netG = dicti["netG"]
    opt.ngf = int(dicti["ngf"])
    assert dicti["train_mode"] == "2d", "For 2.5D (orthoslice) predictions, the model needs to be a 2D model."

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
        patch_halo = (opt.patch_halo, opt.patch_halo)

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    for module in model.netG.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            module.padding = (0, 0) if opt.stitch_mode in ("tile-and-stitch", "valid-no-crop") else (1, 1)

    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo="CycleGAN-and-pix2pix")

    model.eval()
    transform = transforms.Compose([transforms.ConvertImageDtype(dtype=torch.float32)])

    for data in dataset:
        ortho_planes = ["xy", "zy", "zx"]
        prediction_volume = []
        zarr_slices = {}  # For zarr mode, store per-direction zarr arrays
        for direction in ortho_planes:
            dir_volume = []
            if opt.use_zarr:
                zarr_path = data["A"][0]
                zarr_data = zarr.open(zarr_path, mode="r", cache_size=500 * 1024 * 1024)
                # Padding and patch info
                calc_padding = calculate_padding(
                    zarr_data.shape,
                    np.asarray([init_padding, init_padding, init_padding]),
                    np.asarray([patch_size, patch_size, patch_size]),
                    np.asarray([stride, stride, stride]),
                    dim=None,
                )
                padding = (
                    (init_padding, calc_padding[0]),
                    (init_padding, calc_padding[1]),
                    (init_padding, calc_padding[2]),
                )
                # Compute patch indices for this direction
                input_list, patches_per_slice, raw_size, padded_size = compute_patch_indices(
                    zarr_data,
                    padding,
                    np.asarray([patch_size, patch_size]),
                    np.asarray([stride, stride]),
                    direction,
                )
                if not data.get("IsTiff", False):
                    data["A_full_size_raw"] = raw_size
                    data["A_full_size_pad"] = padded_size
                data["patches_per_slice"] = patches_per_slice
                full_pad_dim_1 = padded_size[1]
                full_pad_dim_2 = padded_size[2]
                num_slices = int(len(input_list) // patches_per_slice)

                size_0, size_1 = calculate_output_sizes(
                    opt.stitch_mode,
                    (full_pad_dim_1, full_pad_dim_2),
                    stride,
                    patch_size,
                )

                out_shape = (num_slices, size_0, size_1)
                # Prepare Zarr array for this direction
                base = os.path.basename(data["A"][0][:-5])
                out_name = f"generated_2_5d_zarr_{opt.stitch_mode.replace('-', '_')}_{base}_{direction}"
                out_zarr_path = os.path.join(opt.results_dir, out_name + ".zarr")
                out_zarr = zarr.open(out_zarr_path, mode="w", shape=out_shape, dtype="uint8")
                zarr_slices[direction] = (out_zarr, out_zarr_path)
            else:
                if direction == "xy":
                    full_pad_dim_1 = data["A_full_size_pad"][1]
                    full_pad_dim_2 = data["A_full_size_pad"][2]
                elif direction == "zy":
                    full_pad_dim_1 = data["A_full_size_pad"][0]
                    full_pad_dim_2 = data["A_full_size_pad"][1]
                elif direction == "zx":
                    full_pad_dim_1 = data["A_full_size_pad"][2]
                    full_pad_dim_2 = data["A_full_size_pad"][0]
                input_list = data[direction]
                size_0, size_1 = calculate_output_sizes(
                    opt.stitch_mode,
                    (full_pad_dim_1, full_pad_dim_2),
                    stride,
                    patch_size,
                )
            prediction_map = None
            prediction_slices = None
            pred_index = 0
            slice_idx = 0
            for i in tqdm(
                range(0, math.ceil(len(input_list) / opt.batch_size)),
                desc=f"Inference {direction}",
            ):
                if opt.use_zarr:
                    input_batch_indices = input_list[i * opt.batch_size : i * opt.batch_size + opt.batch_size]
                    batch_patches = load_patches_batch_2_5d(
                        zarr_data,
                        input_batch_indices,
                        (patch_size, patch_size),
                        padding,
                        direction,
                    )
                    input_tensor = torch.stack(batch_patches)
                else:
                    input_tensor = torch.cat(
                        input_list[i * opt.batch_size : i * opt.batch_size + opt.batch_size],
                        dim=0,
                    )
                input_tensor = transform(input_tensor)
                if opt.stitch_mode == "overlap-averaging":
                    input_tensor = _pad(input_tensor, patch_halo)
                model.set_input(input_tensor)
                model.test()
                img_batched = model.fake
                for b in range(opt.batch_size):
                    if b + i * opt.batch_size >= len(input_list):
                        break
                    patch_index = i * opt.batch_size + b
                    img = torch.unsqueeze(img_batched[b], 0)
                    if opt.stitch_mode == "tile-and-stitch":
                        img = img[:, :, init_padding:-init_padding, init_padding:-init_padding]

                    if not opt.use_zarr:
                        patches_per_slice = int(data["patches_per_slice" + "_" + direction])
                    if patch_index % patches_per_slice == 0:
                        if patch_index != 0:
                            if opt.stitch_mode == "overlap-averaging":
                                out_slice = (255 * prediction_map / normalization_map).astype(np.uint8)
                            else:
                                out_slice = (255 * prediction_map).astype(np.uint8)

                            if opt.use_zarr:
                                zarr_slices[direction][0][slice_idx - 1, :, :] = out_slice
                                del prediction_map
                                del normalization_map
                            else:
                                # out_slice = (255 * prediction_map).astype(np.uint8)
                                dir_volume.append(out_slice)
                                del prediction_map
                                del normalization_map
                        prediction_map = np.zeros((size_0, size_1), dtype=np.float32)
                        normalization_map = np.zeros((size_0, size_1), dtype=np.uint8)
                        if opt.stitch_mode == "tile-and-stitch":
                            if opt.use_zarr:
                                prediction_slices = build_slices_zarr_2d(
                                    prediction_map,
                                    [0, 0, 0, 0],
                                    np.asarray([stride, stride]),
                                    np.asarray([stride, stride]),
                                )
                            else:
                                prediction_slices = build_slices(
                                    prediction_map,
                                    [stride, stride],
                                    [stride, stride],
                                    use_shape_only=False,
                                )
                        elif opt.stitch_mode in ("valid-no-crop", "overlap-averaging"):
                            if opt.stitch_mode == "overlap-averaging":
                                normalization_map = np.zeros((size_0, size_1), dtype=np.uint8)
                            if opt.use_zarr:
                                if opt.stitch_mode == "overlap-averaging":
                                    prediction_slices = build_slices_zarr_2d(
                                        prediction_map,
                                        [0, 0, 0, 0],
                                        np.asarray([patch_size, patch_size]),
                                        np.asarray([stride, stride]),
                                    )
                                else:
                                    prediction_slices = build_slices_zarr_2d(
                                        prediction_map,
                                        [0, 0, 0, 0],
                                        np.asarray([patch_size, patch_size]),
                                        np.asarray([patch_size, patch_size]),
                                    )
                            else:
                                if opt.stitch_mode == "overlap-averaging":
                                    prediction_slices = build_slices(
                                        prediction_map,
                                        [patch_size, patch_size],
                                        [stride, stride],
                                        use_shape_only=False,
                                    )
                                else:
                                    prediction_slices = build_slices(
                                        prediction_map,
                                        [patch_size, patch_size],
                                        [patch_size, patch_size],
                                        use_shape_only=False,
                                    )
                        pred_index = 0
                        slice_idx += 1
                    if opt.stitch_mode == "overlap-averaging":
                        img = _unpad(img, patch_halo)
                    img = torch.squeeze(torch.squeeze(img, 0), 0).cpu().float().numpy()
                    if opt.stitch_mode == "overlap-averaging":
                        if opt.use_zarr:
                            pred_h = prediction_slices[pred_index][0]
                            pred_w = prediction_slices[pred_index][1]
                            normalization_map[
                                pred_h : pred_h + patch_size,
                                pred_w : pred_w + patch_size,
                            ] += 1
                        else:
                            normalization_map[prediction_slices[pred_index]] += 1
                    if opt.use_zarr:
                        pred_h = prediction_slices[pred_index][0]
                        pred_w = prediction_slices[pred_index][1]
                        if pred_h < prediction_map.shape[0] and pred_w < prediction_map.shape[1]:
                            if opt.stitch_mode == "tile-and-stitch":
                                prediction_map[pred_h : pred_h + stride, pred_w : pred_w + stride] += img
                            else:
                                prediction_map[
                                    pred_h : pred_h + patch_size,
                                    pred_w : pred_w + patch_size,
                                ] += img
                    else:
                        prediction_map[prediction_slices[pred_index]] += img
                    pred_index += 1
                    if patch_index == len(input_list) - 1:
                        if opt.use_zarr:
                            out_slice = (255 * prediction_map).astype(np.uint8)
                            zarr_slices[direction][0][slice_idx - 1, :, :] = out_slice
                            del prediction_map
                            del normalization_map
                        else:
                            out_slice = (255 * prediction_map).astype(np.uint8)
                            dir_volume.append(out_slice)
                            del prediction_map
                            del normalization_map
            if not opt.use_zarr:
                dir_volume = np.asarray(dir_volume)
                prediction_volume.append(dir_volume)
        # --- Combine, transpose, crop, and save output ---
        mode = "2_5d"
        stitch_str = opt.stitch_mode.replace("-", "_")

        crop_slices_xy = [
            # slice(0, data["A_full_size_raw"][0]),
            slice(init_padding, data["A_full_size_raw"][0] + init_padding),
            slice(0, data["A_full_size_raw"][1]),
            slice(0, data["A_full_size_raw"][2]),
        ]

        crop_slices_zy = [
            slice(init_padding, data["A_full_size_raw"][2] + init_padding),
            slice(0, data["A_full_size_raw"][0]),
            slice(0, data["A_full_size_raw"][1]),
        ]

        crop_slices_zx = [
            slice(init_padding, data["A_full_size_raw"][1] + init_padding),
            slice(0, data["A_full_size_raw"][0]),
            slice(0, data["A_full_size_raw"][2]),
        ]
        if opt.use_zarr:
            # Load all zarr arrays, transpose, crop, and combine
            combined = []
            for i, direction in enumerate(ortho_planes):
                arr = zarr_slices[direction][0][:].astype(np.float32) / 255.0
                # Transpose to (z, y, x) for all directions

                if direction == "xy":
                    arr = arr[:, crop_slices_xy[1], crop_slices_xy[2]]
                    # arr = arr  # already (z, y, x)
                elif direction == "zy":
                    arr = arr[:, crop_slices_zx[1], crop_slices_zx[2]]
                    arr = np.transpose(arr, (1, 0, 2))  # (y, z, x) -> (z, y, x)
                elif direction == "zx":
                    arr = arr[:, crop_slices_zy[1], crop_slices_zy[2]]
                    arr = np.transpose(arr, (1, 2, 0))  # (x, y, z) -> (z, y, x)

                combined.append(arr)
            # Normalize and average
            for i in range(3):
                combined[i] = (combined[i] - combined[i].min()) / (combined[i].max() - combined[i].min() + 1e-8)
            prediction_volume_full = (sum(combined) / 3 * 255).astype(np.uint8)
            base = os.path.basename(data["A"][0][:-5])
            out_name = f"generated_{mode}_zarr_{stitch_str}_{base}"
            if not getattr(opt, "no_tiff", False):
                tifffile.imwrite(
                    os.path.join(opt.results_dir, out_name + ".tif"),
                    prediction_volume_full,
                )
        else:
            # In-memory, fast approach
            for i, direction in enumerate(ortho_planes):
                arr = np.asarray(prediction_volume[i])
                if direction == "xy":
                    arr = arr[crop_slices_xy[0], crop_slices_xy[1], crop_slices_xy[2]]
                elif direction == "zy":
                    arr = arr[crop_slices_zy[0], crop_slices_zy[1], crop_slices_zy[2]]
                    arr = np.transpose(arr, (1, 2, 0))
                elif direction == "zx":
                    arr = arr[crop_slices_zx[0], crop_slices_zx[1], crop_slices_zx[2]]
                    arr = np.transpose(arr, (1, 0, 2))

                prediction_volume[i] = arr
            for i in range(3):
                prediction_volume[i] = (prediction_volume[i] - prediction_volume[i].min()) / (
                    prediction_volume[i].max() - prediction_volume[i].min() + 1e-8
                )
            prediction_volume_full = (sum(prediction_volume) / 3 * 255).astype(np.uint8)
            base = os.path.basename(data["A_paths"][0])
            out_name = f"generated_{mode}_tiff_{stitch_str}_{base}"
            tifffile.imwrite(
                os.path.join(opt.results_dir, out_name),
                prediction_volume_full,
            )


def _pad(m, patch_halo):
    if patch_halo is not None:
        y, x = patch_halo
        return torch.nn.functional.pad(m, (y, y, x, x), mode="reflect")
    return m


def _unpad(m, patch_halo):
    if patch_halo is not None:
        y, x = patch_halo
        return m[..., y:-y, x:-x]
    return m


def calculate_output_sizes(stitch_mode, input_size, stride, patch_size):

    if stitch_mode == "tile-and-stitch":
        size_0 = stride * math.ceil(((input_size[0] - patch_size) / stride) + 1)
        size_1 = stride * math.ceil(((input_size[1] - patch_size) / stride) + 1)
    else:
        size_0 = input_size[0]
        size_1 = input_size[1]

    return size_0, size_1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_tiff",
        action="store_true",
        help="Do not save output as TIFF when using Zarr",
    )
    opt = TestOptions().parse()
    args, _ = parser.parse_known_args()
    opt.no_tiff = args.no_tiff
    opt.num_threads = 12 if opt.use_zarr else 0
    opt.dataset_mode = "patched_2_5d_zarr" if opt.use_zarr else "patched_2_5d"
    opt.input_nc = opt.output_nc = 1
    opt.serial_batches = opt.no_flip = True
    inference_2_5D(opt)
    TestOptions().save_options(opt)
