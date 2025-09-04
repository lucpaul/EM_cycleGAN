"""Sample script for image-to-image translation with 2D patches, using optimized batch patch loading from Zarr datasets.

This version groups patch indices by slice and loads each slice only once per batch, reducing redundant I/O.
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
from data import create_dataset
from data.patched_2d_zarr_dataset import compute_patch_indices
from util.util import calculate_padding
from data.SliceBuilder import build_slices, build_slices_zarr_2d
from models import create_model
from options.test_options import TestOptions
import argparse
from util.util import adjust_patch_size
from collections import defaultdict

try:
    import wandb
except ImportError:
    logging.warning('wandb package cannot be found. The option "--use_wandb" will result in error.')


def load_patches_batch(zarr_data, batch_indices, patch_size, padding):
    """
    Efficiently load a batch of patches from a Zarr dataset by grouping by slice.
    """
    ph, pw = patch_size
    patches_by_slice = defaultdict(list)
    for idx, (s, h, w) in enumerate(batch_indices):
        patches_by_slice[s].append((idx, h, w))
    patches = [None] * len(batch_indices)
    for s, patch_list in patches_by_slice.items():
        slice_data = zarr_data[s, :, :]
        slice_data_pad = np.pad(
            slice_data,
            pad_width=((padding[0], padding[1]), (padding[2], padding[3])),
            mode="reflect",
        )
        for idx, h, w in patch_list:
            patch = slice_data_pad[h : h + ph, w : w + pw]
            patches[idx] = torch.tensor(patch).unsqueeze(0)
    return patches


def inference(opt):
    # --- Load model settings and check mode ---
    with open(os.path.join(opt.name, "train_opt.txt"), "r") as f:
        model_settings = f.read().splitlines()
    dicti = {line.split(":")[0].replace(" ", ""): line.split(":")[1].replace(" ", "") for line in model_settings[1:-1]}
    opt.netG = dicti["netG"]
    if opt.netG.startswith(("unet", "resnet")):
        opt.ngf = int(dicti["ngf"])
    assert dicti["train_mode"] == "2d", "For 2D predictions, the model needs to be a 2D model."

    # --- Set patch and stride parameters based on stitch mode ---
    patch_size = opt.patch_size
    stride = opt.patch_size
    init_padding = 0
    if opt.stitch_mode == "tile-and-stitch":
        adjust_patch_size(opt)
        patch_size = opt.patch_size  # update after adjust_patch_size
        stride = patch_size
        difference = sum(2**i for i in range(2, int(math.log(int(opt.netG[5:]), 2) + 2)))
        stride = patch_size - difference - 2
        init_padding = (patch_size - stride) // 2
    elif opt.stitch_mode == "overlap-averaging":
        adjust_patch_size(opt)
        patch_size = opt.patch_size  # update after adjust_patch_size
        stride = patch_size - opt.patch_overlap
        patch_halo = (opt.patch_halo, opt.patch_halo)

    # --- Create dataset and model ---
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    # --- Set padding for Conv layers based on stitch mode ---
    for module in model.netG.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            module.padding = (0, 0) if opt.stitch_mode in ("tile-and-stitch", "valid-no-crop") else (1, 1)

    # --- Initialize wandb if requested ---
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo="CycleGAN-and-pix2pix")

    # --- Set model to eval mode and define transform ---
    model.eval()
    transform = transforms.Compose([transforms.ConvertImageDtype(dtype=torch.float32)])

    for data in dataset:
        # --- Determine input list and patch info ---
        if isinstance(data["A"], np.ndarray):
            input_list = data["A"][0]
        elif isinstance(data["A"], list) and opt.use_zarr:
            zarr_path = data["A"][0]
            init_pad = np.asarray([init_padding, init_padding])
            patch_sz = np.asarray([patch_size, patch_size])
            stride_arr = np.asarray([stride, stride])
            zarr_data = zarr.open(zarr_path, mode="r")
            calc_pad_h, calc_pad_w = calculate_padding(zarr_data.shape, init_pad, patch_sz, stride_arr, dim=2)
            padding = (init_pad[0], calc_pad_h, init_pad[1], calc_pad_w)

            input_list, patches_per_slice, raw_size, padded_size = compute_patch_indices(zarr_data, padding, patch_sz, stride_arr)
            if not data.get("IsTiff", False):
                data["A_full_size_raw"] = raw_size
                data["A_full_size_pad"] = padded_size
            data["patches_per_slice"] = patches_per_slice
        else:
            input_list = data["A"]

        if opt.use_zarr:
            # --- Prepare Zarr array for output ---
            num_slices = int(len(input_list) // data["patches_per_slice"])
            out_shape = (
                (
                    num_slices,
                    data["A_full_size_pad"][1],
                    data["A_full_size_pad"][2],
                )
                if opt.stitch_mode != "tile-and-stitch"
                else (
                    num_slices,
                    data["A_full_size_raw"][1],
                    data["A_full_size_raw"][2],
                )
            )
            # --- Save Zarr with correct base name ---
            mode = "2d"
            zarr_str = "zarr"
            stitch_str = opt.stitch_mode.replace("-", "_")
            base = os.path.basename(data["A"][0][:-5])
            out_name = f"generated_{mode}_{zarr_str}_{stitch_str}_{base}"
            out_zarr_path = os.path.join(opt.results_dir, out_name + ".zarr")
            out_zarr = zarr.open(out_zarr_path, mode="w", shape=out_shape, dtype="uint8")
        else:
            prediction_volume = []

        prediction_map = None
        prediction_slices = None
        pred_index = 0
        slice_idx = 0
        for i in tqdm(
            range(0, math.ceil(len(input_list) / opt.batch_size)),
            desc="Inference progress",
        ):
            # --- Prepare input tensor for this batch ---
            if opt.use_zarr:
                input_batch_indices = input_list[i * opt.batch_size : i * opt.batch_size + opt.batch_size]
                batch_patches = load_patches_batch(zarr_data, input_batch_indices, patch_sz, padding)
                input_tensor = torch.stack(batch_patches)
            else:
                input_tensor = torch.cat(
                    input_list[i * opt.batch_size : i * opt.batch_size + opt.batch_size],
                    dim=0,
                )
            input_tensor = transform(input_tensor)
            if isinstance(data["A"], np.ndarray):
                input_tensor = torch.unsqueeze(input_tensor, 0)
            if opt.stitch_mode == "overlap-averaging":
                input_tensor = _pad(input_tensor, patch_halo)

            # --- Run model inference on input tensor ---
            model.set_input(input_tensor)
            model.test()
            img_batched = model.fake

            for b in range(opt.batch_size):
                if b + i * opt.batch_size >= len(input_list):
                    break

                # --- Prepare patch and its output location ---
                patch_index = i * opt.batch_size + b
                img = torch.unsqueeze(img_batched[b], 0)

                # --- Crop patch if needed and determine output size ---
                if opt.stitch_mode == "tile-and-stitch":
                    img = img[:, :, init_padding:-init_padding, init_padding:-init_padding]
                    size_0 = stride * math.ceil(((data["A_full_size_pad"][1] - patch_size) / stride) + 1)
                    size_1 = stride * math.ceil(((data["A_full_size_pad"][2] - patch_size) / stride) + 1)
                else:
                    size_0 = data["A_full_size_pad"][1]
                    size_1 = data["A_full_size_pad"][2]

                # --- Initialize prediction/normalization maps and slices for new slice ---
                if patch_index % int(data["patches_per_slice"]) == 0:
                    if patch_index != 0:
                        # Save previous slice to Zarr or append to prediction_volume
                        if opt.use_zarr:
                            if opt.stitch_mode == "tile-and-stitch":
                                out_slice = (
                                    255
                                    * prediction_map[
                                        0 : data["A_full_size_raw"][1],
                                        0 : data["A_full_size_raw"][2],
                                    ]
                                ).astype(np.uint8)
                            elif opt.stitch_mode == "valid-no-crop":
                                out_slice = (
                                    255
                                    * prediction_map[
                                        0 : data["A_full_size_pad"][1],
                                        0 : data["A_full_size_pad"][2],
                                    ]
                                ).astype(np.uint8)
                            elif opt.stitch_mode == "overlap-averaging":
                                out_slice = ((prediction_map / normalization_map) * 255).astype(np.uint8)
                            out_zarr[slice_idx - 1, :, :] = out_slice
                            del prediction_map
                            del normalization_map
                        else:
                            if opt.stitch_mode == "tile-and-stitch":
                                out_slice = (
                                    255
                                    * prediction_map[
                                        0 : data["A_full_size_raw"][1],
                                        0 : data["A_full_size_raw"][2],
                                    ]
                                ).astype(np.uint8)
                            elif opt.stitch_mode == "valid-no-crop":
                                out_slice = (
                                    255
                                    * prediction_map[
                                        0 : data["A_full_size_pad"][1],
                                        0 : data["A_full_size_pad"][2],
                                    ]
                                ).astype(np.uint8)
                            elif opt.stitch_mode == "overlap-averaging":
                                out_slice = ((prediction_map / normalization_map) * 255).astype(np.uint8)
                            prediction_volume.append(out_slice)
                            del prediction_map
                            del normalization_map
                    prediction_map = np.zeros((size_0, size_1), dtype=np.float32)
                    normalization_map = np.zeros(
                        (data["A_full_size_pad"][1], data["A_full_size_pad"][2]),
                        dtype=np.uint8,
                    )
                    if opt.stitch_mode == "tile-and-stitch":
                        prediction_slices = (
                            build_slices_zarr_2d(prediction_map, [0, 0, 0, 0], stride_arr, stride_arr)
                            if opt.use_zarr
                            else build_slices(
                                prediction_map,
                                [stride, stride],
                                [stride, stride],
                                use_shape_only=False,
                            )
                        )
                    elif opt.stitch_mode in ("valid-no-crop", "overlap-averaging"):
                        if opt.stitch_mode == "overlap-averaging":
                            normalization_map = np.zeros(
                                (
                                    data["A_full_size_pad"][1],
                                    data["A_full_size_pad"][2],
                                ),
                                dtype=np.uint8,
                            )
                        if opt.use_zarr:
                            if opt.stitch_mode == "overlap-averaging":
                                prediction_slices = build_slices_zarr_2d(prediction_map, [0, 0, 0, 0], patch_sz, stride_arr)
                            else:
                                prediction_slices = build_slices_zarr_2d(prediction_map, [0, 0, 0, 0], patch_sz, patch_sz)
                        else:
                            if opt.stitch_mode == "overlap-averaging":
                                prediction_slices = build_slices(
                                    prediction_map,
                                    [opt.patch_size, opt.patch_size],
                                    [stride, stride],
                                    use_shape_only=False,
                                )
                            else:
                                prediction_slices = build_slices(
                                    prediction_map,
                                    [opt.patch_size, opt.patch_size],
                                    [opt.patch_size, opt.patch_size],
                                    use_shape_only=False,
                                )
                    pred_index = 0
                    slice_idx += 1

                # --- Unpad if needed and convert to numpy ---
                if opt.stitch_mode == "overlap-averaging":
                    img = _unpad(img, patch_halo)
                img = torch.squeeze(torch.squeeze(img, 0), 0).cpu().float().numpy()

                # --- Update normalization map if overlap-averaging ---
                if opt.stitch_mode == "overlap-averaging":
                    if opt.use_zarr:
                        pred_h, pred_w = prediction_slices[pred_index][:2]
                        normalization_map[pred_h : pred_h + patch_size, pred_w : pred_w + patch_size] += 1
                    else:
                        normalization_map[prediction_slices[pred_index]] += 1

                # --- Add patch prediction to prediction map ---
                if opt.use_zarr:
                    pred_h, pred_w = prediction_slices[pred_index][:2]
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

                # --- Finalize slice if at end of slice ---
                if patch_index == len(input_list) - 1:
                    # Save last slice to Zarr or append to prediction_volume
                    if opt.use_zarr:
                        if opt.stitch_mode == "tile-and-stitch":
                            out_slice = (
                                255
                                * prediction_map[
                                    0 : data["A_full_size_raw"][1],
                                    0 : data["A_full_size_raw"][2],
                                ]
                            ).astype(np.uint8)
                        elif opt.stitch_mode == "overlap-averaging":
                            out_slice = (
                                prediction_map[
                                    0 : data["A_full_size_pad"][1],
                                    0 : data["A_full_size_pad"][2],
                                ]
                                / normalization_map[
                                    0 : data["A_full_size_pad"][1],
                                    0 : data["A_full_size_pad"][2],
                                ]
                                * 255
                            ).astype(np.uint8)
                        else:
                            out_slice = (
                                255
                                * prediction_map[
                                    0 : data["A_full_size_pad"][1],
                                    0 : data["A_full_size_pad"][2],
                                ]
                            ).astype(np.uint8)
                        out_zarr[slice_idx - 1, :, :] = out_slice
                        del prediction_map
                        del normalization_map
                    else:
                        if opt.stitch_mode == "tile-and-stitch":
                            out_slice = (
                                255
                                * prediction_map[
                                    0 : data["A_full_size_raw"][1],
                                    0 : data["A_full_size_raw"][2],
                                ]
                            ).astype(np.uint8)
                        elif opt.stitch_mode == "overlap-averaging":
                            out_slice = (
                                prediction_map[
                                    0 : data["A_full_size_pad"][1],
                                    0 : data["A_full_size_pad"][2],
                                ]
                                / normalization_map[
                                    0 : data["A_full_size_pad"][1],
                                    0 : data["A_full_size_pad"][2],
                                ]
                                * 255
                            ).astype(np.uint8)
                        else:
                            out_slice = (
                                255
                                * prediction_map[
                                    0 : data["A_full_size_pad"][1],
                                    0 : data["A_full_size_pad"][2],
                                ]
                            ).astype(np.uint8)
                        prediction_volume.append(out_slice)
                        del prediction_map
                        del normalization_map

        # --- Save output ---
        mode = "2d"
        stitch_str = opt.stitch_mode.replace("-", "_")
        if opt.use_zarr:
            zarr_str = "zarr"
            base = os.path.basename(data["A"][0][:-5])
            out_name = f"generated_{mode}_{zarr_str}_{stitch_str}_{base}"
            out_zarr_path = os.path.join(opt.results_dir, out_name + ".zarr")
            if not getattr(opt, "no_tiff", False):
                tifffile.imwrite(
                    os.path.join(opt.results_dir, out_name + ".tif"),
                    out_zarr[:],
                )
        else:
            tiff_str = "tiff"
            base = os.path.basename(data["A_paths"][0])
            out_name = f"generated_{mode}_{tiff_str}_{stitch_str}_{base}"
            tifffile.imwrite(
                os.path.join(opt.results_dir, out_name),
                np.asarray(prediction_volume),
            )


# Helper functions (unchanged)
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_tiff",
        action="store_true",
        help="Do not save output as TIFF when using Zarr",
    )
    # Parse known args with TestOptions
    opt = TestOptions().parse()
    args, _ = parser.parse_known_args()
    opt.no_tiff = args.no_tiff
    opt.num_threads = 12 if opt.use_zarr else 0
    opt.dataset_mode = "patched_2d_zarr" if opt.use_zarr else "patched_2d"
    opt.input_nc = opt.output_nc = 1
    opt.serial_batches = opt.no_flip = True
    inference(opt)
    TestOptions().save_options(opt)
