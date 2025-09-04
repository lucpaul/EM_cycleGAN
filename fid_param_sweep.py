import subprocess
import itertools
import os
import datetime

# Parameter grid
test_modes = ["2d", "2.5d", "3d"]
use_zarr_options = [True, False]
eval_directions = ["one-way", "both-ways"]

# Paths for zarr and non-zarr
zarr_paths = {
    "dataroot": "/mnt/d/Synatose_Datasets/All_new_results/Training_Datasets/MiniTestData/trainA_small_zarr",
    "target_domain": "/mnt/d/Synatose_Datasets/All_new_results/Training_Datasets/MiniTestData/trainB_small_zarr",
}
nonzarr_paths = {
    "dataroot": "/mnt/d/Synatose_Datasets/All_new_results/Training_Datasets/MiniTestData/trainA_small",
    "target_domain": "/mnt/d/Synatose_Datasets/All_new_results/Training_Datasets/MiniTestData/trainB_small",
}

model_name = "/mnt/d/Testing/my_cyclegan_model"
results_dir = "/mnt/d/Testing"
model_suffix = "_A"
patch_size = "190"
stitch_mode = "valid-no-crop"

# Log file setup
log_file = "fid_param_sweep.log"
with open(log_file, "w") as log:
    log.write(f"FID Parameter Sweep Log - {datetime.datetime.now()}\n\n")

# Sweep
for test_mode, use_zarr, eval_direction in itertools.product(
    test_modes, use_zarr_options, eval_directions
):
    if use_zarr:
        dataroot = zarr_paths["dataroot"]
        target_domain = zarr_paths["target_domain"]
        zarr_flag = "--use_zarr"
        # Check that zarr paths exist and are directories
        valid_zarr = os.path.isdir(dataroot) and os.path.isdir(target_domain)
        if not valid_zarr:
            with open(log_file, "a") as log:
                log.write(
                    f"\nSKIPPED: Zarr paths do not exist or are not directories: dataroot={dataroot}, target_domain={target_domain}\n"
                )
            print(
                f"SKIPPED: Zarr paths do not exist or are not directories: dataroot={dataroot}, target_domain={target_domain}"
            )
            continue
    else:
        dataroot = nonzarr_paths["dataroot"]
        target_domain = nonzarr_paths["target_domain"]
        zarr_flag = ""
        # Check that non-zarr paths exist and are directories containing at least one .tif or .tiff file
        valid_non_zarr = os.path.isdir(dataroot) and os.path.isdir(target_domain)

    def has_tif_files(path):
        try:
            return any(f.lower().endswith((".tif", ".tiff")) for f in os.listdir(path))
        except Exception:
            return False

    if valid_non_zarr:
        valid_non_zarr = has_tif_files(dataroot) and has_tif_files(target_domain)
    if not valid_non_zarr:
        with open(log_file, "a") as log:
            log.write(
                f"\nSKIPPED: Non-zarr paths do not exist or do not contain .tif/.tiff files: dataroot={dataroot}, target_domain={target_domain}\n"
            )
        print(
            f"SKIPPED: Non-zarr paths do not exist or do not contain .tif/.tiff files: dataroot={dataroot}, target_domain={target_domain}"
        )
        continue

    batch_size = 2 if test_mode == "3d" else 50
    if test_mode == "3d":
        model_name_run = model_name + "_3d"
    else:
        model_name_run = model_name
    cmd = [
        "python",
        "./fid_analysis.py",
        "--test_mode",
        test_mode,
        "--eval_direction",
        eval_direction,
        "--dataroot",
        dataroot,
        "--target_domain",
        target_domain,
        "--name",
        model_name_run,
        "--patch_size",
        patch_size,
        "--results_dir",
        results_dir,
        "--stitch_mode",
        stitch_mode,
        "--batch_size",
        str(batch_size),
    ]
    if eval_direction == "one-way":
        cmd.insert(cmd.index("--patch_size"), "--model_suffix")
        cmd.insert(cmd.index("--patch_size"), model_suffix)
    if zarr_flag:
        cmd.append(zarr_flag)

    param_str = f"test_mode={test_mode}, use_zarr={use_zarr}, eval_direction={eval_direction}, batch_size={batch_size}"
    print(f"Running: {param_str}")
    with open(log_file, "a") as log:
        log.write(f"\nRunning: {param_str}\nCommand: {' '.join(cmd)}\n")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            log.write(f"Return code: {result.returncode}\n")
            if result.returncode == 0:
                log.write("Status: SUCCESS\n")
            else:
                log.write(f"Status: ERROR\nError Output:\n{result.stderr}\n")
            log.write(f"Standard Output:\n{result.stdout}\n")
        except Exception as e:
            log.write(f"Status: EXCEPTION\nException: {str(e)}\n")
