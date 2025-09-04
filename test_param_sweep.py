import os
import subprocess

sample_scripts = [
    ("test_2d.py", "2d"),
    ("test_2_5d.py", "2_5d"),
    ("test_3d.py", "3d"),
]

stitch_modes = ["tile-and-stitch", "overlap-averaging", "valid-no-crop"]
use_zarr_opts = [True, False]

for script, mode in sample_scripts:
    for stitch_mode in stitch_modes:
        for use_zarr in use_zarr_opts:
            # Set dataroot based on zarr usage
            if use_zarr:
                dataroot = "/mnt/d/Synatose_Datasets/All_new_results/Training_Datasets/MiniTestData/trainA_very_small_zarr"
            else:
                dataroot = "/mnt/d/Synatose_Datasets/All_new_results/Training_Datasets/MiniTestData/trainA_very_small"

            # Set model name and batch size for 3d
            if mode == "3d":
                name = "/mnt/d/Testing/my_cyclegan_model_3d"
                batch_size = "2"
            else:
                name = "/mnt/d/Testing/my_cyclegan_model"
                batch_size = "50"

            cmd = [
                "python",
                script,
                "--dataroot",
                dataroot,
                "--name",
                name,
                "--epoch",
                "2",
                "--model_suffix",
                "_A",
                "--patch_size",
                "190",
                "--results_dir",
                "/mnt/d/Testing",
                "--batch_size",
                batch_size,
                "--stitch_mode",
                stitch_mode,
            ]
            if use_zarr:
                cmd.append("--use_zarr")

            print(f"\nRunning: {' '.join(cmd)}\n")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"Return code: {result.returncode}")
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")
            except subprocess.CalledProcessError as e:
                print(
                    f"\nERROR: Script '{script}' failed with stitch_mode='{stitch_mode}', use_zarr={use_zarr}, mode='{mode}'."
                )
                print(f"Command: {' '.join(cmd)}")
                print(f"Return code: {e.returncode}")
                print(f"STDOUT:\n{e.stdout}")
                print(f"STDERR:\n{e.stderr}")
