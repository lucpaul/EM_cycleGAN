"""A script that will calculate the Frechet Inception Distance (FID) for the checkpoints of the trained cycleGAN mode.

This script implements the pytorch-fid code from https://github.com/mseitzer/pytorch-fid, with minor modifications (see util/fid_score.py)

This script will iterate through the checkpoints of a model path provided by the user with the '--name' flag.

Evaluation flags

'--eval_direction'

When running the script, the user has the choice of evaluating the model 'one-way' or 'both-ways' with the --eval_direction' flag.
'one-way' will only evaluate the model in one direction, say AtoB or BtoA. Which direction will be determined by the
use of the '--model_suffix' flag. Using '_A' as '--model_suffix' will evaluate the checkpoints for the models trained in direction AtoB.
'both-ways' will conversely evaluate all models, that is AtoB and BtoA. Importantly, the user SHOULD NOT use the 'model_suffix'
flag in this case.

Setting the target domain(s)

'--dataroot': Generally, the '--dataroot' set by the user will initially always act as the default source domain, i.e. the domain that
              is to be translated into the target domain. This can be dynamically changed if the model is evaluated 'both-ways'

'--target_domain_B_fid_file' and '--target_domain_A_fid_file': To set the target domain against which to evaluate the models' predictions,
                                                               the user can set '--target_domain_B_fid_file' and  which should provide the
                                                               paths to previously calculated FID feature .npz files. If models are evaluated
                                                               'one-way', the user should provide '--target_domain_B_fid_file' when
                                                               evaluating the AtoB models, or vice versa.

'--target_domain': If no files for the target domains have been calculated previously, it will be enough to provide '--target_domain' flag and
                   '--dataroot' which will be used to automatically calculate these FID .npz files. '--dataroot' and '--target_domain' should
                   contain image files. Don't use this flag if the above flags are used, as this will conflict with the above.
                   The code was tested for data where '--dataroot' and '--target_domain' are subfolders of the same parent.

Results

During script execution, temporary predictions of for the checkpoints are stored in two additional folders named
temp_resultsA and temp_resultsB which are saved in the '--results_dir'. '--results_dir' will also hold the FID feature files in .npz format.

Finally, the scores for each checkpoint are stored in a csv file in the model folder ('--name')

Examples

    After training a 2D model on a dataset from domains trainA and trainB, evaluate FID for a test dataset from domain A and compare the predictions
    to images from domain B (note that the correct --model_suffix is provided):

        python ./fid_analysis.py --dataroot path/to/domainA --name path/to/model --results_dir path/to/results --test_mode 2d
                                 --target_domain path/to/domainB --eval_direction one-way --model_suffix _A --patch_size 190

    After training a 3D model on a dataset from domains trainA and trainB, evaluate the FID both ways for test datasets from domain A and B and compare
    the predictions to the target fid files from the respective target domains (note that --model_suffix is OMITTED!):

        python ./fid_analysis.py --dataroot path/to/domainA --name path/to/model --results_dir path/to/results --test_mode 3d
                                 --target_domain path/to/domainB --eval_direction both-ways --patch_size 190
                                 --target_domain_A_fid_file path/to/FID_features_from_domainA.npz
                                 --target_domain_B_fid_file path/to/FID_features_from_domainB.npz

See options/base_options.py and options/test_options.py for the key options and models/test_model.py for additional fid specific options."""

import os
import torch
from options.test_options import TestOptions
from data.SliceBuilder import build_slices
import random
import tifffile
from util.fid_score import calculate_fid_given_paths, save_fid_stats
from calculate_FID_features import fid_features
import shutil


def fid_analysis(opt):

    dicti = {}
    model_settings = open(os.path.join(opt.name, "train_opt.txt"), "r").read().splitlines()
    for x in range(1, len(model_settings) - 1):
        dicti[model_settings[x].split(':')[0].replace(' ', '')] = model_settings[x].split(':')[1].replace(' ', '')

    if dicti['train_mode'] == '3d' and dicti['netG'].startswith('unet'):
        from test_3D import inference
        opt.dataset_mode = 'patched_3d'
    elif dicti['train_mode'] == '3d' and not dicti['netG'].startswith('unet'):
        from test_3D_resnet import inference
        opt.dataset_mode = 'patched_3d'
    elif dicti['train_mode'] == '2d' and dicti['netG'].startswith('unet'):
        from test_2D import inference
        from test_2_5D import inference_2_5D
        opt.dataset_mode = 'patched_2d'
    elif dicti['train_mode'] == '2d' and not dicti['netG'].startswith('unet'):
        from test_2D_resnet import inference
        from test_2_5D_resnet import inference_2_5D_resnet
        opt.dataset_mode = 'patched_2d'
    else:
        if dicti['netG'].startswith('unet'):
            from test_2D import inference
        else:
            from test_2D_resnet import inference

    all_fid_scores = {}
    # Hard-code the initial source and target domains
    source_domain = opt.dataroot
    target_domain = opt.target_domain

    for checkpoint in os.listdir(opt.name):

        '''Inference Part'''
        if not (checkpoint.endswith("G_A.pth") or checkpoint.endswith("G_B.pth")):
            continue
        else:
            if opt.eval_direction == 'both-ways':
                if checkpoint.endswith("G_A.pth"):
                    if opt.model_suffix == "_B":
                        opt.model_suffix = "_A"
                    opt.source_domain = source_domain
                    opt.target_domain = target_domain
                elif checkpoint.endswith("G_B.pth"):
                    if opt.model_suffix == "_A":
                        opt.model_suffix = "_B"
                    opt.source_domain = target_domain
                    opt.target_domain = source_domain

                opt.dataroot = opt.source_domain

            elif opt.eval_direction == 'one-way':
                if opt.model_suffix not in checkpoint:
                    continue

            if os.path.exists(os.path.join(opt.results_dir, "temp_results"+opt.model_suffix)):
                shutil.rmtree(os.path.join(opt.results_dir, "temp_results"+opt.model_suffix))

            os.mkdir(os.path.join(opt.results_dir, "temp_results"+opt.model_suffix))

            opt.results_dir = os.path.join(opt.results_dir, "temp_results"+opt.model_suffix)

            if checkpoint.startswith("l"):
                opt.epoch = "latest"
            elif checkpoint[1] == "_":
                opt.epoch = int(checkpoint[0])
            elif checkpoint[2] == "_":
                opt.epoch = int(checkpoint[:2])
            elif checkpoint[3] == "_":
                opt.epoch = int(checkpoint[:3])

            if opt.test_mode == "2.5d" and dicti['netG'].startswith('unet'):
                opt.dataset_mode = 'patched_2_5d'
                inference_2_5D(opt)
            elif opt.test_mode == "2.5d" and not dicti['netG'].startswith('unet'):
                opt.dataset_mode = 'patched_2_5d'
                inference_2_5D_resnet(opt)
            else:
                inference(opt)

            # Reset results dir to base folder
            opt.results_dir = os.path.split(opt.results_dir)[0]

        '''FID Evaluation on the predictions'''

        patches = []
        for img in os.listdir(os.path.join(opt.results_dir, "temp_results"+opt.model_suffix)):
            dataset = tifffile.imread(os.path.join(opt.results_dir, "temp_results"+opt.model_suffix, img))
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
        indices = random.sample(range(len(patches)), k=2000)

        patches = torch.stack(patches)
        patches = patches[indices, :, :, :]

        if opt.name.endswith('/'):
            opt.name = opt.name[:-1]

        if opt.source_domain.endswith('/'):
            opt.source_domain = opt.source_domain[:-1]


        fingerprint_path = os.path.join(opt.results_dir, "FID_features_for_"+os.path.basename(opt.name) + "_epoch_" + checkpoint[:-12] + "_generated_" + os.path.basename(opt.dataroot) + "_as_" + os.path.basename(opt.target_domain))

        paths = [list(patches), fingerprint_path]

            # Here the paths argument is a list containing a list and a path
        save_fid_stats(paths, batch_size=50, device='cuda', dims=2048)

        # If no ground-truth files exist yet, they can be calculated here
        if opt.target_domain_B_fid_file == '':
            fid_features(target_domain, opt.results_dir, n_samples=20)
            opt.target_domain_B_fid_file = os.path.join(opt.results_dir, "FID_features_for_"+os.path.basename(target_domain)+".npz")

        if opt.target_domain_A_fid_file == '':
            fid_features(source_domain, opt.results_dir, n_samples=20)
            opt.target_domain_A_fid_file = os.path.join(opt.results_dir, "FID_features_for_" + os.path.basename(source_domain) + ".npz")

        # Here the path arguments are both paths to .npz files
        if opt.model_suffix.endswith("A"):
            fid_score = calculate_fid_given_paths([fingerprint_path+".npz", opt.target_domain_B_fid_file],  batch_size=50, device='cuda', dims=2048)
        else:
            fid_score = calculate_fid_given_paths([fingerprint_path+".npz", opt.target_domain_A_fid_file], batch_size=50, device='cuda', dims=2048)
        all_fid_scores["FID_" + os.path.basename(opt.name) + "_" + str(checkpoint[:-12]) + ": generated " + os.path.basename(opt.dataroot) + " as " + os.path.basename(opt.target_domain) + " vs. " + os.path.basename(opt.target_domain)] = fid_score

    with open(opt.name+"/fid_scores.csv" , "w") as fid_csv_scores:
        for key in all_fid_scores.keys():
            fid_csv_scores.write("%s,%s\n" % (key, all_fid_scores[key]))

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.preprocess = 'none'
    opt.input_nc = 1
    opt.output_nc = 1

    fid_analysis(opt)