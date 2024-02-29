import os
import torch
from options.test_options import TestOptions
from data.SliceBuilder import build_slices
import random
import tifffile
from util.fid_score import calculate_fid_given_paths, save_fid_stats

# load model and get datasets


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
    elif dicti['train_mode']=='2d' and not dicti['netG'].startswith('unet'):
        from test_2D_resnet import inference
        from test_2_5D_resnet import inference_2_5D_resnet
        opt.dataset_mode = 'patched_2d'
    else:
        if dicti['netG'].startswith('unet'):
            from test_2D import inference
        else:
            from test_2D_resnet import inference

    # if dicti['netG'].startswith('unet'):
    #     from test_2D import inference
    # elif dicti['netG'].startswith('resnet'):
    #     from test_2D_resnet import inference

    opt.dataset_mode = 'patched_2d'

    #opt.patch_size = int(dicti['patch_size'][:3]) #If patch size has 3 digits, otherwise [:2], need to fix this.
    all_fid_scores = {}
    domain_A = opt.dataroot

    assert (opt.eval_direction == "one-way" or opt.eval_direction=="both-ways"), f"flag --eval_direction accepts as arguments either 'one-way' or 'both-ways', not {opt.eval_direction}"

    for checkpoint in os.listdir(opt.name):

        '''Inference Part'''
        if not (checkpoint.endswith("G_A.pth") or checkpoint.endswith("G_B.pth")):
            continue
        else:
            if opt.eval_direction == 'both-ways':
                if opt.model_suffix not in checkpoint:
                    if opt.model_suffix == "_A":
                        opt.model_suffix = "_B"
                    elif opt.model_suffix == "_B":
                        opt.model_suffix = "_A"
                if opt.model_suffix == "_A":
                    opt.dataroot = domain_A
                elif opt.model_suffix == "_B":
                    opt.dataroot = opt.domain_B

            elif opt.eval_direction == 'one-way':
                if opt.model_suffix not in checkpoint:
                    continue

            if not os.path.exists(os.path.join(opt.results_dir, "temp_results"+opt.model_suffix)):
                os.mkdir(os.path.join(opt.results_dir, "temp_results"+opt.model_suffix))

            print("using model: ", opt.model_suffix, opt.dataroot)
            if not opt.results_dir.endswith(opt.model_suffix[-1]):
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
        indices = random.sample(range(len(patches)), k=20000)

        patches = torch.stack(patches)
        patches = patches[indices, :, :, :]

        if opt.name.endswith('/'):
            opt.name = opt.name[:-1]

        if opt.dataroot.endswith('/'):
            opt.dataroot = opt.dataroot[:-1]

        fingerprint_path = os.path.join(opt.results_dir, "FID_features_for_"+os.path.basename(opt.name)+opt.model_suffix+"_epoch_" + checkpoint[:-12] + "_prediction_on_" + os.path.basename(opt.dataroot))

        paths = [list(patches), fingerprint_path]

            # Here the paths argument is a list containing a list and a path
        save_fid_stats(paths, batch_size=50, device='cuda', dims=2048)

        # Here the path arguments are both paths to .npz files
        if opt.model_suffix.endswith("A"):
            fid_score = calculate_fid_given_paths([fingerprint_path+".npz", opt.target_domain_B_fid_file],  batch_size=50, device='cuda', dims=2048)
            all_fid_scores[os.path.basename(opt.name) +"_"+ str(checkpoint[:-12]+opt.model_suffix) + ": " + os.path.basename(opt.target_domain_B_fid_file) + " vs. " + os.path.basename(opt.dataroot)] = fid_score
        else:
            assert opt.target_domain_A_fid_file != None, "If you chose to evaluate fid both ways, you must provide a target fid file for both domains A and B (in .npz) format."
            fid_score = calculate_fid_given_paths([fingerprint_path + ".npz", opt.target_domain_A_fid_file], batch_size=50, device='cuda', dims=2048)
            all_fid_scores[os.path.basename(opt.name) + "_" + str(checkpoint[:-12]+opt.model_suffix) + ": " + os.path.basename(opt.target_domain_A_fid_file) + " vs. " + os.path.basename(opt.dataroot)] = fid_score

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
    opt.results_dir = '/media/lucas2/Untitled1/Synatose_Datasets/All_new_results/Generated_Datasets/New_Presentation_Files/'#'/fast/AG_Kainmueller/data/domain_adaptation/Models/2D_models/HeLa_NMR/fid_results'
    opt.domain_B = '/media/lucas2/Untitled1/Synatose_Datasets/All_new_results/Training_Datasets/HeLa-NMR/trainA/'#'/fast/AG_Kainmueller/data/domain_adaptation/HeLa_NMR/trainB'
    opt.target_domain_B_fid_file = '/media/lucas2/Untitled1/Synatose_Datasets/All_new_results/FID_related/FID_Fingerprints_raw_resampled_datasets/NMR_FID_fingerprint.npz'#'/fast/AG_Kainmueller/data/domain_adaptation/FID_Fingerprints/raw_data/NMR_hypoxic_FID_fingerprint.npz'
    opt.target_domain_A_fid_file = '/media/lucas2/Untitled1/Synatose_Datasets/All_new_results/FID_related/FID_Fingerprints_raw_resampled_datasets/HeLa_FID_fingerprint.npz'#'/fast/AG_Kainmueller/data/domain_adaptation/FID_Fingerprints/raw_data/HeLa_FID_fingerprint.npz'
    opt.eval_direction = 'one-way'

    fid_analysis(opt)