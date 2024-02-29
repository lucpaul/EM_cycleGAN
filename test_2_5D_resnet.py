"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os

import torchvision.transforms as transforms
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.util import tensor2im
from data.SliceBuilder import build_slices
import numpy as np
import tifffile
import torch
from torch import nn
from tqdm import tqdm

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def inference_2_5D_resnet(opt):
    patch_size = opt.patch_size
    stride = opt.patch_size

    dicti = {}
    model_settings = open(os.path.join(opt.name, "train_opt.txt"), "r").read().splitlines()
    for x in range(1, len(model_settings) - 1):
        dicti[model_settings[x].split(':')[0].replace(' ', '')] = model_settings[x].split(':')[1].replace(' ', '')

    # Here, we make sure that the loaded model will use the same backbone as in training,
    # disregarding if anything else is set in base_options.py

    opt.netG = dicti['netG']
    if opt.netG.startswith('unet') or opt.netG.startswith('resnet'):
        opt.ngf = int(dicti['ngf'])
    assert dicti['train_mode'] == '2d', "For 2D predictions, the model needs to be a 2D model. This model was not trained on 2D patches."
    opt.test_mode == '2_5d'

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    patch_halo = (16, 16) # Makes an enormous difference in the quality of predictions

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name,
                               config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    model.eval()
    transform = transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float32)
    ])
    for j, data in enumerate(dataset):
        prediction_volume = []
        ortho_planes = ['xy', 'zy', 'zx']
        for direction in ortho_planes:
            dir_volume = []
            normalization_volume = []
            if direction == 'xy':
                full_raw_dim_1 = data['A_full_size_raw'][1]
                full_raw_dim_2 = data['A_full_size_raw'][2]
            if direction == 'zy':
                full_raw_dim_1 = data['A_full_size_raw'][0]
                full_raw_dim_2 = data['A_full_size_raw'][1]
            if direction == 'zx':
                full_raw_dim_1 = data['A_full_size_raw'][2]
                full_raw_dim_2 = data['A_full_size_raw'][0]
            print('predictions for orthopatches: ', direction)

            data_is_array = False

            if type(data[direction]) == np.ndarray:
                data_is_array = True
                input_list = data[direction][0]
            elif type(data[direction]) == list:
                input_list = data[direction]

            for i in tqdm(range(0, len(data[direction])), desc=f"Inference Progress for orthoslices {direction}"):
                input = data[direction][i]
                input = transform(input)
                if data_is_array:
                    input = torch.unsqueeze(input, 0)
                input = _pad(input, patch_halo)
                model.set_input(input)
                model.test()
                img = model.fake
                if i % int(data['patches_per_slice'+"_"+direction]) == 0:
                    if i != 0:
                        if direction == 'xy':
                            prediction_map = prediction_map[0:data['A_full_size_raw'][1], 0:data['A_full_size_raw'][2]]
                        if direction == 'zy':
                            prediction_map = prediction_map[0:data['A_full_size_raw'][0], 0:data['A_full_size_raw'][1]]
                        if direction == 'zx':
                            prediction_map = prediction_map[0:data['A_full_size_raw'][2], 0:data['A_full_size_raw'][0]]

                        dir_volume.append(prediction_map)
                        normalization_volume.append(normalization_map)

                    prediction_map = np.zeros((full_raw_dim_1, full_raw_dim_2), dtype=np.float32)
                    normalization_map = np.zeros((full_raw_dim_1, full_raw_dim_2), dtype=np.uint8)
                    prediction_slices = build_slices(prediction_map, [patch_size, patch_size], [stride, stride])
                    pred_index = 0

                img = _unpad(img, patch_halo)
                img = torch.squeeze(torch.squeeze(img, 0), 0)
                img = tensor2im(img)

                normalization_map[prediction_slices[pred_index]] += 1
                prediction_map[prediction_slices[pred_index]] += img  # torch.squeeze(torch.squeeze(img, 0), 0)
                pred_index += 1

                if i == len(input_list)-1:
                    dir_volume.append(prediction_map)
                    normalization_volume.append(normalization_map)

            normalization_volume = np.asarray(normalization_volume)
            dir_volume = np.asarray(dir_volume)
            dir_volume = dir_volume / normalization_volume

            prediction_volume.append(dir_volume)

        prediction_volume[0] = np.asarray(prediction_volume[0])
        prediction_volume[0] = (prediction_volume[0] - prediction_volume[0].min()) / (prediction_volume[0].max() - prediction_volume[0].min())

        prediction_volume[1] = np.transpose(np.asarray(prediction_volume[1]), (1, 2, 0))
        prediction_volume[1] = (prediction_volume[1] - prediction_volume[1].min()) / (prediction_volume[1].max() - prediction_volume[1].min())

        prediction_volume[2] = np.transpose(np.asarray(prediction_volume[2]), (2, 0, 1))
        prediction_volume[2] = (prediction_volume[2] - prediction_volume[2].min()) / (prediction_volume[2].max() - prediction_volume[2].min())

        prediction_volume_full = ((prediction_volume[0] + prediction_volume[1] + prediction_volume[2]) / 3)
        prediction_volume_full = (255 * prediction_volume_full).astype(np.uint8)

        print("writing prediction of shape: ", prediction_volume_full.shape)

        tifffile.imwrite(opt.results_dir + "/generated_" + os.path.basename(data['A_paths'][0]), prediction_volume_full)

# pad and unpad functions from pytorch 3d unet by wolny

def _pad(m, patch_halo):
    if patch_halo is not None:
        y, x = patch_halo
        return nn.functional.pad(m, (y, y, x, x), mode='reflect')
    return m


def _unpad(m, patch_halo):
    if patch_halo is not None:
        y, x = patch_halo
        return m[..., y:-y, x:-x]
    return m

if __name__ == '__main__':
    # read out sys.argv arguments and parse
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.input_nc = 1
    opt.output_nc = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    # opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.dataset_mode = 'patched_2_5d'
    inference_2_5D_resnet(opt)
