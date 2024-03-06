"""An inference script for image-to-image translation on 2d patches, with a standard stitching approach.

Once you have trained your 2d model with train.py, using any backbone other than unet, you can use this script to test the model.
It will load a saved model from '--name' and save the results to '--results_dir'.

It first creates a model and appropriate dataset (patched_2d_dataset) automatically. It will hard-code some parameters.
It then runs inference for the images in the dataroot, assembling full volumes by performing inference on each slice
of the image stack and then assembling the full stack from the slices.

Example (You need to train models first):
    Test a CycleGAN model:
        python test_2D_resnet.py --dataroot path/to/domain_A --name path/to/my_cyclegan_model --epoch 1 --model_suffix _A
                                 --patch_size 188 --test_mode 2d --results_dir path/to/results

See options/base_options.py and options/test_options.py for more test options."""

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
from util.util import adjust_patch_size

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def inference(opt):

    dicti = {}
    model_settings = open(os.path.join(opt.name, "train_opt.txt"), "r").read().splitlines()
    for x in range(1, len(model_settings) - 1):
        dicti[model_settings[x].split(':')[0].replace(' ', '')] = model_settings[x].split(':')[1].replace(' ', '')

    # Here, we make sure that the loaded model will use the same backbone as in training,
    # disregarding if anything else is set in base_options.py

    opt.netG = dicti['netG']
    if opt.netG.startswith('unet') or opt.netG.startswith('resnet'):
        if opt.netG.startswith('resnet'):
            opt.netG = opt.netG[:14]
        opt.ngf = int(dicti['ngf'])
    assert dicti['train_mode'] == '2d', "For 2D predictions, the model needs to be a 2D model. This model was not trained on 2D patches."

    adjust_patch_size(opt)
    patch_size = opt.patch_size
    stride = opt.patch_size - 16

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
        normalization_volume = []
        data_is_array = False

        if type(data['A']) == np.ndarray:
            data_is_array = True
            input_list = data['A'][0]
        elif type(data['A']) == list:
            input_list = data['A']
        for i in tqdm(range(0, len(input_list)), desc ="Inference progress"):
            input = input_list[i]
            input = transform(input)
            if data_is_array:
                input = torch.unsqueeze(input, 0)
            input = _pad(input, patch_halo)

            model.set_input(input)
            model.test()
            img = model.fake
            if i % int(data['patches_per_slice']) == 0:
                if i != 0:
                    prediction_volume.append(prediction_map)
                    normalization_volume.append(normalization_map)

                prediction_map = np.zeros((data['A_full_size_raw'][1], data['A_full_size_raw'][2]), dtype=np.float32)
                normalization_map = np.zeros((data['A_full_size_raw'][1], data['A_full_size_raw'][2]), dtype=np.uint8)
                prediction_slices = build_slices(prediction_map, [patch_size, patch_size], [stride, stride])
                pred_index = 0

            img = _unpad(img, patch_halo)
            img = torch.squeeze(torch.squeeze(img, 0), 0)
            img = tensor2im(img)
            normalization_map[prediction_slices[pred_index]] += 1

            prediction_map[prediction_slices[pred_index]] += img  # torch.squeeze(torch.squeeze(img, 0), 0)
            pred_index += 1

            if i == len(input_list)-1:
                prediction_volume.append(prediction_map)
                normalization_volume.append(normalization_map)

        normalization_volume = np.asarray(normalization_volume)
        prediction_volume = np.asarray(prediction_volume)
        prediction_volume = prediction_volume / normalization_volume

        prediction_volume = (prediction_volume * 255).astype(np.uint8)
        tifffile.imwrite(opt.results_dir + "/generated_" + os.path.basename(data['A_paths'][0]), prediction_volume)

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
    opt.input_nc = 1
    opt.output_nc = 1
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.dataset_mode = 'patched_2d'
    opt.test_mode == '2d'
    inference(opt)
    TestOptions().save_options(opt)