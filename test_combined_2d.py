"""An inference script for image-to-image translation with 2D patches, using tile-and-stitch.

Once you have trained your 2d model with train.py, using a unet backbone, you can use this script to test the model.
It will load a saved model from '--name' and save the results to '--results_dir'.

It first creates a model and appropriate dataset (patched_2d_dataset) automatically. It will hard-code some parameters.
It then runs inference for the images in the dataroot, assembling full volumes by performing inference on volume patches
from which the prediction is then assembled.

Example (You need to train models first):
    Test a CycleGAN model:
        python test_2D.py --dataroot path/to/domain_A --name path/to/my_cyclegan_model --epoch 1 --model_suffix _A
                          --patch_size 190 --test_mode 2d --results_dir path/to/results

See options/base_options.py and options/test_options.py for more test options."""
import math
import os

import numpy as np
import tifffile
import torch
from torchvision import transforms
from tqdm import tqdm

from data import create_dataset
from data.SliceBuilder import build_slices
from models import create_model
from options.test_options import TestOptions
from util.util import adjust_patch_size
from util.util import tensor2im

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
        opt.ngf = int(dicti['ngf'])
    assert dicti[
               'train_mode'] == '2d', "For 2D predictions, the model needs to be a 2D model. This model was not trained on 2D patches."

    if opt.stitch_mode == "tile-and-stitch":
        adjust_patch_size(opt)
        patch_size = opt.patch_size
        difference = 0
        for i in range(2, int(math.log(int(opt.netG[5:]), 2) + 2)):
            difference += 2 ** i
        stride = patch_size - difference - 2
        init_padding = int((patch_size - stride) / 2)


    elif opt.stitch_mode == "overlap-averaging":
        stride = opt.patch_size - 16
        patch_halo = (8, 8)
    # patch_size = opt.patch_size

    # Need to still test this again

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)

    for module in model.netG.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
            if opt.stitch_mode == "tile-and-stitch" or opt.stitch_mode == "valid-no-crop":
                # parameters = model.netG.module.parameters()
                module.padding = (0, 0)

                # layer = model.netG.module
                # print(layer)
                # for new_module in layer.modules():
                #     print(new_module)
                #     print(layer.new_module)
                #     if isinstance(new_module, torch.nn.Conv2d):
                #         print(new_module)
                #         model.netG.module.new_module.padding = (0,0)

                # features = model.netG.module.parameters()
            else:
                model.netG.module.model.padding = 1

    # regular setup: load and print networks; create schedulers
    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name,
                               config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    model.eval()
    transform = transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float32)
    ])
    for j, data in enumerate(dataset):
        prediction_volume = []
        data_is_array = False

        if opt.stitch_mode == "overlap-averaging":
            normalization_volume = []
        if type(data['A']) == np.ndarray:
            data_is_array = True
            input_list = data['A'][0]
        elif type(data['A']) == list:
            input_list = data['A']
        print("input_patches: ", len(input_list))

        prediction_map = None
        prediction_slices = None
        pred_index = 0

        for i in tqdm(range(0, math.ceil(len(input_list) / opt.batch_size)), desc="Inference progress"):
            input = input_list[i * opt.batch_size:i * opt.batch_size + opt.batch_size]
            input = torch.cat(input, dim=0)
            input = transform(input)
            if data_is_array:
                input = torch.unsqueeze(input, 0)

            if opt.stitch_mode == "overlap-averaging":
                input = _pad(input, patch_halo)
            # print(input.shape)
            model.set_input(input)
            model.test()
            img_batched = model.fake
            append_index = 0
            for b in range(0, opt.batch_size):
                if b + i * opt.batch_size >= len(input_list):
                    break

                patch_index = i * opt.batch_size + b

                img = img_batched[b]

                img = torch.unsqueeze(img, 0)

                # print(data['A_full_size_raw'], data['A_full_size_pad'])
                if opt.stitch_mode == "tile-and-stitch":
                    img = img[:, :, init_padding:-init_padding, init_padding:-init_padding]
                    size_0 = stride * math.ceil(((data['A_full_size_pad'][1] - patch_size) / stride) + 1)
                    size_1 = stride * math.ceil(((data['A_full_size_pad'][2] - patch_size) / stride) + 1)

                elif opt.stitch_mode == "valid-no-crop" or opt.stitch_mode == "overlap-averaging":
                    size_0 = data['A_full_size_pad'][1]
                    size_1 = data['A_full_size_pad'][2]

                if patch_index % int(data['patches_per_slice']) == 0:
                    if i != 0:
                        if opt.stitch_mode == "tile-and-stitch":
                            prediction_map = prediction_map[0:data['A_full_size_raw'][1], 0:data['A_full_size_raw'][2]]
                            # print("Data Raw Map: ", data['A_full_size_raw'][1], data['A_full_size_raw'][2])
                            # prediction_volume.append(prediction_map)

                        elif opt.stitch_mode == "valid-no-crop":
                            prediction_map = prediction_map[0:data['A_full_size_pad'][1],
                                             0:data['A_full_size_pad'][2]]  # is this even needed?

                        elif opt.stitch_mode == "overlap-averaging":
                            normalization_volume.append(normalization_map)

                        prediction_volume.append(prediction_map)
                        append_index += 1
                    # prediction_map = np.zeros((data['A_full_size_pad'][1], data['A_full_size_pad'][2]), dtype=np.float32)
                    prediction_map = np.zeros((size_0, size_1), dtype=np.uint8)
                    normalization_map = np.zeros((data['A_full_size_pad'][1], data['A_full_size_pad'][2]),
                                                 dtype=np.uint8)

                    if opt.stitch_mode == "tile-and-stitch":
                        prediction_slices = build_slices(prediction_map, [stride, stride], [stride, stride])

                    elif opt.stitch_mode == "valid-no-crop":
                        prediction_slices = build_slices(prediction_map, [opt.patch_size, opt.patch_size],
                                                         [opt.patch_size, opt.patch_size])

                    elif opt.stitch_mode == "overlap-averaging":
                        normalization_map = np.zeros((data['A_full_size_pad'][1], data['A_full_size_pad'][2]),
                                                     dtype=np.uint8)
                        prediction_slices = build_slices(prediction_map, [opt.patch_size, opt.patch_size],
                                                         [stride, stride])

                    pred_index = 0

                if opt.stitch_mode == "overlap-averaging":
                    img = _unpad(img, patch_halo)

                img = torch.squeeze(torch.squeeze(img, 0), 0)
                img = (tensor2im(img) * 255).astype(np.uint8)

                if opt.stitch_mode == "overlap-averaging":
                    normalization_map[prediction_slices[pred_index]] += 1

                prediction_map[prediction_slices[pred_index]] += img

                pred_index += 1

                if patch_index == len(input_list) - 1:
                    prediction_map = prediction_map[0:data['A_full_size_raw'][1], 0:data['A_full_size_raw'][2]]
                    prediction_volume.append(prediction_map)

                    if opt.stitch_mode == "overlap-averaging":
                        normalization_volume.append(normalization_map)

        if opt.stitch_mode == "overlap-averaging":
            normalization_volume = np.asarray(normalization_volume)
            prediction_volume = np.asarray(prediction_volume)
            prediction_volume = prediction_volume / normalization_volume

        prediction_volume = np.asarray(prediction_volume)
        # print(prediction_volume.shape)

        tifffile.imwrite(opt.results_dir + "/generated_" + os.path.basename(data['A_paths'][0]), prediction_volume)


# pad and unpad functions from pytorch 3d unet by wolny

def _pad(m, patch_halo):
    if patch_halo is not None:
        y, x = patch_halo
        return torch.nn.functional.pad(m, (y, y, x, x), mode='reflect')
    return m


def _unpad(m, patch_halo):
    if patch_halo is not None:
        y, x = patch_halo
        return m[..., y:-y, x:-x]
    return m


if __name__ == '__main__':
    # read out sys.argv arguments and parse
    opt = TestOptions().parse()
    opt.num_threads = 0  # test code only supports num_threads = 0
    # opt.batch_size = 1  # test code only supports batch_size = 1
    opt.input_nc = 1
    opt.output_nc = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.dataset_mode = 'patched_2d'
    inference(opt)
    TestOptions().save_options(opt)
