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
from data.SliceBuilder import build_slices_3d
import numpy as np
import tifffile
import math
import torch

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def inference(opt):
    patch_size = opt.patch_size
    stride = opt.stride_A
    dicti = {}
    model_settings = open(os.path.join(opt.name, "train_opt.txt"), "r").read().splitlines()
    for x in range(1, len(model_settings) - 1):
        dicti[model_settings[x].split(':')[0].replace(' ', '')] = model_settings[x].split(':')[1].replace(' ', '')

    # Here, we make sure that the loaded model will use the same backbone as in training,
    # disregarding if anything else is set in base_options.py

    opt.netG = dicti['netG']
    opt.ngf = dicti['ngf']
    assert dicti['train_mode'] == '3d'
    opt.test_mode == '3d'
    opt.dataset_mode = 'patched_3d'

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    init_padding = int((patch_size - stride) / 2)

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
        prediction_map = None
        data_is_array = False
        if type(data['A']) == np.ndarray:
            data_is_array = True
            input_list = data['A'][0]
        elif type(data['A']) == list:
            input_list = data['A']
        for i in range(0, len(input_list)):
            input = input_list[i]
            input = transform(input)

            if data_is_array:
                input = torch.unsqueeze(input, 0)

            model.set_input(input)
            model.test()
            img = model.fake

            img = img[:, :, init_padding:-init_padding, init_padding:-init_padding, init_padding:-init_padding]

            if prediction_map is None:
                size_0 = stride * math.ceil(((data['A_full_size_pad'][0] - patch_size) / stride) + 1)  # for 190 use 64 instead of 38
                size_1 = stride * math.ceil(((data['A_full_size_pad'][1] - patch_size) / stride) + 1)
                size_2 = stride * math.ceil(((data['A_full_size_pad'][2] - patch_size) / stride) + 1)
                prediction_map = np.zeros((size_0, size_1, size_2), dtype=np.uint8)

                # prediction_map = torch.zeros(size_0, size_1, size_2, dtype=torch.float32, device="cuda")

                prediction_slices = build_slices_3d(prediction_map, [stride, stride, stride], [stride, stride, stride])

            img = torch.squeeze(torch.squeeze(img, 0), 0)
            img = (tensor2im(img) * 255).astype(np.uint8)

            prediction_map[prediction_slices[i]] += img  # torch.squeeze(torch.squeeze(img, 0), 0)

        prediction_map = prediction_map[0:data['A_full_size_raw'][0], 0:data['A_full_size_raw'][1],0:data['A_full_size_raw'][2]]

        tifffile.imwrite(opt.results_dir + "/generated_" + os.path.basename(data['A_paths'][0]), prediction_map)


if __name__ == '__main__':
    # read out sys.argv arguments and parse
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.dataset_mode = 'patched_3d'
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    inference(opt)
