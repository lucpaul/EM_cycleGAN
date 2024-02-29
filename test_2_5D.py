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
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.util import tensor2im
from data.SliceBuilder import build_slices
import numpy as np
import tifffile
import math
import torch
from tqdm import tqdm

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

def inference_2_5D(opt):
    patch_size = opt.patch_size
    dicti = {}
    model_settings = open(os.path.join(opt.name, "train_opt.txt"), "r").read().splitlines()
    for x in range(1, len(model_settings) - 1):
        dicti[model_settings[x].split(':')[0].replace(' ', '')] = model_settings[x].split(':')[1].replace(' ', '')

    # Here, we make sure that the loaded model will use the same backbone as in training,
    # disregarding if anything else is set in base_options.py

    opt.netG = dicti['netG']
    opt.ngf = int(dicti['ngf'])
    assert dicti['train_mode'] == '2d', "For 2.5D (orthoslice) predictions, the model needs to be a 2D model. This model was not trained on 2D patches."
    opt.test_mode == '2_5d'
    #opt.dataset_mode = 'patched_2_5d'

    # A quick calculation to ensure tile-and-stitch compatible stride for the given patch size and backbone
    difference = 0
    for i in range(2, int(math.log(int(opt.netG[5:]), 2) + 2)):
        difference += 2 ** i
    stride = patch_size - difference - 2

    init_padding = int((patch_size - stride) / 2)
    dataset = create_dataset(opt)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    model.eval()
    for j, data in enumerate(dataset):
        prediction_volume = []
        ortho_planes = ['xy', 'zy', 'zx']
        for direction in ortho_planes:
            dir_volume = []
            if direction == 'xy':
                full_pad_dim_1 = data['A_full_size_pad'][1]
                full_pad_dim_2 = data['A_full_size_pad'][2]
            if direction == 'zy':
                full_pad_dim_1 = data['A_full_size_pad'][0]
                full_pad_dim_2 = data['A_full_size_pad'][1]
            if direction == 'zx':
                full_pad_dim_1 = data['A_full_size_pad'][2]
                full_pad_dim_2 = data['A_full_size_pad'][0]
            print('predictions for orthopatches: ', direction)
            for i in tqdm(range(0, len(data[direction])), desc="Inference progress"):
                size_0 = stride * math.ceil(((full_pad_dim_1 - patch_size) / stride) + 1)
                size_1 = stride * math.ceil(((full_pad_dim_2 - patch_size) / stride) + 1)
                input = data[direction][i] # instead of A
                model.set_input(input)
                model.test()
                img = model.fake
                img = img[:, :, init_padding:-init_padding, init_padding:-init_padding]
                if i % data['patches_per_slice'+"_"+direction] == 0:
                    if i != 0:
                        if direction == 'xy':
                            prediction_map = prediction_map[0:data['A_full_size_raw'][1], 0:data['A_full_size_raw'][2]]
                        if direction == 'zy':
                            prediction_map = prediction_map[0:data['A_full_size_raw'][0], 0:data['A_full_size_raw'][1]]
                        if direction == 'zx':
                            prediction_map = prediction_map[0:data['A_full_size_raw'][2], 0:data['A_full_size_raw'][0]]

                        dir_volume.append(prediction_map)

                    prediction_map = np.zeros((size_0, size_1), dtype=np.float32)
                    prediction_slices = build_slices(prediction_map, [stride,stride], [stride,stride])
                    pred_index = 0

                img = torch.squeeze(torch.squeeze(img, 0), 0)
                img = tensor2im(img)
                #img = (tensor2im(img) * 255).astype(np.uint8)

                prediction_map[prediction_slices[pred_index]] += img
                pred_index += 1
            prediction_volume.append(dir_volume)

        prediction_volume[0] = np.asarray(prediction_volume[0])
        prediction_volume[0] = (prediction_volume[0] - prediction_volume[0].min())/(prediction_volume[0].max() - prediction_volume[0].min())

        prediction_volume[1] = np.transpose(np.asarray(prediction_volume[1]), (1, 2, 0))
        prediction_volume[1] = (prediction_volume[1] - prediction_volume[1].min()) / (prediction_volume[1].max() - prediction_volume[1].min())

        prediction_volume[2] = np.transpose(np.asarray(prediction_volume[2]), (2, 0, 1))
        prediction_volume[2] = (prediction_volume[2] - prediction_volume[2].min()) / (prediction_volume[2].max() - prediction_volume[2].min())

        prediction_volume_full = ((prediction_volume[0] + prediction_volume[1] + prediction_volume[2]) / 3)

        prediction_volume_full = (255 * prediction_volume_full).astype(np.uint8)

        print("writing prediction of shape: ", prediction_volume_full.shape)

        tifffile.imwrite(opt.results_dir + "/generated_" + os.path.basename(data['A_paths'][0]), prediction_volume_full)

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.input_nc = 1
    opt.output_nc = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.dataset_mode = 'patched_2_5d'
    inference_2_5D(opt)
