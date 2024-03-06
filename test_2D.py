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
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.util import tensor2im
from data.SliceBuilder import build_slices
import numpy as np
import tifffile
import math
from torchvision import transforms
import torch
from tqdm import tqdm
from util.util import adjust_patch_size


try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def inference(opt):
    #patch_size = opt.patch_size

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

    adjust_patch_size(opt)
    patch_size = opt.patch_size

    # Need to still test this again
    difference = 0
    for i in range(2, int(math.log(int(opt.netG[5:]), 2)+2)):
        difference += 2 ** i
    stride = patch_size - difference - 2

    init_padding = int((patch_size - stride) / 2)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    model.eval()
    transform = transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float32)
    ])
    for j, data in enumerate(dataset):
        prediction_volume = []
        data_is_array = False

        if type(data['A']) == np.ndarray:
            data_is_array = True
            input_list = data['A'][0]
        elif type(data['A']) == list:
            input_list = data['A']
        print("input_patches: ", len(input_list))
        for i in tqdm(range(0, len(input_list)), desc="Inference progress"):
            input = input_list[i]

            input = transform(input)
            if data_is_array:
                input = torch.unsqueeze(input, 0)

            model.set_input(input)
            model.test()
            img = model.fake
            img = img[:, :, init_padding:-init_padding, init_padding:-init_padding]

            size_0 = stride * math.ceil(((data['A_full_size_pad'][1] - patch_size) / stride) + 1)
            size_1 = stride * math.ceil(((data['A_full_size_pad'][2] - patch_size) / stride) + 1)


            if i % int(data['patches_per_slice']) == 0:
                if i != 0:
                    prediction_map = prediction_map[0:data['A_full_size_raw'][1], 0:data['A_full_size_raw'][2]]
                    prediction_volume.append(prediction_map)

                prediction_map = np.zeros((size_0, size_1), dtype=np.uint8)
                prediction_slices = build_slices(prediction_map, [stride, stride], [stride, stride])
                pred_index = 0

            img = torch.squeeze(torch.squeeze(img, 0), 0)
            #img = tensor2im(img)
            img = (tensor2im(img) * 255).astype(np.uint8)

            prediction_map[prediction_slices[pred_index]] += img
            pred_index += 1

            if i == len(input_list)-1:
                prediction_map = prediction_map[0:data['A_full_size_raw'][1], 0:data['A_full_size_raw'][2]]
                prediction_volume.append(prediction_map)

        prediction_volume = np.asarray(prediction_volume)

        tifffile.imwrite(opt.results_dir + "/generated_" + os.path.basename(data['A_paths'][0]), prediction_volume)

if __name__ == '__main__':
    # read out sys.argv arguments and parse
    opt = TestOptions().parse()
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.input_nc = 1
    opt.output_nc = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.dataset_mode = 'patched_2d'
    inference(opt)
    TestOptions().save_options(opt)