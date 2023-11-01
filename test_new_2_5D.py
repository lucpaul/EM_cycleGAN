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

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.dataset_mode = 'patched'
    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
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
            for i in range(0, len(data[direction])):
                size_0 = 128 * math.ceil(((full_pad_dim_1 - 254) / 128) + 1)
                size_1 = 128 * math.ceil(((full_pad_dim_2 - 254) / 128) + 1)
                input = data[direction][i] # instead of A
                model.set_input(input)
                model.test()
                img = model.fake
                img = img[:, :, 63:-63, 63:-63]
                img = tensor2im(img)  # + 1) / 2 * 255
                if i % data['patches_per_slice'+"_"+direction] == 0:
                    if i != 0:
                        if direction == 'xy':
                            prediction_map = prediction_map[0:data['A_full_size_raw'][1], 0:data['A_full_size_raw'][2]]
                        if direction == 'zy':
                            prediction_map = prediction_map[0:data['A_full_size_raw'][0], 0:data['A_full_size_raw'][1]]
                        if direction == 'zx':
                            prediction_map = prediction_map[0:data['A_full_size_raw'][2], 0:data['A_full_size_raw'][0]]

                        prediction_map = (prediction_map * 255).astype(np.uint8)
                        #prediction_volume.append(prediction_map)
                        dir_volume.append(prediction_map)
                    prediction_map = np.zeros((size_0, size_1), dtype=np.float32)
                    prediction_slices = build_slices(prediction_map, [128,128], [128,128])
                    pred_index = 0

                prediction_map[prediction_slices[pred_index]] += np.squeeze(img)#.astype(np.uint8))
                pred_index += 1
            prediction_volume.append(dir_volume)

        prediction_volume[0] = np.asarray(prediction_volume[0])
        prediction_volume[0] = 255*(prediction_volume[0] - prediction_volume[0].min())/(prediction_volume[0].max() - prediction_volume[0].min())

        prediction_volume[1] = np.transpose(np.asarray(prediction_volume[1]),(1,2,0))
        prediction_volume[1] = 255 * (prediction_volume[1] - prediction_volume[1].min()) / (prediction_volume[1].max() - prediction_volume[1].min())

        prediction_volume[2] = np.transpose(np.asarray(prediction_volume[2]),(2,0,1))
        prediction_volume[2] = 255 * (prediction_volume[2] - prediction_volume[2].min()) / (prediction_volume[2].max() - prediction_volume[2].min())

        #print(prediction_volume[0].shape, prediction_volume[1].shape, prediction_volume[2].shape)
        prediction_volume = np.asarray(prediction_volume)
        prediction_volume = 255*(prediction_volume - prediction_volume.min())/(prediction_volume.max()-prediction_volume.min()) # This works
        prediction_volume_full = ((prediction_volume[0] + prediction_volume[1] + prediction_volume[2]) / 3).astype(np.uint8)
        prediction_volume = 255*(prediction_volume - prediction_volume.min())/(prediction_volume.max()-prediction_volume.min()) # Looks better with two normalisations
        print("writing prediction of shape: ", prediction_volume_full.shape)
        tifffile.imwrite(opt.results_dir+"/image_"+str(j+1)+"_prediction.tif", prediction_volume_full)
