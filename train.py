"""General-purpose training script for image-to-image translation.

This script is for training on a dataroot folder with subfolders trainA and trainB. This will train generators and discriminators
A and B and save checkpoints for each epoch as checkpoints in a folder named '--checkpoints_dir'. Note that unlike the original
cycleGAN, more parameters are hard-coded at the end of the file. Primarily, patches are no longer preprocessed by resizing and cropping, and thus,
the preprocessing flag is hard-coded to 'none'.

The main flags you need to specify:
 - '--dataroot'
 - '--checkpoints_dir'
 - '--name' (the name of your model)
 - '--train_mode' (3d or 2d)
 - '--netG'  (the model backbone: e.g. resnet_9block, unet_32, swinunetr)
 - '--patch_size' (The side_length of the patches the dataset is tiled into.
                   If the provided value is not compatible with the backbone, the patch size will be automatically adjusted.)
 - 'stride_A' (the distance between two neighbouring patches for dataset A)
 - 'stride_B' (the distance between two neighbouring patches for dataset B)

Example:
    Train a CycleGAN model:
        python train.py --dataroot path/to/datasets --checkpoints_dir path/to/checkpoints --name my_cyclegan_model
                        --train_mode 3d --netG resnet_9blocks --patch_size 160 --stride_A 160 --stride_B 160

Further parameters such as options for loss, batch_size, epoch number etc. can be seen in the
options/base_options.py and options/train_options.py files.
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import gc
import torch
from util.visualizer import Visualizer
from util.util import adjust_patch_size


def train(opt):
    gc.collect()
    torch.cuda.empty_cache()
    if opt.train_mode == "3d":
        #opt.dataset_mode = 'zarr'
        opt.dataset_mode = 'patched_unaligned_3d'
    elif opt.train_mode == "2d":
        opt.dataset_mode = 'patched_unaligned_2d'

    adjust_patch_size(opt)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)

            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(losses) # for the patched dataset I'll use this

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate()

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    # Hard code a few options
    opt.preprocess = "none"
    opt.input_nc = 1
    opt.output_nc = 1
    opt.use_wandb = True
    train(opt)
    TrainOptions().save_options(opt) # Save the training options in train_opt.txt