"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import gc
import torch
from util.visualizer import Visualizer
from argparse import Namespace


def train(opt):

#if __name__ == '__main__':
    # gc.collect()
    # torch.cuda.empty_cache()
    #opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        gc.collect()
        torch.cuda.empty_cache()
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        # model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        print(len(dataset))
        for i, data in enumerate(dataset):  # inner loop within one epoch

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    #if opt.display_id > 0:
                        #visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses) # for the patched dataset I'll use this

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate()  # update learning rates in the beginning of every epoch.

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))


#if __name__ == '__main__':
#train()

options = {
    'dataroot': '/media/lucas2/Untitled1/Synatose_Datasets/All_new_results/Training_Datasets/HeLa-NMR',
    'name': 'swinunetr_combined',
    'gpu_ids': [0],
    'checkpoints_dir': '/media/lucas2/Untitled1/Synatose_Datasets/All_new_results/Models/',
    'model': 'cycle_gan_3d',
    'input_nc': 1,
    'output_nc': 1,
    'ngf': 64,
    'ndf': 64,
    'netD': 'basic',
    'netG': 'swinunetr',
    'n_layers_D': 3,
    'norm': 'instance',
    'init_type': 'normal',
    'init_gain': 0.02,
    'no_dropout': True,
    'dataset_mode': 'patched_unaligned_3d',
    'patch_size': 64,
    'stride_A': 128,
    'stride_B': 128,
    'direction': 'AtoB',
    'phase': 'train',
    'serial_batches': True,
    'num_threads': 32,
    'batch_size': 1,
    'load_size': 128,
    'crop_size': 128,
    'max_dataset_size': float('inf'),
    'preprocess': 'none',
    'no_flip': False,
    'display_winsize': 256,
    'epoch': 'latest',
    'load_iter': 0,
    'verbose': True,
    'suffix': '',
    'use_wandb': True,
    'wandb_project_name': 'CycleGAN-and-pix2pix',
    'results_dir': '/media/lucas2/Untitled1/Synatose_Datasets/All_new_results/Generated_Datasets/HeLaAsMouse/',
    'aspect_ratio': 1.0,
    'n_epochs': 25,
    'n_epochs_decay': 25,
    'beta1': 0.5,
    'lr': 0.0002,
    'gan_mode': 'lsgan',
    'pool_size': 50,
    'lr_policy': 'linear',
    'lr_decay_iters': 50,
    'lambda_ssim_G': 0.2,
    'lambda_ssim_cycle': 0.2,
    'display_freq': 10,
    'display_ncols': 3,
    #'display_id': 0,
    #'display_server': 'localhost',
    #'display_env': 'main',
    #'display_port': 8097,
    #'update_html_freq': 1000,
    'print_freq': 100,
    'no_html': True,
    # network saving and loading parameters
    'save_latest_freq': 2500,
    'save_epoch_freq': 2,
    'save_by_iter': True,
    'continue_train': False,
    'isTrain': True,
    'epoch_count': 1,
    'lambda_A': 10.0,
    'lambda_B': 10.0,
    'lambda_identity': 0.5,
    'train_mode': '3d'
}
#
current_options = Namespace(**options)

train(current_options)

