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


def train(opt):
    gc.collect()
    torch.cuda.empty_cache()
    if opt.train_mode == "3d":
        opt.dataset_mode = 'patched_unaligned_3d'
    elif opt.train_mode == "2d":
        opt.dataset_mode = 'patched_unaligned_2d'
    #old_patch_size = opt.patch_size
    _adjust_patch_size(opt)
    # if opt.netG.startswith('unet'):
    #     depth_factor = int(opt.netG[5:])
    #     # print("depth factor: ", depth_factor)
    #     patch_size = opt.patch_size
    #     # print(patch_size, (patch_size + 2) % depth_factor)
    #     if (patch_size + 2) % depth_factor == 0:
    #         pass
    #     else:
    #         # In the valid unet, the patch sizes that can be evenly downsampled in the layers (i.e. without residual) are
    #         # limited to values which are divisible by 32, after adding the pixels lost in the valid conv layer, i.e.:
    #         # 158 (instead of 160), 190 (instead of 192), 222 (instead of 224), etc. Below, the nearest available patch size
    #         # selected to patch the image accordingly. (Choosing a smaller value than the given patch size, should ensure
    #         # that the patches are not bigger than any dimensions of the whole input image)
    #         new_patch_sizes = opt.patch_size - torch.arange(1, depth_factor)
    #         new_patch_size = int(new_patch_sizes[(new_patch_sizes + 2) % depth_factor == 0])
    #         opt.patch_size = new_patch_size
    #         print(f"The provided patch size {old_patch_size} is not compatible with the chosen unet backbone with valid convolutions. Patch size was changed to {new_patch_size}")
    #
    # elif opt.netG.startswith("resnet"):
    #     patch_size = opt.patch_size
    #     if patch_size % 4 == 0:
    #         pass
    #     else:
    #         new_patch_sizes = opt.patch_size - torch.arange(1,4)
    #         new_patch_size = int(new_patch_sizes[(new_patch_sizes % 4) == 0])
    #         opt.patch_size = new_patch_size
    #         print(f"The provided patch size {old_patch_size} is not compatible with the resnet backbone. Patch size was changed to {new_patch_size}")
    #
    # elif opt.netG.startswith("swinunetr"):
    #     patch_size = opt.patch_size
    #     if patch_size % 32 == 0:
    #         pass
    #     else:
    #         new_patch_sizes = opt.patch_size - torch.arange(1,32)
    #         new_patch_size = int(new_patch_sizes[(new_patch_sizes % 32) == 0])
    #         opt.patch_size = new_patch_size
    #         print(f"The provided patch size {old_patch_size} is not compatible with the swinunetr backbone. Patch size was changed to {new_patch_size}")

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
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

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

def _adjust_patch_size(opt):

    old_patch_size = opt.patch_size

    if opt.netG.startswith('unet'):
        depth_factor = int(opt.netG[5:])
        # print("depth factor: ", depth_factor)
        patch_size = opt.patch_size
        # print(patch_size, (patch_size + 2) % depth_factor)
        if (patch_size + 2) % depth_factor == 0:
            pass
        else:
            # In the valid unet, the patch sizes that can be evenly downsampled in the layers (i.e. without residual) are
            # limited to values which are divisible by 32, after adding the pixels lost in the valid conv layer, i.e.:
            # 158 (instead of 160), 190 (instead of 192), 222 (instead of 224), etc. Below, the nearest available patch size
            # selected to patch the image accordingly. (Choosing a smaller value than the given patch size, should ensure
            # that the patches are not bigger than any dimensions of the whole input image)
            new_patch_sizes = opt.patch_size - torch.arange(1, depth_factor)
            new_patch_size = int(new_patch_sizes[(new_patch_sizes + 2) % depth_factor == 0])
            opt.patch_size = new_patch_size
            print(
                f"The provided patch size {old_patch_size} is not compatible with the chosen unet backbone with valid convolutions. Patch size was changed to {new_patch_size}")

    elif opt.netG.startswith("resnet"):
        patch_size = opt.patch_size
        if patch_size % 4 == 0:
            pass
        else:
            new_patch_sizes = opt.patch_size - torch.arange(1, 4)
            new_patch_size = int(new_patch_sizes[(new_patch_sizes % 4) == 0])
            opt.patch_size = new_patch_size
            print(
                f"The provided patch size {old_patch_size} is not compatible with the resnet backbone. Patch size was changed to {new_patch_size}")

    elif opt.netG.startswith("swinunetr"):
        patch_size = opt.patch_size
        if patch_size % 32 == 0:
            pass
        else:
            new_patch_sizes = opt.patch_size - torch.arange(1, 32)
            new_patch_size = int(new_patch_sizes[(new_patch_sizes % 32) == 0])
            opt.patch_size = new_patch_size
            print(
                f"The provided patch size {old_patch_size} is not compatible with the swinunetr backbone. Patch size was changed to {new_patch_size}")

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options