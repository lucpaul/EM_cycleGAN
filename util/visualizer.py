"""Visualizer utility for displaying, saving images, and logging information during training.

This module provides the Visualizer class, which uses Visdom and wandb for visualization and logging.
"""

import logging
import os
import sys
import time
from subprocess import Popen, PIPE

import numpy as np

from . import util

try:
    import wandb
except ImportError:
    logging.warning('wandb package cannot be found. The option "--use_wandb" will result in error.')

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


class Visualizer:
    """
    Visualizer class for displaying/saving images and logging information during training.

    Uses Visdom for display and wandb for experiment tracking. Also supports HTML export via dominate (HTML).
    """

    def __init__(self, opt):
        """
        Initialize the Visualizer class.

        Args:
            opt: Stores all the experiment flags; needs to be a subclass of BaseOptions.
        """
        self.opt = opt  # cache the option
        self.name = opt.name
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.wandb_project_name = opt.wandb_project_name
        self.current_epoch = 0
        self.train_mode = opt.train_mode
        self.raw_patch_size = opt.patch_size

        if self.use_wandb:
            self.wandb_run = (
                wandb.init(
                    dir=opt.checkpoints_dir,
                    project=self.wandb_project_name,
                    name=opt.name,
                    config=opt,
                )
                if not wandb.run
                else wandb.run
            )
            self.wandb_run._label(repo="CycleGAN-and-pix2pix")

        if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.name)):
            os.mkdir(os.path.join(opt.checkpoints_dir, opt.name))
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, "loss_log.txt")
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write("================ Training Loss (%s) ================\n" % now)

    def reset(self):
        """
        Reset the self.saved status.
        """
        self.saved = False

    def create_visdom_connections(self):
        """
        Attempt to connect to a Visdom server, or start a new server if connection fails.
        """
        cmd = sys.executable + " -m visdom.server -p %d &>/dev/null &" % self.port
        logging.warning("Could not connect to Visdom server. Trying to start a server....")
        logging.info("Command: %s", cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch):
        """
        Display current results on wandb and optionally save to an HTML file.

        Args:
            visuals (OrderedDict): Dictionary of images to display or save.
            epoch (int): The current epoch.
        """

        if self.use_wandb:
            columns = [key for key, _ in visuals.items()]
            columns.insert(0, "epoch")
            result_table = wandb.Table(columns=columns)
            table_row = [epoch]
            ims_dict = {}
            for label, image in visuals.items():
                if self.train_mode == "2d":
                    image_numpy = util.tensor2im(image)
                    if image_numpy.ndim >= 3:
                        wandb_image = wandb.Image(image_numpy[0])
                    else:
                        wandb_image = wandb.Image(image_numpy)
                elif self.train_mode == "3d":
                    image_numpy = np.squeeze(util.tensor2im(image), axis=0)
                    image_numpy = image_numpy[:, :, :, self.raw_patch_size // 2]
                    image_numpy = (image_numpy * 255).astype(np.uint8)
                    wandb_image = wandb.Image(image_numpy)
                table_row.append(wandb_image)
                ims_dict[label] = wandb_image
            self.wandb_run.log(ims_dict)
            if epoch != self.current_epoch:
                self.current_epoch = epoch
                result_table.add_data(*table_row)
                self.wandb_run.log({"Result": result_table})

    def plot_current_losses(self, losses):
        """
        Display the current losses on wandb.

        Args:
            losses (OrderedDict): Training losses stored as (name, float) pairs.
        """
        if self.use_wandb:
            self.wandb_run.log(losses)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """
        Print current losses to the console and save them to disk.

        Args:
            epoch (int): Current epoch.
            iters (int): Current training iteration during this epoch.
            losses (OrderedDict): Training losses stored as (name, float) pairs.
            t_comp (float): Computational time per data point (normalized by batch_size).
            t_data (float): Data loading time per data point (normalized by batch_size).
        """
        message = "(epoch: %d, iters: %d, time: %.3f, data: %.3f) " % (
            epoch,
            iters,
            t_comp,
            t_data,
        )
        for k, v in losses.items():
            message += "%s: %.3f " % (k, v)

        logging.info(message)
        with open(self.log_name, "a") as log_file:
            log_file.write("%s\n" % message)
