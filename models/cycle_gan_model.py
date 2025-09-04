import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel


class CycleGANModel(BaseModel):
    """
    Implements the CycleGAN model for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Add new dataset-specific options, and rewrite default values for existing options.

        Args:
            parser: Original option parser.
            is_train (bool): Whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            ArgumentParser: The modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument(
                "--lambda_A",
                type=float,
                default=10.0,
                help="weight for cycle loss (A -> B -> A)",
            )
            parser.add_argument(
                "--lambda_B",
                type=float,
                default=10.0,
                help="weight for cycle loss (B -> A -> B)",
            )
            parser.add_argument(
                "--lambda_identity",
                type=float,
                default=0.5,
                help="use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1",
            )

        return parser

    def __init__(self, opt):
        """
        Initialize the CycleGAN class.

        Args:
            opt: Option class storing all the experiment flags; needs to be a subclass of BaseOptions.
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = [
            "D_A",
            "G_A",
            "cycle_A",
            "idt_A",
            "D_B",
            "G_B",
            "cycle_B",
            "idt_B",
        ]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:  # during test time, only load Gs
            self.model_names = ["G_A", "G_B"]

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        if self.opt.train_mode == "2d":
            from . import networks_2d as networks
            from .SSIM import SSIM as SSIM

        elif self.opt.train_mode == "3d":
            from . import networks_3d as networks
            from .SSIM import SSIM3D as SSIM

        self.netG_A = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )
        self.netG_B = networks.define_G(
            opt.output_nc,
            opt.input_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(
                opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )
            self.netD_B = networks.define_D(
                opt.input_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert opt.input_nc == opt.output_nc
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            if opt.lambda_ssim_cycle > 0:
                self.criterionCycle = SSIM()
            else:
                self.criterionCycle = torch.nn.L1Loss()

            self.criterion_SSIM_G = SSIM()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # Mixed precision option
            self.use_amp = getattr(opt, "use_amp", False)
            if self.use_amp and torch.cuda.is_available():
                self.scaler_G = torch.cuda.amp.GradScaler()
                self.scaler_D = torch.cuda.amp.GradScaler()
            else:
                self.scaler_G = None
                self.scaler_D = None

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): Includes the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)

        # if self.opt.use_zarr:

        # self.real_A = torch.permute(self.real_A, (0, 2, 1, 3))
        # self.real_B = torch.permute(self.real_B, (0, 2, 1, 3))

        # fself.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """
        Run forward pass; called by both functions <optimize_parameters> and <test>.
        """
        if getattr(self, "use_amp", False):
            with torch.cuda.amp.autocast():
                self.fake_B = self.netG_A(self.real_A)  # G_A(A)
                self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
                self.fake_A = self.netG_B(self.real_B)  # G_B(B)
                self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))
        else:
            self.fake_B = self.netG_A(self.real_A)
            self.rec_A = self.netG_B(self.fake_B)
            self.fake_A = self.netG_B(self.real_B)
            self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        """
        Calculate GAN loss for the discriminator.

        Args:
            netD (nn.Module): The discriminator D.
            real (Tensor): Real images.
            fake (Tensor): Images generated by a generator.

        Returns:
            Tensor: The discriminator loss.

        In the new version, we do NOT call loss_D.backward() here.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # Previous version:
        # loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """
        Calculate GAN loss for discriminator D_A.
        """
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        # Previous version:
        # self.loss_D_A.backward()

    def backward_D_B(self):
        """
        Calculate GAN loss for discriminator D_B.
        """
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        # Previous version:
        # self.loss_D_B.backward()

    def backward_G(self):
        """
        Calculate the loss for generators G_A and G_B.
        """
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_G_gen = self.opt.lambda_ssim_G
        lambda_G_cycle = self.opt.lambda_ssim_cycle

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = (1 - lambda_G_gen) * self.criterionGAN(self.netD_A(self.fake_B), True) + lambda_G_gen * (
            1 - self.criterion_SSIM_G(self.real_A, self.fake_B)
        )

        # GAN loss D_B(G_B(B))
        self.loss_G_B = (1 - lambda_G_gen) * self.criterionGAN(self.netD_B(self.fake_A), True) + lambda_G_gen * (
            1 - self.criterion_SSIM_G(self.real_B, self.fake_A)
        )
        # Forward cycle loss || G_B(G_A(A)) - A||
        # Backward cycle loss || G_A(G_B(B)) - B||
        if lambda_G_cycle > 0:
            self.loss_cycle_A = (1 - self.criterionCycle(self.rec_A, self.real_A)) * lambda_A  # if using with SSIM
            self.loss_cycle_B = (1 - self.criterionCycle(self.rec_B, self.real_B)) * lambda_B  # if using with SSIM
        else:
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combined loss (do not call backward here)
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        # Previous version:
        # self.loss_G.backward()

    def optimize_parameters(self):
        """
        Calculate losses, gradients, and update network weights; called in every training iteration.
        """
        if getattr(self, "use_amp", False) and self.scaler_G is not None and self.scaler_D is not None:
            # Mixed precision training
            # Forward and backward for G
            self.set_requires_grad([self.netD_A, self.netD_B], False)
            self.optimizer_G.zero_grad()
            with torch.cuda.amp.autocast():
                self.forward()
                self.backward_G()
            self.scaler_G.scale(self.loss_G).backward()
            self.scaler_G.step(self.optimizer_G)
            self.scaler_G.update()
            # Forward and backward for D
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()
            with torch.cuda.amp.autocast():
                self.backward_D_A()
                self.backward_D_B()
            total_loss_D = self.loss_D_A + self.loss_D_B
            self.scaler_D.scale(total_loss_D).backward()
            self.scaler_D.step(self.optimizer_D)
            self.scaler_D.update()
        else:
            # Standard (full precision) training
            self.forward()  # compute fake images and reconstruction images.
            # G_A and G_B
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
            self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            self.backward_G()  # calculate gradients for G_A and G_B
            self.loss_G.backward()  # <-- Added explicit backward for generator
            # Previous version:
            # self.backward_G()  # calculate gradients for G_A and G_B (called backward inside)
            self.optimizer_G.step()  # update G_A and G_B's weights
            # D_A and D_B
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
            self.backward_D_A()  # calculate gradients for D_A
            self.backward_D_B()  # calculate graidents for D_B
            (self.loss_D_A + self.loss_D_B).backward()  # <-- Added explicit backward for discriminator
            # Previous version:
            # self.backward_D_A()  # calculate gradients for D_A (called backward inside)
            # self.backward_D_B()  # calculate gradients for D_B (called backward inside)
            self.optimizer_D.step()  # update D_A and D_B's weights
