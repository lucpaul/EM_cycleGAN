from .base_model import BaseModel


class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.add_argument('--model_suffix', type=str, default='_A', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
        # Below options are only when evaluating model using fid etc.
        parser.add_argument('--eval_direction', type=str, default='both-ways', help='whether to evaluate only domain A to domain B or both-ways.')
        parser.add_argument('--target_domain', type=str, default='', help='if eval_direction=both-ways, then the model in the reverse direction (BtoA) will be evaluated on this dataset')
        parser.add_argument('--source_domain',  type=str, default='', help='initially this will be equivalent to dataroot, but can be dynamically changed if eval_direction=both-ways')
        parser.add_argument('--target_domain_A_fid_file', type=str, default='', help='Only required when eval_direction=both-ways. Gives path to a FID .npz file from the source domain. When model was trained with trainA and trainB, this should be from A.')
        parser.add_argument('--target_domain_B_fid_file', type=str, default='', help='required. Gives path to a FID .npz file from the source domain. When models was trained with trainA and trainB, this should be from B.')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        if opt.test_mode == '2d' or opt.test_mode == '2.5d':
            from . import networks_2d as networks
        elif opt.test_mode == '3d':
            from . import networks_3d as networks

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real = input.to(self.device)

    def forward(self):
        """Run forward pass."""
        self.fake = self.netG(self.real)  # G(real)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
