from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=3, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--disable_visdom', default=False, action="store_true", help='set to disable visdom')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000,
                            help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true',
                            help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # training parameters
        parser.add_argument('--num_epochs', type=int, default=25, help='# of iter at starting learning rate')
        # parser.add_argument('--data_percentage')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--adv_lr', type=float, default=0.01, help='initial_learning_rate for adv_loss adam')
        # extra datasets for visualization during training
        parser.add_argument("--infer_dataset_root", type=str, default=None,
                            help="if present, will load the val folder within each subfolders in infer_dataset_root"
                                 " as inference datasets for visualization/evaluation purposes.")
        # network saving and loading parameters
        parser.add_argument('--restore_experiment', type=str, default="./",
                            help='path to experiment dir (from current experiment dir) to use for restoring.')
        parser.add_argument('--restore_filename', type=str, default=None,
                            help='name of file in restore_experiment to restore. None = start from scratch.')
        parser.add_argument('--dont_restore_optimizer', default=True, action="store_false",
                            help='if set then the optimizers parameters will not be restored.')
        parser.add_argument("--output_prefix", type=str, default="", help="prefix before network save name")
        # Loss params
        parser.add_argument("--adversarial_dataset_root", type=str, default=None,
                            help="dataset to use for adversarial training. ")
        parser.add_argument("--adversarial_start_epoch", type=int, default=-1,
                            help="don't start adversarial training until this epoch")
        # Schedulers
        parser.add_argument('--lr_policy', type=str, default=None,
                            help="policy for lr scheduler. Option is only step right now.")
        parser.add_argument('--adv_lr_policy', type=str, default=None, help="policy for adversarial scheduler")
        parser.add_argument('--separate_discs', default=False, action="store_true",
                            help="separate discriminators in adversarial loss")
        parser.add_argument("--batch_alternate_adversarial", default=False, action="store_true",
                            help="if set then will alternate adversarial training by batches instead of by epoch.")
        parser.add_argument("--start_infer_epoch", default=0, type=int, help="when to start inference")
        # parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        self.isTrain = True

        return parser
