import argparse
import os
import sys

import torch

import datasets
from seg_utils import utils

IS_DEBUG = sys.gettrace() is not None

class BaseOptions:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and models class.
    """

    def __init__(self, save=True, display=True):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.save = save
        self.display = display

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', required=True,
                            help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='run_name',
                            help='name of this run. It decides where to store samples and models')
        parser.add_argument('--experiment', type=str, default="other_experiments",
                            help="experiments are collections of runs. Sets subdir within checkpoints dir and mlflow param ")
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--phase', type=str, default='train',
                            help='train | val | test. Expects a dataset subfolder of the same name.')
        # models parameters
        parser.add_argument('--model', type=str, default='unet_128',
                            help='chooses which models to use. [unet_128 | unet_256 | resnet_6blocks | resnet_9blocks]')
        parser.add_argument('--input_nc', type=int, default=1,
                            help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=1,
                            help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--filters', type=int, default=64, help='# of  filters in the last conv layer')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # parser.add_argument('--bbox_loss_weighting', default=0, type=float, help="if >0 than add bbox as an objective")
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='segmentation',
                            help='chooses how datasets are loaded. [segmentation | infer]')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=None,
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains '
                                 'more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--max_infer_dataset_size', type=int, default=None,
                            help='Max samples allowed for the inference dataset. Defaults to max_dataset_size.')
        parser.add_argument('--preprocess', type=str, default=['resize", "crop'], nargs="*",
                            help='preprocessing steps. Any combination of: [resize | crop | affine | norm | gamma]'
                                 'separate by space.')
        parser.add_argument('--display_winsize', type=int, default=256,
                            help='display window size for both visdom and HTML')
        # additional parameters
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {models}_{netG}_size{load_size}')
        parser.add_argument('--metrics', default="dice", type=str, nargs="*",
                            help="metrics to use, any combination of: [iou | dice | curvature | shape | hausdorff |"
                                 "simplicity | convexity | cosine | all]. Use all to get everything. separate by space")
        parser.add_argument('--loss', default="vanilla", type=str, nargs='*',
                            help="loss to use (can be multiple separated by space) Options are: "
                                 "dice | tversky | focal | crossentropy | vanilla | bbox. separate by space.")
        parser.add_argument('--weights', default=None, type=float, nargs='*',
                            help="weight for each loss function. Should match length of losses. Default is 1 for all."
                                 "separate by space.")
        parser.add_argument("--include_pseudo", default=False, action="store_true",
                            help="If true than will add the psuedo image as a channel. This means a pseudo image must "
                                 "be generated using the CycleGAN")
        parser.add_argument('--coord_conv', default=False, action="store_true", help="include coordinate convolution")

        parser.add_argument("--label_vals", type=str, default="all",
                            help="choose to only use one specific label. If used then output_nc should be set to 2."
                                 "Options are all, lv, lv_epi, or la.")
        parser.add_argument("--data_filter", type=str, default=(), nargs="*",
                            help="only include data files with these strings in the file name")
        parser.add_argument('--norm_dataroot', type=str, default=None,
                            help='path to the dataset used for mean normalization if desired. Else uses the current dataset.')

        self.initialized = True
        return parser

    def gather_options(self, inp_str=None):
        """Initialize our parser with basic options(only once).
        Add additional models-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in models and dataset classes.
        """

        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

            # get the basic options

            opt, _ = parser.parse_known_args(inp_str)

            # modify models-related parser options
            # model_name = opt.model
            # model_option_setter = models.get_option_setter(model_name)
            # parser = model_option_setter(parser, self.isTrain)
            # opt, _ = parser.parse_known_args()  # parse again with new defaults

            # modify dataset-related parser options
            dataset_name = opt.dataset_mode
            dataset_option_setter = datasets.get_option_setter(dataset_name)
            parser = dataset_option_setter(parser, self.isTrain)

            # save and return the parser
            self.parser = parser
        return self.parser

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        if self.save:
            # save to the disk
            expr_dir = os.path.join(opt.checkpoints_dir, opt.experiment, opt.name)
            utils.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    def parse(self, inp_str=None):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options(inp_str).parse_args(inp_str)
        opt.isTrain = self.isTrain  # train or test

        if IS_DEBUG:
            if self.display:
                print("Detected DEBUG mode, setting num_threads to 0")
            opt.num_threads = 0
        opt.include_bbox = "bbox" in opt.loss
        opt.include_adversarial = "adversarial" in opt.loss
        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if opt.label_vals != "all":
            if opt.output_nc > 2:
                print(f"single label_val specified ({opt.label_vals}), but output_nc is {opt.output_nc}, setting to 2")
                opt.output_nc = 2

        if not self.isTrain or (self.isTrain and opt.restore_filename is not None):
            opt.restore_file = os.path.join(opt.restore_experiment, opt.restore_filename)
        else:
            opt.restore_file = None

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # check other options for consistency
        if opt.include_pseudo:
            pseudo_dir = os.path.join(opt.dataroot, opt.phase, "pseudos")
            assert os.path.exists(pseudo_dir), f"--include_pseudo specified  but no pseudo dir at {pseudo_dir}"
        if opt.include_adversarial:
            assert opt.adversarial_dataset_root is not None, "adversarial loss is included but no dataset specified"

        if os.path.basename(opt.dataroot) == "":
            opt.dataroot = os.path.split(opt.dataroot)[0]  # remove trailing slashes
        if opt.isTrain and opt.infer_dataset_root is not None and os.path.basename(opt.infer_dataset_root) == "":
            opt.infer_dataset_root = os.path.split(opt.infer_dataset_root)[0]  # remove trailing slashes
        if self.display:
            self.print_options(opt)
        self.opt = opt
        return self.opt
