from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        # separate initialization of restore file here since it is required for test
        parser.add_argument('--restore_experiment', default="./",
                            help="path to experiment dir (from current experiment dir) containing restore file")
        parser.add_argument('--restore_filename', required=True, help="name of file in restore_experiment dir to use")
        parser.add_argument("--no_images", default=False, action="store_true",
                            help="set to not save images output from model")
        # perhaps refine confidence argument to include a cutoff val?
        parser.add_argument('--include_confidence', default=False, action="store_true",
                            help="if set, will discard some images based on confidence metric (convexity)")
        parser.add_argument('--plot_miccai_fig', default=False, action="store_true",
                            help="if set, plot the best, median, and worst case results figure used for miccai paper.")
        parser.add_argument('--post_process', default=False, action="store_true",
                            help="apply post_processing to output")
        # rewrite default values
        parser.set_defaults(phase='val')
        self.isTrain = False
        return parser
