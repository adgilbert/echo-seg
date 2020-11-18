from .base_dataset import BaseDataset


class SegmentationDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of images/X.png and labels/X.png.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, phase=None, dataroot=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            phase (str) -- if given, will overwrite the phase in opt. This is useful for instance during training when
                            loading both training and validation datasets
        """
        if phase is None:
            phase = opt.phase  # use the one specified in command line
        if dataroot is None:
            dataroot = opt.dataroot
        BaseDataset.__init__(self, opt, dataroot, phase, has_label=True, has_pseudo=opt.include_pseudo)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
