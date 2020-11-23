import copy
import os.path

from .base_dataset import BaseDataset
from .image_folder import make_dataset


class InferDataset(BaseDataset):
    """A dataset class for inference.
    This acts the same as base_dataset except for it doesn't assume the presence of labels.
    """

    def __init__(self, opt, phase, dataroot=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            phase (str) -- sets the phase we're loading
            dataroot (str) -- location of data - overwrites opt.dataroot if given
        """
        if dataroot is None:
            dataroot = opt.dataroot
        has_label = os.path.exists(os.path.join(dataroot, phase, "labels"))
        if opt.max_infer_dataset_size is not None:
            opt = copy.deepcopy(opt)
            opt.max_dataset_size = opt.max_infer_dataset_size

        BaseDataset.__init__(self, opt, dataroot, phase, has_label=has_label, has_pseudo=opt.include_pseudo)

        # overwrite images to make sure we get all available for the inference dataset - don't use max_dataset_size
        self.image_paths = sorted(make_dataset(self.dir_images, filters=self.data_filter))
