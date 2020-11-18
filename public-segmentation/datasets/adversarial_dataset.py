import logging
import os.path

import numpy as np
from PIL import Image

from .base_dataset import BaseDataset
from .image_folder import make_dataset


class AdversarialDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of images/X.png and labels/X.png.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, phase, dataroot):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            dataroot -- location of the real images that will be fed into the network
            phase (str) -- sets the phase we're loading
        """

        # intentionally don't include label during training to make sure it is not being used
        has_label = False if phase == "train" else True
        BaseDataset.__init__(self, opt, dataroot, phase, has_label=has_label)

        # fetch labels from the original dataroot - synthetic images
        adv_dir = os.path.join(opt.dataroot, phase, "labels")
        self.adversarial_labels = make_dataset(adv_dir, opt.max_dataset_size, filters=self.data_filter)
        if opt.label_vals != "all" and self.adv_labels_have_multiple_regions():
            self.adv_label_filter_val = self.label_values[opt.label_vals.lower()]
        else:
            self.adv_label_filter_val = None

    def __getitem__(self, index):
        transformed = BaseDataset.__getitem__(self, index)

        # we can perform the transform separately since having the same transform applied is not important
        adversarial_ind = index % len(self.adversarial_labels)  # in case adversarial labels is shorter than others
        adversarial_label = Image.open(self.adversarial_labels[adversarial_ind])
        adversarial_label = self._filter_label(adversarial_label, self.adv_label_filter_val)
        transformed["adversarial_segs"] = self.transform(labels=adversarial_label)["labels"]
        return transformed

    def adv_labels_have_multiple_regions(self):
        vals = list()
        for i in range(10):  # check 10 images
            label_path = self.adversarial_labels[i]
            label = np.array(Image.open(label_path))
            num_vals = len(set(label.ravel()))
            vals.append(num_vals)
        if all([v == 2 for v in vals]):
            logging.info(f"found that labels for dataset {self.name} are binary... disabling label filtering")
            return False
        elif any([v == 2 for v in vals]):
            logging.warning([f"found some binary labels in dataset {self.name}... check values"])
            return True
        else:
            return True
