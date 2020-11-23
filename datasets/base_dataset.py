"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import json
import logging
import os
from abc import ABC

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from seg_utils.utils import get_bboxes
from .image_folder import make_dataset
from .segmentation_transforms import get_transform


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    Also define self.name in Dataset for inference

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt, dataroot, phase, has_label=True, has_pseudo=False):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.phase = phase
        self.is_train = phase == "train"
        if len(os.path.basename(dataroot).split('_')) > 1:
            short_name = "".join([word[0] for word in os.path.basename(dataroot).split('_')])  # abbreviated name
            self.name = "_".join([phase, short_name] + list(opt.data_filter))
        else:
            self.name = "_".join([phase, os.path.basename(dataroot)])

        self.has_label = has_label
        self.has_pseudo = has_pseudo
        self.include_bbox = opt.include_bbox

        self.data_filter = opt.data_filter

        self.dir_images = os.path.join(dataroot, phase, "images")  # get the image directory
        self.image_paths = sorted(make_dataset(self.dir_images, opt.max_dataset_size, filters=self.data_filter))

        # do preliminary check to make sure all labels and images are present
        if self.has_label:
            self.dir_labels = os.path.join(self.dir_images, "..", "labels")
            for ip in self.image_paths:
                assert os.path.exists(os.path.join(self.dir_labels, os.path.basename(
                    ip))), f"corresponding label for im {ip} not found, looked in {self.dir_labels}"
        else:
            # assert self.phase != "train", "train dataset must have labels"
            logging.info(f"No labels found for {self}, treating as inference dataset ")

        if self.has_pseudo:
            self.dir_pseudo = os.path.join(self.dir_images, "..", "pseudos")
            for ip in self.image_paths:
                assert os.path.exists(os.path.join(self.dir_pseudo, os.path.basename(
                    ip))), f"corresponding pseudo for im {ip} not found, looked in {self.dir_pseudo}"

        # Other params
        if "norm" in opt.preprocess:
            if not self.is_train and opt.norm_dataroot is not None:
                logging.info(f"loading mean normalization values from {opt.norm_dataroot}")
                self.mean_and_std = self.get_mean_std(opt.norm_dataroot)
            else:
                self.mean_and_std = self.get_mean_std()
        else:
            self.mean_and_std = None
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.input_nc

        self.label_values = dict(lv=85, lv_epi=170, la=255)

        if opt.label_vals != "all" and self.labels_have_multiple_regions():
            self.label_filter_val = self.label_values[opt.label_vals.lower()]
        else:
            self.label_filter_val = None

        self.transform = get_transform(opt, grayscale=(self.input_nc == 1), mean_and_std=self.mean_and_std,
                                       is_train=self.is_train)

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

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            images (tensor) - - an image in the input domain
            labels (tensor) - - its corresponding label
            pseudos (tensor) - - a pseudo image in the input domain
            img_paths (str) - - image paths
        """
        img_path = self.image_paths[index]
        image = Image.open(img_path)
        # original = image.copy()
        to_transform = dict(images=image)

        if self.has_label:
            label_path = os.path.join(self.dir_labels, os.path.basename(img_path))
            label = Image.open(label_path)
            label = self._filter_label(label, self.label_filter_val)
            # Image here is auto-loaded as 0-255, so we need to figure out how many labels are present
            to_transform.update(dict(labels=label))

        if self.has_pseudo:
            pseudo_path = os.path.join(self.dir_pseudo, os.path.basename(img_path))
            pseudo = Image.open(pseudo_path)
            to_transform.update(dict(pseudos=pseudo))
        transformed = self.transform(**to_transform)

        if self.include_bbox:
            bboxes = get_bboxes(transformed["labels"])
            transformed.update(bboxes=torch.as_tensor(bboxes).type(torch.float32))

        if self.has_label:
            transformed["segs"] = transformed.pop("labels")  # switch labeling to match output
        transformed["image_paths"] = img_path
        # transformed["original"] = original  # avoid having to do reverse transfor
        return transformed

    def __repr__(self):
        subclass = str(type(self)).split('.')[-1].strip("\'>")
        return f"(Dataset {subclass}: {self.name} in  {self.dir_images})"

    def get_mean_std(self, stat_file_root=None):
        """
        Load dataset mean and standard deviation. First try to load it from file. If it is not there then calculate and
        save to file
        :param stat_file_root: path to stat file if from a different dataset
        :return: mean and std
        """
        if stat_file_root is None:
            stat_file = os.path.join(self.dir_images, "..", "..", "train_stats.json")
        else:
            stat_file = os.path.join(stat_file_root, "train_stats.json")
        mean_and_std = json.load(open(stat_file, 'r')) if os.path.exists(stat_file) else None
        if mean_and_std is None or (self.has_label and "labels" not in mean_and_std):
            if self.is_train:
                mean_and_std = self.calculate_mean_and_std()
                json.dump(mean_and_std, open(stat_file, 'w'))
            else:
                raise ValueError(f"need to define mean and std for training dataset for {self}. Run script "
                                 f"calc_dataset_mean_std to generate train_stats.json")
        return mean_and_std

    @staticmethod
    def _calc_mean_std(image_paths):
        n, mean, std = 0, 0, 0
        for ip in image_paths:
            img = np.array(Image.open(ip)) / 255.
            n += img.size
            mean += img.sum()
            std += (img ** 2).sum()
        mean /= n
        std = np.sqrt(std / n - mean ** 2)
        return mean, std

    def calculate_mean_and_std(self):
        """ returns a dict with mean and std for image and pseudo dirs"""
        image_mean, image_std = self._calc_mean_std(self.image_paths)
        res = dict(images=dict(mean=image_mean, std=image_std))
        if self.has_label:
            label_paths = [ip.replace(self.dir_images, self.dir_labels) for ip in self.image_paths]
            label_mean, label_std = self._calc_mean_std(label_paths)
            res.update(dict(labels=dict(mean=label_mean, std=label_std)))
        if self.has_pseudo:
            pseudo_paths = [ip.replace(self.dir_images, self.dir_pseudo) for ip in self.image_paths]
            pseudo_mean, pseudo_std = self._calc_mean_std(pseudo_paths)
            res.update(dict(pseudos=dict(mean=pseudo_mean, std=pseudo_std)))
        return res

    def _filter_label(self, label, label_filter_val):
        """ Filter a label image to only include one target region. """
        if label_filter_val is None:
            return label
        else:
            new_label = np.zeros_like(np.array(label))
            new_label[np.array(label) == label_filter_val] = 255
            return Image.fromarray(new_label)

    def labels_have_multiple_regions(self):
        """ Check if labels have multiple values. Useful for automatically processing EchoNet dataset (which contains
        only binary labels) and the other datasets (which contain mutliple labels) with the same code. """
        if not self.has_label:
            return False  # no labels so it doesnt matter
        vals = list()
        for i in range(min(10, len(self.image_paths))):  # check 10 images
            image_path = self.image_paths[i]
            label_path = os.path.join(self.dir_labels, os.path.basename(image_path))
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
