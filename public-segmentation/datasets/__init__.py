"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import logging
import os

import torch.utils.data

from .base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        # print(f"name = {name}, class = {cls}")
        if name.lower() == target_dataset_name.lower():  # and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (
                dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_data_loaders(opt, splits=("train",)):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        from .datasets import create_dataset
        dataset = create_dataset(opt)
    """
    data_loaders = dict()
    assert len(splits) >= 1, "at least one split must be provided"
    for split in splits:
        assert split in ["train", "val", "test"], f"split must be train|val|test|infer but was {split}"
        data_loaders[split] = CustomDatasetDataLoader(opt, split, opt.dataset_mode, opt.batch_size,
                                                      not opt.serial_batches, opt.num_threads)

    # Fetch extra data loader that will be used just for inference
    # Note that by default "val" is used for inference
    if opt.isTrain and opt.infer_dataset_root is not None:  # else we are test or not set
        phase = "val"
        dataroot = opt.infer_dataset_root
        assert os.path.exists(dataroot), f"infer_dataset_root {dataroot} does not exist"
        assert os.path.exists(os.path.join(dataroot, phase)), f"couldn't find {os.path.join(dataroot, phase)}"
        data_loaders["infer"] = CustomDatasetDataLoader(opt, phase, "infer", opt.batch_size, False, opt.num_threads,
                                                        dataroot=dataroot)
        logging.info(f"added extra inference dataset from {os.path.join(dataroot, phase)}")

    # Fetch extra data loader that will be used just for adversarial training
    # This will attempt to load "train" and "val" phases
    if opt.isTrain and opt.include_adversarial:
        dataroot = opt.adversarial_dataset_root
        assert os.path.exists(dataroot), f"adversarial dataroot {dataroot} does not exist"
        data_loaders["adv_train"] = CustomDatasetDataLoader(opt, "train", "adversarial", opt.batch_size,
                                                            not opt.serial_batches, opt.num_threads, dataroot=dataroot)
        # data_loaders["adv_val"] = CustomDatasetDataLoader(opt, "val", "adversarial", 4, False, opt.num_threads,
        #                                                   dataroot=dataroot)
        logging.info(f"added adversarial dataset from {os.path.join(dataroot)}")

    return data_loaders


class CustomDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, phase, dataset_mode, batch_size, shuffle, num_workers, **kwargs):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(dataset_mode)
        self.dataset = dataset_class(opt, phase=phase, **kwargs)
        print(f"dataset {type(self.dataset).__name__} with len {len(self.dataset)} was created for split {phase}")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

    def get_random(self):
        """ used"""
        import random
        cuda = len(self.opt.gpu_ids) > 0
        ind = random.choice(range(0, len(self.dataloader)))
        for i, data in enumerate(self.dataloader):
            if i == ind:
                return self.process_data(data, cuda)
        raise ValueError("mismatch in sizes?")

    def process_data(self, data, to_cuda):
        """
        This function will process the data group returned by the dataset into just inputs and labels

        Do this here instead of in dataset to make visualizations easier later by maintaining data in independent
        dictionary keys
        """
        if "pseudos" in data:
            input = torch.cat([data["images"], data["pseudos"]], dim=1)  # concat along channels dimension
        else:
            input = data["images"]
        labels = {k: v for k, v in data.items() if k not in ["images", "pseudos", "image_paths"]}

        if to_cuda:
            input = input.cuda()
            labels = {k: v.cuda() for k, v in labels.items()}
        return input, labels

    def renorm_data(self, data):
        """
            If data has been normalized (subtract mean and divide by std) this function can be used to undo that
            normalization.
        """
        for k, v in data.items():
            if self.dataset.mean_and_std is not None and k in self.dataset.mean_and_std and k != "labels":
                data[k] = (v * self.dataset.mean_and_std[k]["std"]) + self.dataset.mean_and_std[k]["mean"]
