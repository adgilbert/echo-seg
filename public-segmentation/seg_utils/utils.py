import logging
import os
import shutil

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import one_hot


def find_existing_mlflow_run(opt):
    mlflow.set_experiment(opt.experiment)
    run_df = mlflow.search_runs()  # returns database of keys
    # list of keys that will actually change results to use as database filters.
    # One could use all keys but that risks recreating runs for stupid changes. The risk of this approach is that
    # it needs to be constantly updated as new params are added. It is easy to change to using all instead
    # (iterate over opt_dict instead of result_changers) so it can be changed back in the future.
    result_changers = ["name", "dataroot", "phase", "model", "filters", "norm", "init_type", "init_gain", "no_dropout",
                       "serial_batches", "batch_size", "load_size", "crop_size", "max_dataset_size", "preprocess",
                       "weights", "include_pseudo", "coord_conv", "num_epochs", "vbeta1", "lr", "restore_file",
                       "include_bbox", "isTrain", "num_test", "include_confidence",
                       "infer_dataset_root", "adversarial_dataset_root"]
    # for all iterate over this instead:
    opt_dict = dict(**vars(opt))
    # may not be inside yet if experiment was just created and also some are train/test specific (won't be in opt_dict)
    result_changers = [rc for rc in result_changers if f"params.{rc}" in list(run_df.columns) and rc in opt_dict]
    mask = pd.DataFrame([run_df[f"params.{key}"].astype(str) == str(opt_dict[key]) for key in result_changers]).T.all(
        axis=1)

    # for debugging purposes we can do one at a time
    # mask = pd.DataFrame([True]*run_df.shape[0]).T
    # for key, val in result_changers:
    #     mask &= run_df[].astype(str) == str(val)
    match_rows = run_df[mask]

    if match_rows.shape[0] > 0:
        assert match_rows.shape[0] == 1, f"multiple matching runs found: \n{match_rows}"
        logging.info(f"Restoring run from: {match_rows.iloc[0].run_id}")
        return match_rows.iloc[0].run_id
    logging.info("Creating new run")
    return None


class DataSaver:
    """ A class to save data during teh course of a run. Save images will also save inputs and segs but will
    significantly increase the size.
    """

    def __init__(self, save_images=True):
        self.df = pd.DataFrame()
        self.save_images = save_images
        self.experiment_name = None
        self.run_name = None

    def set_experiment(self, experiment_name):
        self.experiment_name = experiment_name

    def set_run(self, run_name):
        self.run_name = run_name

    @property
    def has_results(self):
        return self.df.shape[0] > 0

    def __call__(self, input, output, batch_res):
        res = dict(experiment=self.experiment_name, run=self.run_name)
        for k, v in batch_res.items():
            if len(v) > 0 and v[0] is not None:
                res[k] = float(v[0])  # returned val from metrics is always list
        if self.save_images:
            res["image"] = input["images"].detach().cpu().numpy().squeeze()
            res["label"] = input["segs"].detach().cpu().numpy().squeeze()
            res["output"] = output["segs"].detach().cpu().numpy().squeeze()
        res["path"] = input["image_paths"][0]
        try:
            res["dataroot"], res["phase"], _, res["filename"] = res["path"].split(os.sep)[-4:]
        except IndexError:
            pass  # path didn't have expected form - may happen with new setup.
        self.df = self.df.append(res, ignore_index=True)

    def save(self, filename):
        if self.has_results:
            if os.path.exists(filename):
                pvs_df = pd.read_csv(filename, index_col=0)
                df = pd.concat([pvs_df, self.df], ignore_index=True)
            else:
                df = self.df
            df.to_csv(filename)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # remove previous handlers for multiple experiments
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


def save_checkpoint(checkpoint, state, tags, prefix=''):
    """Saves models and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        checkpoint: (string) folder where parameters are to be saved
        state: (dict) contains models's state_dict, may contain other keys such as epoch, optimizer state_dict
        tags: (list) types of model being saved (e.g. latest, best, etc...)

    """
    if len(tags) > 0:  # may call with no tags --  then don't save anything
        tag = tags[0]  # latest
        filepath = os.path.join(checkpoint, f'{prefix}{tag}.pth.tar')
        if not os.path.exists(checkpoint):
            print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
            os.mkdir(checkpoint)
        torch.save(state, filepath)

        # copy for others
        if len(tags) > 1:
            for tag in tags[1:]:
                shutil.copyfile(os.path.join(checkpoint, f'{tags[0]}.pth.tar'),
                                os.path.join(checkpoint, f'{tag}.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None, loss=None):
    """Loads models parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) models for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        loss (loss.Losses) optional: restore loss from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise OSError("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    if loss:
        loss.load_state_dict(checkpoint["loss_dict"])

    return checkpoint


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def rmdir(path):
    """  Removes a directory
    """
    if os.path.exists(path):
        shutil.rmtree(path)


def mk_clean_dir(path):
    rmdir(path)
    mkdir(path)


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image.astype(imtype)
        if image_tensor.shape[0] > 1:
            image_tensor = image_tensor.argmax(dim=0)  # handle multi-label output
        else:
            image_tensor = image_tensor[0]
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.min() < 0:
            image_numpy -= image_numpy.min()
        if image_numpy.max() > 1:
            image_numpy /= image_numpy.max()  # rescale if necessary
        image_numpy *= 255.
        # if image_numpy.shape[0] == 1:  # grayscale to RGB
        #     image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def get_bboxes(label, label_val=1.):
    """
    Get the bounding boxes of labels in the image
    :param label: torch tensor of shape (1, H, W) or (H, W)
    :param label_val: (float) value we are searching for
    :return:
    """
    assert len(label.shape) == 2 or (
                len(label.shape) == 3 and label.shape[0] == 1), "get_bboxes expects tensor of shape (1, H, W) or (H, W)"
    assert label.shape[-2] == label.shape[-1], "currently expects H = W, but it is an easy change if not"
    rr, cc = torch.meshgrid([torch.arange(0, label.shape[-2]), torch.arange(0, label.shape[-1])])
    z = -1 * torch.ones(*label.shape, dtype=torch.int64)
    rows = torch.where(label == label_val, rr, z)
    cols = torch.where(label == label_val, cc, z)
    min_row = rows[rows >= 0].min()
    max_row = rows[rows >= 0].max()
    min_col = cols[cols >= 0].min()
    max_col = cols[cols >= 0].max()
    # assert max_row > min_row > 0, "no label found in image"
    return np.array([min_row, min_col, max_row - min_row, max_col - min_col], dtype=np.float32) / label.shape[-1]


def dict_to_opts(d):
    """ We assume that if a value is None then k is a flag and should be appended with no value"""
    opts = list()
    for k, v in d.items():
        opts.append(f"--{k}")
        if type(v) is list:
            for sub_v in v:
                opts.append(str(sub_v))
        elif v is None:
            pass  # k is a flag
        else:
            opts.append(str(v))
    return opts


def clean_dict(d):
    clean_d = dict()
    for k, v in d.items():
        if type(v) is float:
            is_nan = np.isnan(v)
        else:
            is_nan = False
        if v is not None and v != "n/a" and not is_nan:
            clean_d[k] = v
        else:
            clean_d[k] = -0.1  # allow plotting still
    return clean_d


def convert_binary_output_to_classes(inp):
    """ convert a binary output to a channel output"""
    assert inp.shape[1] == 1, "should only be used for binary output"
    unsqueeze = inp.shape[0] == 1
    res = inp > 0
    res = one_hot(res.type(torch.int64), 2).squeeze()
    res = res.unsqueeze(0) if unsqueeze else res
    res = res.permute((0, 3, 1, 2))
    return res
