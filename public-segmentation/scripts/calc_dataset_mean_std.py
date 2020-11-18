import json
import os

from UltrasoundSegmentation.datasets.infer_dataset import InferDataset
from UltrasoundSegmentation.options.train_options import TrainOptions

if __name__ == '__main__':
    # run this file to get the mean and std of a dataset
    opt = TrainOptions().parse()
    assert opt.phase == "train", "mean and std should be calculated for training dataset"
    dset = InferDataset(opt, "train")
    mean_and_std = dset.calculate_mean_and_std()
    for im_type in mean_and_std:
        print(f"for dataset {im_type} mean is {mean_and_std[im_type]['mean']}, std is {mean_and_std[im_type]['std']}")
    save_path = os.path.join(dset.dir_images, "..", "..", "train_stats.json")
    print(f"saving to: {save_path}")
    json.dump(mean_and_std, open(save_path, 'w'))
