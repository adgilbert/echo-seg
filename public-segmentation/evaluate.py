"""Evaluates the models"""
import logging
import os

import torch

from datasets import create_data_loaders
from evaluators import collect_metrics
from models import create_model
from options.test_options import TestOptions
from seg_utils import utils
from seg_utils.visualizer import get_visuals


def evaluate(model, dataloader, metrics, opt, epoch, visualizer, prefix="val"):
    """Evaluate the models on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a class that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (Metrics) a class to keep track of metrics
        opt: (BaseOptions) parameters
        epoch: (int) current epoch number
        visualizer: (Visualizer) visualizer object for plotting results
        prefix: (str) prefix to use for metrics - default is 'val'
    """
    use_cuda = len(opt.gpu_ids) > 0
    # set models to evaluation mode
    model.eval()
    metrics.epoch_reset(prefix)  # track metrics

    # compute metrics over the dataset
    for i, data in enumerate(dataloader):
        input, labels = dataloader.process_data(data, use_cuda)

        # compute models output
        output = model(input)
        metrics(output, labels)

        if i % 10 == 0:
            bbox_dict = dict(labels=labels["bboxes"], output=output["bboxes"]) if opt.include_bbox else None
            visuals = get_visuals(dict(images=data["images"], labels=labels["segs"], output=output["segs"]),
                                  mean_and_std=dataloader.dataset.mean_and_std, bboxes=bbox_dict)[0]
            visualizer.display_current_results(visuals, epoch, True, prefix)

            loss_vals = metrics.to_dict(prefix=prefix)
            # visualizer.print_current_losses(epoch, i*opt.batch_size / len(dataloader), losses, "val")
            visualizer.plot_current_losses(epoch, i * opt.batch_size / len(dataloader), loss_vals, prefix)

    metrics.check_best(epoch)
    logging.info(str(metrics))


if __name__ == '__main__':
    """
        Evaluate the models on the test set.
    """
    # Load the parameters from json file
    opt = TestOptions().parse()  # get training options

    # Set the random seed for reproducible experiments
    torch.manual_seed(21)
    if len(opt.gpu_ids) > 0:
        torch.cuda.manual_seed(21)

    # Set the logger
    utils.set_logger(os.path.join(opt.checkpoint_dir, opt.experiment, opt.name, f'{opt.phase}.log'))

    # Create the input data pipeline
    logging.info(f"Loading the datasets for {opt.phase}...")
    eval_dl = create_data_loaders(opt, (opt.phase,))[opt.phase]

    # Define the models and optimizer
    model = create_model(opt)

    # fetch loss function and metrics
    metrics = collect_metrics(opt)

    # Reload weights from the saved file
    restore_path = os.path.join(opt.checkpoints_dir, opt.experiment, opt.name, opt.restore_file + '.pth.tar')
    logging.info("Restoring parameters from {}".format(restore_path))
    utils.load_checkpoint(restore_path, model)

    # Evaluate
    logging.info("Starting evaluation")
    evaluate(model, eval_dl, metrics, opt)
    logging.info(str(metrics))
    logging.info("- done")
