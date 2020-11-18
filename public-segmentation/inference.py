"""Inference the models"""
import logging
import os

import mlflow
import torch
from tqdm import tqdm

from datasets import create_data_loaders
from evaluators import collect_metrics
from evaluators.confidence import SimplicityConvexityConfidence
from models import create_model
from options.test_options import TestOptions
from seg_utils import utils
from seg_utils.post_process_masks import PostProcessMultiChannel
from seg_utils.visualizer import get_visuals, ImageSaver


def inference(model, data_loader, metrics, opt, epoch, visualizer=None, image_saver=None, post_processor=None,
              data_saver=None):
    """Evaluate the models on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        data_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        opt: (BaseOptions) options controlling flow
        epoch: (int) epoch number
        visualizer: (Visualizer) that is only used if not None
        image_saver: a SaveImages object that will only used if epoch is None
        post_processor (PostProcessor) an object used to process the segmentation masks
        miccai_plotter (MiccaiPlotter) an object used to save the figure used for MICCAI

    """
    use_cuda = len(opt.gpu_ids) > 0
    # set models to evaluation mode
    model.eval()
    metrics.epoch_reset(data_loader.dataset.name)  # track metrics

    called_from_training = epoch is not None  # epoch will be specified if calling from train
    has_labels = data_loader.dataset.has_label  # check if we have labels
    name = data_loader.dataset.name
    batch_size = data_loader.dataloader.batch_size  # can be different for inference

    # compute metrics over the dataset
    with tqdm(total=int(len(data_loader) / batch_size) + 1) as t:
        for i, data in enumerate(data_loader):
            input, labels = data_loader.process_data(data, use_cuda)

            # compute models output
            output = model(input)
            if post_processor is not None:
                post_processor(output)
            if has_labels:
                batch_res = metrics(output, labels)
                if data_saver is not None:
                    normed_data = data.copy()
                    data_loader.renorm_data(normed_data)
                    data_saver(normed_data, output, batch_res)

            if called_from_training and i % 10 == 5 and visualizer is not None:
                vis_dict = {"images": data["images"], "output": output["segs"]}
                bbox_dict = dict(output=output["bboxes"]) if opt.include_bbox else None
                if has_labels:
                    vis_dict.update({"labels": labels["segs"]})
                    if opt.include_bbox:
                        bbox_dict.update(dict(label=labels["bboxes"]))
                visuals = get_visuals(vis_dict, mean_and_std=data_loader.dataset.mean_and_std, bboxes=bbox_dict)[0]
                visualizer.display_current_results(visuals, epoch, False, name)

                if has_labels:
                    loss_vals = metrics.to_dict(prefix=name)
                    # visualizer.print_current_losses(epoch, i / len(data_loader), losses, name)
                    visualizer.plot_current_losses(epoch, i * batch_size / len(data_loader), loss_vals, name)
            if image_saver is not None:
                image_saver(data, output)

            t.update()

    if not called_from_training:
        # some special ops for when inference is called independently
        mlflow.log_metrics(metrics.to_dict(prefix='mean'))
        mlflow.log_metrics(metrics.to_dict(method="std", prefix="std"))
        mlflow.log_metrics(metrics.to_dict(method="median", prefix="median"))
        mlflow.log_metrics(metrics.to_dict(method="median_absolute_deviation", prefix="mad"))
        if has_labels:
            metrics.add_to_summary_v2(opt)

    if has_labels:
        logging.info(f"inference for dataset {name}: {str(metrics)}")
        print(f"{metrics.failed_confidence}/{batch_size * len(data_loader)} images failed confidence")

        # metrics.add_to_summary_results(opt.name, data_loader.dataset.name, opt.checkpoints_dir)
    else:
        logging.info(f"inference for dataset {name} done")


def run_inference(opt_str=None, tags=None, data_saver=None):
    """
    Run inference (including confidence, post processing, and image saving)
    Leaving opt_str as None will process command line input
    """
    # Load the parameters from json file
    opt = TestOptions().parse(opt_str)  # get training options

    # Set the random seed for reproducible experiments
    torch.manual_seed(21)
    if len(opt.gpu_ids) > 0:
        torch.cuda.manual_seed(21)

    # Set the logger
    utils.set_logger(os.path.join(opt.checkpoints_dir, opt.experiment, opt.name, f'{opt.phase}.log'))

    # check for data saver
    if data_saver is not None:
        if opt.batch_size != 1:
            opt.batch_size = 1
            logging.info("changing batch_size to 1 to be able to use data saver")

    # Create the input data pipeline
    logging.info(f"Loading the datasets for {opt.phase}...")
    infer_dl = create_data_loaders(opt, (opt.phase,))[opt.phase]

    # Define the models and optimizer
    model = create_model(opt)

    # include confidence?
    confidence = SimplicityConvexityConfidence() if opt.include_confidence else None

    # fetch loss function and metrics
    metrics = collect_metrics(opt, confidence=confidence)

    # fetch image_saver
    if not opt.no_images:
        image_save_dir = os.path.join(opt.checkpoints_dir, opt.experiment, opt.name, "inference", infer_dl.dataset.name)
        image_saver = ImageSaver(image_save_dir, infer_dl.dataset.has_label, opt.include_pseudo, opt.include_bbox,
                                 save_masks_separately=False, mean_and_std=infer_dl.dataset.mean_and_std,
                                 confidence=confidence)
    else:
        image_saver = None

    # define segmentation post processor
    if opt.post_process:
        post_processor = PostProcessMultiChannel(opt.output_nc)
    else:
        post_processor = None
    logging.info(f"post processor = {post_processor}")

    # Reload weights from the saved file
    restore_path = os.path.join(opt.checkpoints_dir, opt.experiment, opt.name, opt.restore_file + '.pth.tar')
    logging.info("Restoring parameters from {}".format(restore_path))
    utils.load_checkpoint(restore_path, model)

    # initialize mlflow experiment tracker
    mlflow.set_experiment(opt.experiment)
    # run_id = utils.find_existing_mlflow_run(opt)  # returns run_id if found else None
    with mlflow.start_run(run_name=opt.name + f"_{opt.phase}"):  # run_name is ignored if run_id found
        mlflow.set_tag("run_type", "infer")
        mlflow.set_tag("dataset", infer_dl.dataset.name)
        if tags is not None:
            for k, v in tags.items():
                mlflow.set_tag(k, v)
        mlflow.log_params(dict(**vars(opt)))

        # Evaluate
        logging.info("Starting evaluation")
        inference(model, infer_dl, metrics, opt, None, visualizer=None, image_saver=image_saver,
                  post_processor=post_processor, data_saver=data_saver)

    return data_saver

if __name__ == '__main__':
    run_inference()
