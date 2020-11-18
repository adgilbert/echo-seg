"""Train the models"""

import logging
import os

import mlflow
import torch
from torch.autograd import Variable
from tqdm import tqdm

from datasets import create_data_loaders
from evaluate import evaluate
from evaluators import collect_metrics, collect_losses
from models import create_model
from models.net import collect_scheduler, collect_optimizer
from options.train_options import TrainOptions
from seg_utils import utils
from seg_utils.visualizer import Visualizer, get_visuals


def train(model, optimizer, losses, data_loader, metrics, opt, epoch, visualizer=None, prefix="train"):
    """Train the models on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of models
        losses: a class that takes batch_output and batch_labels and computes the loss for the batch
        data_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (Metrics) a class to keep track of metrics
        opt: (BaseOptions) parameters
        epoch: (int) current epoch number
        visualizer: (Visualizer) visualizer object for plotting results
        prefix: (str) prefix to use for metrics - default is 'train'
    """
    use_cuda = len(opt.gpu_ids) > 0
    # set models to training mode
    model.train()
    if metrics:
        metrics.epoch_reset(prefix)  # clear values from previous epoch

    # Use tqdm for progress bar
    with tqdm(total=int(len(data_loader) / opt.batch_size) + 1) as t:
        for i, data in enumerate(data_loader):

            input, labels = data_loader.process_data(data, use_cuda)

            # convert to torch Variables
            input = Variable(input)

            # compute models output and loss
            output = model(input)
            loss = losses(output, labels)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            # performs updates using calculated gradients
            optimizer.step()

            # calculate metrics
            if metrics:
                metrics(output, labels)

            if i % 10 == 5 and visualizer is not None:
                bbox_dict = dict(labels=labels["bboxes"], output=output["bboxes"]) if opt.include_bbox else None
                pseudos = data["pseudos"] if opt.include_pseudo else None
                viz_labels = labels["segs"] if "segs" in labels else labels["adversarial_segs"]
                visuals = \
                    get_visuals(
                        dict(images=data["images"], labels=viz_labels, output=output["segs"], pseudos=pseudos),
                        mean_and_std=data_loader.dataset.mean_and_std, bboxes=bbox_dict)[0]
                visualizer.display_current_results(visuals, epoch, True, prefix)

                if metrics:
                    loss_vals = metrics.to_dict(prefix=prefix)
                    # visualizer.print_current_losses(epoch, i*opt.batch_size / len(data_loader), epoch_metrics, "train")
                    visualizer.plot_current_losses(epoch, i * opt.batch_size / len(data_loader), loss_vals, prefix)

            t.set_postfix(loss=loss.item())
            t.update()

    if metrics:
        metrics.check_best(epoch)


def train_and_evaluate(model, dataloaders, optimizer, losses, metrics, opt, scheduler=None):
    """Train the models and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        data_loaders: (dict) contains DataLoader objects for at least "train" and "val"
        optimizer: (torch.optim) optimizer for parameters of models
        losses: (dict) a dictionary of loss functions
        metrics: (dict) a dictionary of functions that compute a metric
        opt: (Params) parameters
        scheduler: (torch.optim.lr_scheduler) Scheduler for optimizer
    NB: keys of output from model should match keys from losses and metrics and should be present in data from data_loader
    """
    assert all([t in dataloaders.keys() for t in ["train", "val"]]), "data_loaders must contain train and val"
    # reload weights from restore_file if specified
    if opt.restore_file is not None:
        restore_path = os.path.join(opt.checkpoints_dir, opt.experiment, opt.name, opt.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        if opt.dont_restore_optimizer:
            utils.load_checkpoint(restore_path, model, optimizer=None, loss=losses)
        else:
            utils.load_checkpoint(restore_path, model, optimizer, losses)

        # metrics.restore(opt.name, opt.checkpoints_dir, )

    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    epoch_metrics = dict()  # {f"best_{k}": 0 for k in metrics.check_best().keys()}  # best result is epoch 0 for now

    for epoch in range(opt.num_epochs):
        # Run one epoch
        lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Epoch {epoch + 1}/{opt.num_epochs} - lr = {lr}")

        if opt.include_adversarial:
            if epoch < opt.adversarial_start_epoch:
                assert opt.loss[0] in ["dice", "vanilla", "weighted_vanilla"], "might need to change this code section"
                logging.info(f"before adversarial start epoch, enabling only {opt.loss[0]}")
                losses.enable_only(opt.loss[0])
            elif epoch == opt.adversarial_start_epoch:
                logging.info("reached adversarial start epoch, enabling all")
                losses.enable_all()
            else:
                losses.enable_all()

        train(model, optimizer, losses, dataloaders["train"], metrics, opt, epoch, visualizer)
        epoch_metrics.update(metrics.to_dict(prefix="train"))

        # Evaluate for one epoch on validation set
        evaluate(model, dataloaders["val"], metrics, opt, epoch, visualizer)
        epoch_metrics.update(metrics.to_dict(prefix="val", include_best=False))

        # Perform adversarial training loop
        if opt.include_adversarial and not opt.batch_alternate_adversarial and opt.adversarial_start_epoch <= epoch < opt.num_epochs - 1:
            # Evaluate before adversarial training
            if "infer" in dataloaders:
                evaluate(model, dataloaders["infer"], metrics, opt, epoch, visualizer,
                         prefix=dataloaders["infer"].dataset.name + "_pre_adv")
                epoch_metrics.update(metrics.to_dict(prefix=dataloaders["infer"].dataset.name, include_best=False))

            # set adversarial loss to target the 'adversarial_segs' label from dataloader
            losses.set_loss_target("adversarial", "adversarial_segs")
            losses.enable_only("adversarial")
            # Don't pass in metrics or visualizer for adversarial training - they will break
            train(model, optimizer, losses, dataloaders["adv_train"], metrics, opt, epoch, visualizer,
                  prefix="adv_train")
            epoch_metrics.update(metrics.to_dict(prefix="adv_train"))
            losses.enable_all()
            losses.reset_targets()  # target 'segs' again

        # update schedulers if present
        if scheduler is not None:
            scheduler.step()
        losses.scheduler_step()

        # test on inference
        if "infer" in dataloaders and epoch >= opt.start_infer_epoch:
            evaluate(model, dataloaders["infer"], metrics, opt, epoch, visualizer,
                     prefix=dataloaders["infer"].dataset.name)
            epoch_metrics.update(metrics.to_dict(prefix=dataloaders["infer"].dataset.name))

        mlflow.log_metrics(epoch_metrics, step=epoch)

        # models can be saved for each metric.
        tags = []  # ["latest"]  # always save latest model
        for k, val in metrics.is_best.items():
            # add others here if model should be saved
            save_tags = ["val_best_dice"] if opt.output_nc < 4 else ["val_best_lv_endo_dice"]
            if "infer" in dataloaders:
                save_tags.append(f"{dataloaders['infer'].dataset.name}_best_lv_simplicity")
            if k in save_tags and "train" not in k and val == epoch:
                logging.info(f"- found new best accuracy for metric {k}: {epoch_metrics[k.replace('best_', '')]}")
                if "simplicity" in k:
                    tags.append("infer_best_simplicity")
                elif "dice" in k:
                    tags.append("val_best_dice")
                else:
                    tags.append(k)

        # Save weights
        utils.save_checkpoint(os.path.join(opt.checkpoints_dir, opt.experiment, opt.name),
                              {'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict(),
                               'loss_dict': losses.state_dict()},
                              tags=tags,
                              prefix=opt.output_prefix)
    metrics.print_best()


def run_training(opt_str=None, tags=None):
    """ Run a training round.
    If opt_str is None will process input from command line
    """

    # Load the parameters from json file
    opt = TrainOptions().parse(opt_str)  # get training options

    # Set the random seed for reproducible experiments
    torch.manual_seed(21)
    if len(opt.gpu_ids) > 0:
        torch.cuda.manual_seed(21)

    # Set the logger
    utils.set_logger(os.path.join(opt.checkpoints_dir, opt.experiment, opt.name, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    dataloaders = create_data_loaders(opt, ("train", "val"))
    logging.info("- done.")

    # Define the models and optimizer
    model = create_model(opt)
    optimizer = collect_optimizer(opt, model)

    # fetch loss function and metrics
    losses = collect_losses(opt)
    metrics = collect_metrics(opt, losses)

    # fetch schedulers
    scheduler = collect_scheduler(opt, optimizer)

    # Initialize mlflow experiment tracker
    mlflow.set_experiment(opt.experiment)
    # run_id = utils.find_existing_mlflow_run(opt)  # returns run_id if found else None
    with mlflow.start_run(run_name=opt.name + f"_{opt.phase}"):  # run_name is ignored if run_id found
        mlflow.set_tag("run_type", "train")
        mlflow.set_tag("dataset", dataloaders["train"].dataset.name)
        if tags is not None:
            for k, v in tags.items():
                mlflow.set_tag(k, v)
        mlflow.log_params(dict(**vars(opt)))

        # Train the models
        logging.info("Starting training for {} epoch(s)".format(opt.num_epochs))
        train_and_evaluate(model, dataloaders, optimizer, losses, metrics, opt, scheduler)


if __name__ == '__main__':
    run_training()
