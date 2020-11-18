import logging
import os
import random
import sys
from subprocess import Popen, PIPE

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from PIL import Image
from .utils import tensor2im, mk_clean_dir, mkdirs

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def show_img(img, grayscale=True):
    cmap = 'gray' if grayscale else "viridis"
    plt.imshow(img, cmap=cmap)
    plt.show()


def get_next_free_ind(d:dict):
    """ given a dictionary of indices, get the next free index """
    return max(d.values()) + 1


class ImageSaver:
    def __init__(self, results_dir, has_label=False, has_pseudo=False, include_bbox=False,
                 save_masks_separately=False, mean_and_std=None, confidence=None):
        self.results_dir = results_dir
        mkdirs(self.results_dir)
        self.has_label = has_label
        logging.info(f"initialized image saver to dir {results_dir} with has_label = {has_label}")
        self.include_bbox = include_bbox
        self.has_pseudo = has_pseudo
        if self.include_bbox:
            raise NotImplementedError("need to include bboxes in ImageSaver")
        self.ind = 0
        self.save_masks_separately = save_masks_separately
        if self.save_masks_separately:
            mk_clean_dir(os.path.join(self.results_dir, "labels"))
            mk_clean_dir(os.path.join(self.results_dir, "outputs"))
        self.mean_and_std = mean_and_std
        self.confidence = confidence

    def _fix_images_for_mean_and_std(self, data_batch):
        for k in data_batch:
            if type(data_batch[k]) == torch.Tensor:
                data_batch[k] = np.array(data_batch[k])
        if self.mean_and_std is not None:
            for k in ["images", "pseudos"]:
                if k in data_batch:
                    data_batch[k] = (data_batch[k] * self.mean_and_std[k]["std"]) + self.mean_and_std[k]["mean"]

    @staticmethod
    def _clean_ax(ax):
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    def _plt_imgs(self, axs, images, titles):  # pass in list of Nones for titles if no title desired
        for ax, img, title in zip(axs, images, titles):
            img += img.min()
            ax.imshow(img.squeeze(), cmap="gray", vmin=0., vmax=1.)
            if title is not None:
                self._clean_ax(ax)
                ax.set_title(title)

    def _plt_seg_overlay(self, axs, images, segs, titles, num_channels=1):
        self._plt_imgs(axs, images, titles)
        for ax, seg in zip(axs, segs):
            vmax = max([1, num_channels - 1])  # not sure if this is necessary but just in case
            ax.imshow(seg.squeeze(), cmap="viridis", alpha=.2, vmin=0, vmax=vmax)  # overlay

    def _plt_seg_difference(self, axs, segs1, segs2, titles, num_channels=1):
        for ax, seg1, seg2, title in zip(axs, segs1, segs2, titles):
            diff = seg1.astype(np.float32) - seg2.astype(np.float32)
            vmax = max([1, num_channels - 1])
            ax.imshow(diff.squeeze(), cmap="RdYlGn", vmin=-vmax, vmax=vmax)
            self._clean_ax(ax)
            ax.set_title(title)

    def save_label(self, save_dir, labels):
        for i, label in enumerate(labels):
            im = Image.fromarray(tensor2im(label))
            im.save(os.path.join(self.results_dir, save_dir, f"{self.ind}_{i}.png"))

    @staticmethod
    def get_titles(data_batch):
        return [os.path.splitext(os.path.basename(ip))[0] for ip in data_batch["image_paths"]]

    def confidence_titles(self, segs):
        titles = []
        for seg in segs:
            conf = self.confidence(seg, vals=True)
            titles += [f"s {conf['simplicity']:.2f}, c {conf['convexity']:.2f}"]
            # titles += ["Passed" if self.confidence(seg) else "FAILED"]
        return titles

    def __call__(self, data_batch, outputs):
        """ data_batch comes from data_loader, outputs comes from model"""
        if self.confidence is not None:
            titles = self.confidence_titles(outputs["segs"])
        else:
            titles = self.get_titles(data_batch)
        if outputs["segs"].requires_grad:
            outputs["segs"] = outputs["segs"].detach().cpu()
        batch_size = outputs["segs"].shape[0]
        outputs["segs"] = np.array(outputs["segs"])
        num_channels = outputs["segs"].shape[1]
        if num_channels > 1:
            outputs["segs"] = outputs["segs"].argmax(axis=1)
        self._fix_images_for_mean_and_std(data_batch)  # make images display okay
        if self.has_label:
            assert "segs" in data_batch, f"has_label was set, but segs not in data batch {data_batch.keys()}"
        ncols = 2 + (2 * self.has_label) + self.has_pseudo
        fig, axs = plt.subplots(batch_size, ncols, figsize=(3 * ncols, 3 * batch_size))
        if batch_size == 1:
            axs = np.expand_dims(axs, 0)  # batch size of 1 automatically flattens array
        self._plt_imgs(axs[:, 0], data_batch["images"], titles)
        self._plt_seg_overlay(axs[:, 1 + self.has_pseudo], data_batch["images"], outputs["segs"],
                              ["outputs"] * batch_size, num_channels=num_channels)
        if self.has_pseudo:
            self._plt_imgs(axs[:, 1], data_batch["pseudos"], "Pseudo")

        if self.has_label:
            self._plt_seg_overlay(axs[:, 2 + self.has_pseudo], data_batch["images"], data_batch["segs"],
                                  ["labels"] * batch_size, num_channels=num_channels)
            self._plt_seg_difference(axs[:, 3 + self.has_pseudo], outputs["segs"], data_batch["segs"],
                                     ["out-label"] * batch_size, num_channels=num_channels)
        plt.savefig(os.path.join(self.results_dir, f"{self.ind}.png"))
        mlflow.log_artifact(os.path.join(self.results_dir, f"{self.ind}.png"))
        if self.save_masks_separately:
            self.save_label("labels", data_batch["segs"])
            self.save_label("outputs", outputs["segs"])
        plt.close()
        self.ind += 1


def get_visuals(inp_tensors: dict, mean_and_std: dict = None, bboxes: dict = None, save_all: bool = False):
    assert len(inp_tensors) > 0, "at least one input must be provided"
    im = inp_tensors[next(iter(inp_tensors))]  # get first element
    batch_size = im.shape[0]
    visuals = list()
    if batch_size > 1 and not save_all:
        batch_index_select = [random.choice(range(batch_size))]
    else:
        batch_index_select = range(batch_size)
    for batch_index in batch_index_select:
        plot_ims = dict()
        # TODO: centralize renorming of data to one location!!
        for k, v in inp_tensors.items():
            if mean_and_std is not None and k in mean_and_std and k != "labels":
                v = (v * mean_and_std[k]["std"]) + mean_and_std[k]["mean"]
            if v is not None:
                plot_ims[k] = tensor2im(v[batch_index])
        if bboxes is not None:
            plot_ims["bboxes"] = add_bboxes([v[batch_index] for v in bboxes.values()])
        visuals.append(plot_ims)
    return visuals


def add_bboxes(bboxes, im_shape=(256, 256)):
    assert len(bboxes) < 3, "bbox uses channels so no more than 2 bboxes should be included"
    bbox_image = np.zeros((im_shape))
    for i, bbox in enumerate(bboxes):
        bbox *= im_shape[0]
        min_r, min_c, h, w = bbox.type(torch.int32)
        bbox_image[min_r:min_r + h, min_c:min_c + w] += 100.
    return bbox_image


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = dict(image_table=1) # image table is says which images are which
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        self.visdom_on = not opt.disable_visdom
        if self.visdom_on:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols + opt.include_bbox
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result, phase):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
            phase (str) -- phase name
        """
        if self.visdom_on:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:  # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                # create a table of images.
                title = self.name + ': ' + phase
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = tensor2im(image)
                    if len(image_numpy.shape) == 3:
                        image_numpy = image_numpy.transpose([2, 0, 1])
                    if len(image_numpy.shape) == 2:
                        image_numpy = np.expand_dims(image_numpy, 0)
                    images.append(image_numpy)
                    idx += 1
                white_image = np.ones_like(image_numpy) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    idx += 1
                if f"{phase}_images" not in self.display_id:
                    self.display_id[f"{phase}_images"] = get_next_free_ind(self.display_id)
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id[f"{phase}_images"],
                                    padding=2, opts=dict(title=title + ' images'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:  # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        if f"{phase}_images_{idx}" not in self.display_id:
                            self.display_id[f"{phase}_images_{idx}"] = get_next_free_ind(self.display_id)
                        image_numpy = tensor2im(image)
                        self.vis.image(image_numpy, opts=dict(title='_'.join([phase, label])),
                                       win=self.display_id[f"{phase}_images_{idx}"])
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

    def plot_current_losses(self, epoch, counter_ratio, losses, phase):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
            phase (str)           -- phase name
        """
        if self.visdom_on:
            if f"{phase}_metrics" not in self.display_id:
                self.display_id[f"{phase}_metrics"] = get_next_free_ind(self.display_id)
            if not hasattr(self, 'plot_data'):
                self.plot_data = dict()
            if phase not in self.plot_data.keys():
                self.plot_data[phase] = {'X': [], 'Y': [], 'legend': list(losses.keys())}
            self.plot_data[phase]['X'].append(epoch + counter_ratio)
            self.plot_data[phase]['Y'].append([l for l in losses.values()])
            try:
                self.vis.line(
                    X=np.stack([np.array(self.plot_data[phase]['X'])] * len(losses), 1),
                    Y=np.array(self.plot_data[phase]['Y']),
                    opts={
                        'title': self.name + '_' + phase + ': loss over time',
                        'legend': self.plot_data[phase]['legend'],
                        'xlabel': 'epoch',
                        'ylabel': 'loss'},
                    win=self.display_id[f"{phase}_metrics"])
            except VisdomExceptionBase:
                self.create_visdom_connections()
            except AssertionError:
                print(f"visdom plot failed, X= {self.plot_data[phase]['X'][-1]}, Y={self.plot_data[phase]['Y'][-1]}")
                print(losses)
                print(self.plot_data)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, phase):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            phase (str) -- phase name
        """
        message = f'(phase {phase} epoch: {epoch}, iters: {iters}) '
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        # with open(self.log_name, "a") as log_file:
        #     log_file.write('%s\n' % message)  # save the message
