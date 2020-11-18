import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage.morphology import binary_fill_holes
from torch.nn.functional import one_hot

from seg_utils.utils import convert_binary_output_to_classes

DEBUG = True


def show_img(img, title=None, ax=None, show=True):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.imshow(img)
    if title is not None:
        ax.set_title(title)
    if show:
        plt.show()


def to_cpu(t):
    if "cuda" in str(t.device):
        t = t.cpu()
    if t.requires_grad:
        t = t.detach()
    return t


def binarize_img(img):
    """ minumum processing to convert to a binary image and fill holes. Should not fail"""
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    img = binary_fill_holes(img).astype(np.float32)
    return img


def open_and_close(img, plot=False):
    """ find the largest contour... may fail if contours are not found..."""

    if plot:
        show_img(img, "og")
    kernel = np.ones((15, 15), np.uint8)
    num_iters = 3
    for i in range(num_iters):
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    for i in range(num_iters * 2):
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    for i in range(num_iters):
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    if plot:
        show_img(img, "closed")
    return img


def find_largest_contour(img, plot=False):
    """ find the largest contour... may fail if contours are not found..."""
    img = img.astype(np.uint8)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        largest = get_largest_contour(contours)
    except ValueError as e:
        print(e)
        return img
    # draw largest contour filled on image
    img = np.zeros(img.shape[0:2]).astype(np.uint8)
    cv2.drawContours(img, largest, -1, (1, 1, 1), -1)
    if plot:
        show_img(img, "contour image 2")
    return img


def smooth_contour(img, plot=False):
    for i in range(3):
        img = cv2.GaussianBlur(img, (15, 15), 0)
    if plot:
        show_img(img, "blurred")
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    if plot:
        show_img(img, "thresholded")
    return img


def simplicity(img):
    """ return the simplicity of a contour. Image should be a binary mask image """
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest = get_largest_contour(contours)
    perimeter = cv2.arcLength(largest[0], True)
    area = cv2.contourArea(largest[0])
    return np.sqrt(4 * np.pi * area) / perimeter


def choose_who_gets_overlapping_region_by_simplicity(img1, img2):
    """ for two images with an area of overlap this function will find who gets the overlapping region by evaluating
    the simplicity of each with and without the region """
    mask1 = img1.astype(np.bool)
    mask2 = img2.astype(np.bool)
    simplicity_img1_with = simplicity(mask1.astype(np.uint8))
    simplicity_img2_with = simplicity(mask2.astype(np.uint8))
    try:
        simplicity_img1_wout = simplicity((mask1 & ~mask2).astype(np.uint8))
    except ValueError:
        # entire LA is in overlapping region - LA gets it
        mask2 = mask2 & ~mask1
        return mask1.astype(np.uint8), mask2.astype(np.uint8)
    try:
        simplicity_img2_wout = simplicity((mask2 & ~mask1).astype(np.uint8))
    except ValueError:
        # entire LV is in overlapping region?? seems bad but give to LV
        print("WARNING: detected strange overlap between LV and LA")
        mask1 = mask1 & ~mask2
        return mask1.astype(np.uint8), mask2.astype(np.uint8)
    change1 = simplicity_img1_with - simplicity_img1_wout
    change2 = simplicity_img2_with - simplicity_img2_wout
    # higher simplicity with the region means that the region should be included
    if change1 > change2:
        mask2 = mask2 & ~mask1
    else:
        mask1 = mask1 & ~mask2
    return mask1.astype(np.uint8), mask2.astype(np.uint8)


def get_largest_contour(contours):
    """ find all contours above threshold """
    largest = None
    current_biggest = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > current_biggest:
            largest = contour
            current_biggest = area
    if largest is None:
        raise ValueError("no contours in image > 0 area")
    return [largest]


class MaskCombiner(object):
    """ combine masks for input to post processing then uncombine after.
    Specifically designed to combine the LV myocardium with LV blood pool for post processing
    Call combine before post-processing and uncombine after.
    """

    def __init__(self, channels_to_merge, output_channel=None):
        assert len(channels_to_merge) >= 2, "must provide at least two channels to merge"
        self.channels_to_merge = channels_to_merge
        if output_channel is None:
            self.output_channel = channels_to_merge[0]
        else:
            self.output_channel = output_channel

    def combine(self, output):
        """ merge several channels into a single channel to simplify post processing"""
        assert len(output.shape) == 3, f"function is designed to handle output of shape (C, W, H) got {output.shape}"
        output[self.output_channel, :, :] = output[(self.channels_to_merge), :, :].max(0)

    def uncombine(self, output):
        """ every channel in output should now be a processed binary mask.
        This function will subtract the binary masks from the other channels.
        """
        # import pdb
        # pdb.set_trace()
        # print('pdb')
        converted = output.argmax(0)
        oc = output[self.output_channel]
        for c in self.channels_to_merge:
            if c != self.output_channel:
                oc[converted == c] = output.min()  # these pixels will no longer be attributed to output channel
                print(f"uncombining channel {c} from {self.output_channel}")
                print(f"found {(converted == c).sum()} pixels for channel {c}")
                print(f"found {(converted == self.output_channel).sum()} pixels for channel {self.output_channel}")


class PostProcessMultiChannel:
    """ version 2 of a post processor intended for multi-channel output"""

    def __init__(self, output_nc):
        # self.mask_combiner = MaskCombiner(channels_to_merge=(1, 2), output_channel=2)
        self.output_nc = output_nc

    @staticmethod
    def _post_process_single(img, plot=False):
        img = binarize_img(img)  # convert to binary
        img = open_and_close(img, plot=plot)
        # contour finding... may fail
        try:
            img = find_largest_contour(img, plot=plot)
        except ValueError as e:
            print(f"post processing failed because {e}")
            # redo with plotting on
            try:
                find_largest_contour(img, plot=True)
            except ValueError:
                pass
        img = smooth_contour(img, plot=plot)
        img = img.astype(np.bool)
        return img

    def merge_multiple(self, la, other, plot=False):
        overlap = la.astype(np.bool) & other.astype(np.bool)
        if la.sum() > 0 and overlap.sum() / la.sum() > 0.04:
            if plot:
                f, axs = plt.subplots(1, 3)
                show_img(la, title="LA pre", ax=axs[0], show=False)
                show_img(other, title="Other pre", ax=axs[1], show=False)
                show_img(overlap, title="overlap pre", ax=axs[2])
            la, other = choose_who_gets_overlapping_region_by_simplicity(la, other)
        return la, other

    def process_four_channel(self, output):
        segs = to_cpu(output["segs"]).numpy()
        for i in range(segs.shape[0]):
            classes = segs[i].argmax(0)
            la_orig = classes == 3
            la = self._post_process_single(la_orig.astype(np.uint8).copy(), plot=False)
            lv_endo_orig = classes == 1
            lv_endo = self._post_process_single(lv_endo_orig.astype(np.uint8).copy())
            lv_epi_orig = ((classes == 1) | (classes == 2))
            lv_epi = self._post_process_single(lv_epi_orig.astype(np.uint8).copy())
            la, lv_endo = self.merge_multiple(la, lv_endo, plot=False)
            la, lv_epi = self.merge_multiple(la, lv_epi, plot=False)

            if la.sum() / la_orig.sum() < 0.01:
                la = la_orig
                logging.warning("post processing reduced la to <1%. Resetting to original")
            if lv_endo.sum() / lv_endo_orig.sum() < 0.01:
                lv_endo = lv_endo_orig
                logging.warning("post processing reduced lv endo to <1%. Resetting to original")
            if lv_epi.sum() / lv_epi_orig.sum() < 0.01:
                lv_epi = lv_epi_orig
                logging.warning("post processing reduced lv epi to <1%. Resetting to original")

            # may cut some regions off so find largest again
            la = find_largest_contour(la).astype(np.bool)
            lv_endo = find_largest_contour(lv_endo).astype(np.bool)
            lv_epi = find_largest_contour(lv_epi).astype(np.bool)

            # now fill image
            res = np.zeros(classes.shape, dtype=segs.dtype)
            res[la] = 3
            res[lv_endo] = 1
            res[lv_epi & ~lv_endo] = 2
            res = torch.LongTensor(res)
            res = one_hot(res, num_classes=self.output_nc).permute((2, 0, 1))
            segs[i] = np.array(res).astype(segs.dtype)
        output["segs"] = torch.tensor(segs)

    def process_two_channel(self, output, output_nc=None):
        output_nc = output_nc if output_nc is not None else self.output_nc
        segs = to_cpu(output["segs"]).numpy()
        for i in range(segs.shape[0]):
            classes = segs[i].argmax(0)
            lv_endo = classes == 1
            lv_endo = self._post_process_single(lv_endo.astype(np.uint8))

            # now fill image
            res = np.zeros(classes.shape, dtype=segs.dtype)
            res[lv_endo] = 1
            res = torch.LongTensor(res)
            res = one_hot(res, num_classes=output_nc).permute((2, 0, 1))
            segs[i] = np.array(res).astype(segs.dtype)
        output["segs"] = torch.tensor(segs)

    def __call__(self, output):
        if self.output_nc == 1:
            output["segs"] = convert_binary_output_to_classes(output["segs"])
            self.process_two_channel(output, 2)
        elif self.output_nc == 2:
            self.process_two_channel(output)
        elif self.output_nc == 4:
            self.process_four_channel(output)
        else:
            raise ValueError(f"post processing for output_nc = {self.output_nc} not implemented")

    def __repr__(self):
        return "PostProcessorMultiChannel"
#
