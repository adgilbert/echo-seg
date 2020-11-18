import logging
import os
from abc import abstractmethod
from typing import Callable, Tuple

import cv2
import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from scipy.spatial.distance import cosine as cosine_dist
from scipy.spatial.distance import directed_hausdorff
from torch.nn.functional import one_hot

from seg_utils.post_process_masks import get_largest_contour
from seg_utils.utils import convert_binary_output_to_classes, clean_dict
from .Curvature.single_mask_processing import Mask2Contour

try:
    from surface_distance.metrics import compute_surface_distances, compute_average_surface_distance
except ImportError:
    print("Could not import surface distance metrics. Install from https://github.com/deepmind/surface-distance if "
          "surface distance metrics will be used.")


def show_img(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


def convert_to_classes(inp): return torch.argmax(inp, dim=1)  # convert an input array to class values


def check_input_format(output, target):
    """ check input format of outpt and target

    output: BxCxHxW tensor where C is the number of classes
    target: Bx1xHxW tensor where every entry is an int in the range [0, C-1]
    """
    try:
        assert len(output.shape) == 4
        assert len(target.shape) == 4
        assert target.shape[1] == 1
        assert torch.max(target) <= output.shape[1] - 1
    except AssertionError:
        raise ValueError(f"Shape error: \nOutput should be [B, C, H, W], found {output.shape} "
                         f"\nTarget should be [B, 1, H, W], found {target.shape}. "
                         f"\nMax target should be <= C-1, found {torch.max(target)}")


class SliceMerger(object):
    def __init__(self, merge: tuple):
        """ merge is a tuple giving the slices along dimension 1 to merge"""
        assert len(merge) > 1, "must provide at least two slices to merge"
        self.merge = merge

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ merges all slices given by merge in dimension 1 using max
        output and target should """
        check_input_format(output, target)
        # create copies and detach from computation graph
        output = output.clone().detach()
        target = target.clone().detach()
        slices = np.arange(output.shape[1])  # all slices
        new_slice = self.merge[0]  # save res in this slice
        output[:, new_slice] = torch.max(output[:, self.merge], 1)[0]  # just want max_vals
        for s in slices:
            if s in self.merge:
                target[target == s] = new_slice  # merge vals in target
            else:
                subtract_val = sum(s > self.merge[1:])  # subtract one for every removed slice
                target[target == s] -= subtract_val
        keep_slices = [s for s in slices if s not in self.merge or s == new_slice]
        output = output[:, keep_slices]
        return output, target


def get_intersection_and_sums(output: torch.Tensor, target: torch.Tensor) -> tuple:
    """
    Calculates the intersection between output and target as well as the individual sums of each
    Args:
        output: BxCxHxW tensor where C is the number of classes
        target: Bx1xHxW tensor where every entry is an int in the range [0, C-1]

    Returns:
        intersection: area of intersection between output and target [B, C] array
        output_sum: sum of output for each class [B, C] array
        target_sum: sump of target for each class [B, C] array
    """
    check_input_format(output, target)
    num_classes = output.shape[1]
    # convert output and target to [C, B*H*W] tensors where every value is one-hot encoded and C is classes
    # for output have to first find chosen class then convert to one_hot
    output = one_hot(convert_to_classes(output), num_classes).type(torch.int).permute(3, 0, 1, 2).view(num_classes, -1)
    target = one_hot(target.squeeze(1), num_classes).type(torch.int).permute(3, 0, 1, 2).contiguous().view(num_classes,
                                                                                                           -1)
    assert output.shape == target.shape, f"output/target shape mismatch after processing ({output.shape}!={target.shape}"
    intersection = (output * target).sum(dim=1)
    output_sum = output.sum(dim=1)
    target_sum = target.sum(dim=1)
    return intersection, output_sum, target_sum


class MetricBase:
    def __init__(self, out_type: str = "segs", modifier: Callable = None, lower_is_better: bool = False,
                 requires_target=True, calculate_during_train=True):
        """
        Abstract base class which all metrics should inherit from
        :param out_type: what key to use in output/target dictionary
        :param modifier: a callable function which should take (output, target) as input and return output/target
        :param lower_is_better: decides whether or lower or higher value of the metric is better
        """
        self.best = {"train": None}
        self.phase = "train"
        self.results = list()
        self._type = out_type  # which key in output/target dict to use
        self.target_name = out_type
        self.lower_is_better = lower_is_better
        self.modifier = modifier
        self.requires_target = requires_target
        self.calculate_during_train = calculate_during_train

    @property
    def type(self):
        return self._type

    def set_target(self, target_name: str):
        self.target_name = target_name

    def reset_target(self):
        self.target_name = self._type

    def check_best(self, res):
        if res is not None:
            if self.best[self.phase] is None or (not self.lower_is_better and res >= self.best[self.phase]) or \
                    (self.lower_is_better and res <= self.best[self.phase]):
                self.best[self.phase] = res
                return True
        return False

    def epoch_reset(self, phase):
        # reset results for next epoch
        self.results = list()
        self.phase = phase
        if phase not in self.best:
            self.best[phase] = None

    def reduce(self, method="median"):
        assert method in dir(self), f"reduction method {method} not found"
        if len(self.results) > 0:
            return self.__getattribute__(method)()
        else:
            logging.debug("0 results found. Returning None")
            return None

    def absmean(self):
        return np.mean(abs(np.array(self.results)))

    def mean(self):
        mean = np.mean(np.array(self.results))
        if np.isnan(mean):
            logging.warning("Found NAN in results, using nanmean")
            mean = np.nanmean(np.array(self.results))
        return mean

    def median(self):
        med = np.median(np.array(self.results))
        if np.isnan(med):
            logging.warning("Found NAN in results, using nanmedian")
            med = np.nanmedian(np.array(self.results))
        return med

    def absmedian(self):
        return np.median(abs(np.array(self.results)))

    def std(self):
        std = np.std(np.array(self.results))
        if np.isnan(std):
            logging.warning("Found NAN in results, using nanstd")
            std = np.nanstd(np.array(self.results))
        return std

    def ci_95(self):
        res = np.array(self.results)
        return st.t.interval(0.95, len(res) - 1, loc=np.mean(res), scale=st.sem(res))

    def ci_low(self):
        return self.ci_95()[0]

    def ci_high(self):
        return self.ci_95()[1]

    def median_absolute_deviation(self):
        """ defined as median(|X_i - median(X)|)"""
        med = self.median()  # will handle nan and print warning.
        res = np.array(self.results)
        res = res[~np.isnan(res)]
        return np.median(abs(res - med))

    def set_best(self, phase, val):
        """ used for setting the best value when restoring"""
        self.best[phase] = val

    @abstractmethod
    def process_single(self, output, target=None):
        # implement here the actual metric, should return a val to add to self.results
        # a return value of None indicates that something went wrong in the processing. This class will ignore and
        # continue, but the subclass should log a warning message.
        pass

    def __call__(self, outputs: dict, targets: dict = None, confidence=None):
        if self.requires_target and (targets is None or self._type not in targets):
            return []
        if not self.calculate_during_train and self.phase == "train":
            return []
        self.batch_results = list()
        if confidence is None:
            confidence = [True] * outputs["segs"].shape[0]
        if targets is None or self._type not in targets:
            for o, c in zip(outputs[self._type], confidence):
                if c and o is not None:
                    o = o.unsqueeze(0)
                    res = self.process_single(o)
                    if res is not None:
                        self.results.append(res)
                        self.batch_results.append(res)
                    else:
                        self.batch_results.append(None)
        else:
            for o, t, c in zip(outputs[self._type], targets[self._type], confidence):
                if c and o is not None:
                    o, t = o.unsqueeze(0), t.unsqueeze(0)
                    if self.modifier is not None:
                        o, t = self.modifier(o, t)
                    res = self.process_single(o, t)
                    if res is not None:
                        self.results.append(res)
                        self.batch_results.append(res)
                    else:
                        self.batch_results.append(None)
        return self.batch_results


class IoU(MetricBase):
    """ wrapper around kornia to keep everything in one place
    Matching cross-entropy loss format, output is a [B, C, H, W] float tensor where C is number of classes
    target is a [B, H, W] integer tensor where every entry is in the range [0, C-1]

    Function first converts output to match target and then uses kornia to compute mean_iou
    """

    def __init__(self, num_classes: int, include_classes=None):
        MetricBase.__init__(self, out_type="segs")
        self.num_classes = num_classes
        self.include_classes = include_classes  # which classes to include

    def process_single(self, output: torch.Tensor, target: torch.Tensor = None) -> float:
        intersection, output_sum, target_sum = get_intersection_and_sums(output, target)
        iou = intersection.type(torch.float) / (output_sum + target_sum - intersection).type(torch.float)
        if self.include_classes is None:
            return torch.mean(iou[1:])  # return mean of non-background classes.
        else:
            raise NotImplementedError("add handling for specific include classes")


class DiceScore(MetricBase):
    def __init__(self, include_classes: list = None, modifier: Callable = None):
        """
        dices score for a segmentation result
        Args:
            include_classes: (list) which classes to include in the output, default is [1:] excluding background
            modifier: (Callable) an optional function to apply to output before calculations
        """
        MetricBase.__init__(self, out_type="segs", modifier=modifier)
        self.include_classes = include_classes  # which classes to include in the DiceScore

    def process_single(self, output: torch.Tensor, target: torch.Tensor = None) -> float:
        """
        Processes a single example
        Args:
            output: BxCxHxW tensor where C is the number of classes
            target: Bx1xHxW tensor where every entry is an int in the range [0, C-1]

        Returns: (float) the mean dice score

        """
        if output.shape[1] == 1:
            output = convert_binary_output_to_classes(output)
        intersection, output_sum, target_sum = get_intersection_and_sums(output, target)
        dice = 2 * intersection.type(torch.float) / (output_sum + target_sum).type(torch.float)
        if self.include_classes is None:
            return torch.mean(dice[1:])  # same as iou, return mean of values for non-background classes
        else:
            return torch.mean(dice[self.include_classes])  # return mean of values for designated classes


class Bias(MetricBase):
    def __init__(self, include_classes: list = None, modifier: Callable = None):
        """
        bias score for a segmentation result. Bias is the percentage difference in size of a given result
        Args:
            include_classes: (list) which classes to include in the output, default is [1:] excluding background
            modifier: (Callable) an optional function to apply to output before calculations
        """
        MetricBase.__init__(self, out_type="segs", modifier=modifier)
        self.include_classes = include_classes  # which classes to include in the DiceScore

    def process_single(self, output: torch.Tensor, target: torch.Tensor = None) -> float:
        """
        Processes a single example
        Args:
            output: BxCxHxW tensor where C is the number of classes
            target: Bx1xHxW tensor where every entry is an int in the range [0, C-1]

        Returns: (float) the mean dice score

        """
        if output.shape[1] == 1:
            output = convert_binary_output_to_classes(output)
        intersection, output_sum, target_sum = get_intersection_and_sums(output, target)
        bias = (output_sum - target_sum).type(torch.float) / (0.5 * output_sum + 0.5 * target_sum).type(torch.float)
        if self.include_classes is None:
            return torch.mean(bias[1:])  # same as iou, return mean of values for non-background classes
        else:
            return torch.mean(bias[self.include_classes])  # return mean of values for designated classes


class CurvatureIndividual(MetricBase):
    """ Calculate the curvature of the output segmentation """
    def __init__(self, segment_name, side):
        MetricBase.__init__(self, out_type="curve")
        assert segment_name in ['basal', 'mid', 'apical']
        assert side in [1, 2]
        self.curve_segment = '_'.join([segment_name, 'curvature', str(side), "mean_endo"])

    def process_single(self, out_cc: dict, target_cc: dict):
        res = (out_cc["curvature"][self.curve_segment] - target_cc["curvature"][self.curve_segment]) / \
              np.mean([abs(target_cc["curvature"][self.curve_segment]), abs(out_cc["curvature"][self.curve_segment])])
        return res


class Convexity(MetricBase):
    """ Calculate the convexity of the segmentation """
    def __init__(self, label_val, modifier=None):
        MetricBase.__init__(self, out_type="segs", modifier=modifier, requires_target=False,
                            calculate_during_train=False)
        self.label_val = label_val

    def process_single(self, output: torch.Tensor, target: torch.Tensor = None) -> float:
        out_labels = convert_to_classes(output).detach().numpy()
        selected_label_mask = (out_labels.squeeze() == self.label_val).astype(np.uint8)
        contours, _ = cv2.findContours(selected_label_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        try:
            largest = get_largest_contour(contours)[0]
        except ValueError as e:
            logging.warning(f"finding metric Convexity failed because {e}, skipping...")
            return None
        area = cv2.contourArea(largest)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        return area / hull_area


class Simplicity(MetricBase):
    """ Calculate the simplicty of the segmentation """
    def __init__(self, label_val, modifier=None):
        MetricBase.__init__(self, out_type="segs", modifier=modifier, requires_target=False,
                            calculate_during_train=False)
        self.label_val = label_val

    def process_single(self, output: torch.Tensor, target: torch.Tensor = None) -> float:
        out_labels = convert_to_classes(output).detach().numpy()
        selected_label_mask = (out_labels.squeeze() == self.label_val).astype(np.uint8)
        contours, _ = cv2.findContours(selected_label_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        try:
            largest = get_largest_contour(contours)[0]
        except ValueError as e:
            logging.warning(f"finding metric Simplicity failed because {e}, skipping...")
            return None
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        return np.sqrt(4 * np.pi * area) / perimeter


class CosineSim(MetricBase):
    """ Cosine similarity but calculates curvature internally rather than using the output.  Reason for doing this
    is to allow the use of the modifier syntax in metric base to calculate curvature on modified versions of the output.
    """
    def __init__(self, label_val: int, modifier: Callable = None, nsegs: int = None):
        MetricBase.__init__(self, out_type="segs", modifier=modifier)
        self.label_val = label_val
        if nsegs is None:
            self.suffix = ''
        else:
            self.suffix = f'_{nsegs}'

    def process_single(self, output: torch.Tensor, target: torch.Tensor) -> float:
        out_labels = convert_to_classes(output).detach().numpy()
        target = target.detach().numpy()
        selected_label_mask = 255 * (out_labels.squeeze() == self.label_val).astype(np.uint8)
        target_mask = 255 * (target.squeeze() == self.label_val).astype(np.uint8)
        try:
            out_cc = Mask2Contour(selected_label_mask).get_contour_and_markers(show=False)
            target_cc = Mask2Contour(target_mask).get_contour_and_markers(show=False)
        except Exception as e:
            # in some cases curvature will fail. In those cases do it again, but show what happened
            logging.warning(f"Mask2Contour failed because {e}. Skipping...")
            return None
        res = np.array(
            [1 - cosine_dist(u, v) for u, v in zip(out_cc["shape" + self.suffix], target_cc["shape" + self.suffix])])
        return res.mean()


class SurfaceDist(MetricBase):
    """ Calculate the distance between two surfaces """
    def __init__(self, label_val, modifier=None):
        MetricBase.__init__(self, out_type="segs", modifier=modifier, requires_target=True,
                            calculate_during_train=False)
        self.label_val = label_val

    def process_single(self, output: torch.Tensor, target: torch.Tensor = None) -> float:
        out_labels = convert_to_classes(output).detach().numpy()
        target_labels = target.detach().numpy()
        output_mask = out_labels.squeeze() == self.label_val
        target_mask = target_labels.squeeze() == self.label_val
        surface_distances = compute_surface_distances(target_mask, output_mask, (1., 1.))
        avg_dist = compute_average_surface_distance(surface_distances)
        return avg_dist[0]


class Curvature(MetricBase):
    def __init__(self):
        MetricBase.__init__(self, out_type="curve")

    @staticmethod
    def mse(inp, tar):
        return np.array([(i - t) ** 2 for (i, t) in zip(inp.values(), tar.values())]).mean()

    def process_single(self, out_cc: dict, target_cc: dict):
        res = self.mse(out_cc["curvature"], target_cc["curvature"])
        return res


class Hausdorff(MetricBase):
    def __init__(self):
        MetricBase.__init__(self, out_type="curve")

    def process_single(self, out_cc: dict, target_cc: dict):
        u, v = out_cc["contour"], target_cc["contour"]
        res = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
        return res


class Metrics:
    """Class to hold all of the metrics that will be calculated during training.
    Will also calculate auxillary outputs (curvature/bbox) if they are needed by any of the metrics
    """
    def __init__(self, metrics: dict, phase, confidence=None):
        """
        Args:
            metrics: a dict where each key is the metric name and the value is a subclass of MetricBase
            phase: the current training phase
            confidence: A confidence metric which can be used to determine if an output should be included in the
              metrics. This should be a callable which takes as input the segmentation and returns True/False. By default
              all images will be included.
        """
        self.metrics = metrics
        self.has_curve = any([m.type == "curve" for m in self.metrics.values()])
        self.has_bbox = any([m.type == "bbox" for m in self.metrics.values()])
        logging.info(f"Initialized metrics with curve = {self.has_curve}, bbox = {self.has_bbox}")
        self.failed_confidence = 0
        self.total = 0
        self.res = dict()
        self.is_best = dict()
        self.confidence = confidence
        self.phase = phase

    def check_best(self, epoch):
        """ checks whether each result is the best so far"""
        self._calc_res()
        for k, metric_class in self.metrics.items():
            if metric_class.check_best(self.res[k]):
                self.is_best['_'.join([self.phase, "best", k])] = epoch  # either epoch num or just True
        return self.is_best

    @staticmethod
    def _detach_and_to_cpu(t):
        if type(t) == torch.Tensor:
            if t.get_device() >= 0:
                t = t.data.cpu()
            if t.requires_grad:
                t = t.detach()
        return t

    @staticmethod
    def convert_to_mask(inp): return inp > 0.

    def _get_curvature(self, outputs: torch.Tensor, targets: torch.Tensor):
        output_curves, target_curves = list(), list()
        for output, target in zip(outputs, targets):
            output = 255 * self.convert_to_mask(output).numpy().squeeze().astype(np.uint8)
            target = 255 * target.numpy().squeeze().astype(np.uint8)
            try:
                out_cc = Mask2Contour(output).get_contour_and_markers(show=False)
                target_cc = Mask2Contour(target).get_contour_and_markers(show=False)
                output_curves.append(out_cc)
                target_curves.append(target_cc)
            except:
                # in some cases curvature will fail. In those cases do it again, but show what happened
                output_curves.append(None)
                target_curves.append(None)  # to ensure length is always the same
                try:
                    Mask2Contour(output).get_contour_and_markers(show=True)
                except:  # we expect to to fail
                    pass
        return output_curves, target_curves

    def __call__(self, outputs: dict, targets: dict = None):
        # convert to cpu and detach for metric calculations
        outputs = {ok: self._detach_and_to_cpu(outputs[ok]) for ok in outputs}
        if targets is not None:
            targets = {tk: self._detach_and_to_cpu(targets[tk]) for tk in targets}

        # add curvature if necessary:
        if self.has_curve:
            if targets is None:
                raise NotImplementedError("Need to handle targets being None for curvature")
            outputs["curve"], targets["curve"] = self._get_curvature(outputs["segs"], targets["segs"])
        if self.confidence is not None:
            passes = [self.confidence(o) for o in outputs["segs"]]
            self.failed_confidence += len(passes) - sum(passes)
        else:
            passes = [True] * outputs["segs"].shape[0]
        self.total += outputs["segs"].shape[0]
        batch_res = dict()
        with torch.no_grad():  # disable backprop when evaluating metrics. Useful for adversarial metrics.
            for k, metric_class in self.metrics.items():
                batch_res[k] = metric_class(outputs, targets, passes)
        return batch_res

    def _calc_res(self, method="mean"):
        """ Calculate the results across an epoch """
        for k, metric_class in self.metrics.items():
            self.res[k] = metric_class.reduce(method)

    def epoch_reset(self, phase):
        for k, metric_class in self.metrics.items():
            metric_class.epoch_reset(phase)
        self.failed_confidence = 0
        self.total = 0
        self.phase = phase

    def __repr__(self):
        self._calc_res()
        string = f'metrics ({self.phase}): '
        for k, metric_class in self.metrics.items():
            mn = metric_class.mean()
            absmn = metric_class.absmean()
            md = metric_class.median()
            absmed = metric_class.absmedian()
            std = metric_class.std()
            ciL, ciH = metric_class.ci_95()
            if self.res[k] is None:
                res = "n/a"
            else:
                res = f"med [absmed]: {md:.2g} [{absmed:.2g}], mn [absmn] (std): {mn:.2g} [{absmn:2g}] ({std:.2g}), 95% ci: [{ciL:.2g}, {ciH:.2g}]"
            string += f"\n\t{k} = {res}"
        # string = string[:-2]  # strip last ,
        return string

    def to_dict(self, prefix=None, method="mean", include_best=False, clean=True):
        """
        output current metrics as dictionary
        Args:
            prefix: prefix to append to each metric
            method: how to reduce the metrics across the current epoch results
            include_best: include whether current results are the best
            clean: clean the dictionary of Nones and nans before returning. If true, replaces those values with -.1

        Returns:

        """
        if prefix is None:
            prefix = self.phase
        self._calc_res(method=method)
        if len(prefix) > 0:  # if empty prefix is passed in then don't add _
            metric_dict = {prefix + "_" + k: v for k, v in self.res.items()}
        else:
            metric_dict = {k: v for k, v in self.res.items()}
        if clean:
            metric_dict = clean_dict(metric_dict)
        if include_best:
            metric_dict.update(self.is_best)
        return metric_dict

    def print_best(self):
        for k, metric_class in self.metrics.items():
            print(f"== {k} ==")
            for phase, val in metric_class.best.items():
                if val is not None:
                    print(f"{phase}:\t{val:.4f}")

    def add_to_summary_v2(self, opt):
        fname = os.path.join(opt.checkpoints_dir, "ResultsSummary.xlsx")
        if os.path.exists(fname):
            df = pd.read_excel(fname, "Summary", index_col=0)
        else:
            df = pd.DataFrame()

        res = {
            "Experiment": opt.experiment,
            "Run": opt.name,
            "Dataset Name": os.path.basename,
            "Phase": opt.phase,
            "Restore File": opt.restore_file,
            "Has Confidence": opt.include_confidence,
            "Total": self.total,
            "Failed": self.failed_confidence,
        }
        keep_keys = list(res.keys())
        for method in ["median", "median_absolute_deviation", "mean", "std"]:
            res.update(self.to_dict(prefix=method, method=method))
        keep_keys += ["dice", "convexity", "simplicity", "shape_sim", "cosine"]
        res = {k: v for k, v in res.items() if any([x in k for x in keep_keys])}
        df = df.append(res, ignore_index=True)
        df = df.drop_duplicates(["Experiment", "Run", "Dataset Name", "Phase", "Restore File", "Has Confidence"],
                                keep="last")

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(fname, engine='xlsxwriter')
        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Summary')

        # Get the xlsxwriter objects from the dataframe writer object.
        # workbook = writer.book
        worksheet = writer.sheets["Summary"]

        # add conditional formatting
        excel_col_names = 'BCDEFGHIJKLMNOPQRSTUVWXZ'

        num_rows = df.shape[0]
        for col_name, xcel_name in zip(df.columns, excel_col_names[:len(df.columns)]):
            if not any([x in col_name for x in ["Name", "Total"]]):  # don't format the exp_name column
                neg_col = any([x in col_name for x in ["loss", "hausdorff", "std"]])
                parameters = dict(
                    type="3_color_scale",
                    min_color="green" if neg_col else "red",
                    mid_color="yellow",
                    max_color="red" if neg_col else "green",
                )
                worksheet.conditional_format(f"{xcel_name}2:{xcel_name}{num_rows + 1}", parameters)
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

# def save_sheet(df, writer, sheet_name):
