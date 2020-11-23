import torch

from .loss import Losses, Dice, BCEWithLogits, MSE, CrossEntropy, WCE
from .metrics import IoU, Metrics, Curvature, Hausdorff, DiceScore, Simplicity, Convexity, CurvatureIndividual, \
    CosineSim, SliceMerger, Bias, SurfaceDist


def collect_losses(opt):
    loss_fcn = Losses()
    # Only way to correctly parameterize is to manually parse
    if opt.weights is not None:
        assert len(opt.loss) == len(opt.weights), f"length of weights ({opt.weights}) must match losses ({opt.loss})"
        weights = opt.weights
    else:
        weights = [1] * len(opt.loss)
    for loss, loss_weight in zip(opt.loss, weights):
        if loss == "dice":
            loss_fcn.add_loss(loss, Dice(loss_weight, out_type="segs"))
        elif loss == "vanilla":
            loss_fcn.add_loss(loss, CrossEntropy(loss_weight, out_type="segs"))
        elif loss == "weighted_vanilla":
            if opt.output_nc == 4:
                weight = torch.Tensor([.2, 1., 1., 2.])
            elif opt.output_nc == 2:
                weight = torch.Tensor([.2, 1.])
            else:
                raise ValueError(f"weighted vanilla loss selected but no weights set for output_nc = {opt.output_nc}")
            weight = weight.cuda() if torch.cuda.is_available() else weight
            loss_fcn.add_loss(loss, CrossEntropy(loss_weight, out_type="segs", weight=weight))
        elif loss == "ignore_vanilla":
            loss_fcn.add_loss(loss, CrossEntropy(loss_weight, out_type="segs", ignore_index=0))
        elif loss == "wce":
            loss_fcn.add_loss(loss, WCE(loss_weight, out_type="segs"))
        elif loss == "bce":
            loss_fcn.add_loss(loss, BCEWithLogits(loss_weight, out_type="segs"))
        elif loss == "bbox":
            loss_fcn.add_loss(loss, MSE(loss_weight, out_type="bboxes"))
        elif loss == "weight_decay":
            pass  # this is handled in the optimizer initialization (above) so nothing needed here
        # Add others here as needed
        else:
            raise ValueError(f"Loss {loss} not recognized.")
    return loss_fcn


def add_losses_to_metrics(losses: loss.Losses):
    """ adds an element of loss"""
    metric_dict = dict()
    for name, loss in losses.loss_dict.items():
        metric_dict["loss_" + name] = loss  # LossWrapper(loss)
    return metric_dict


def collect_metrics(opt, losses: loss.Losses = None, confidence=None):
    """ collect the metrics to be used for evaluation. Will also add losses as metrics if provided"""
    metric_dict = dict()
    use_all = "all" in opt.metrics
    if use_all or "iou" in opt.metrics:
        metric_dict["iou"] = IoU(num_classes=opt.output_nc)
    if use_all or "dice" in opt.metrics:
        metric_dict["dice"] = DiceScore()
        if opt.output_nc == 4:
            metric_dict["lv_endo_dice"] = DiceScore(include_classes=[1])
            # LV EPI is calculated from blood pool plus myocardium
            metric_dict["lv_epi_dice"] = DiceScore(include_classes=[1], modifier=SliceMerger((1, 2)))
            metric_dict["la_dice"] = DiceScore(include_classes=[3])
    if use_all or "bias" in opt.metrics:
        metric_dict["bias"] = Bias()
        if opt.output_nc == 4:
            metric_dict["lv_endo_bias"] = Bias(include_classes=[1])
            # LV EPI is calculated from blood pool plus myocardium
            metric_dict["lv_epi_bias"] = Bias(include_classes=[1], modifier=SliceMerger((1, 2)))
            metric_dict["la_bias"] = Bias(include_classes=[3])
    if use_all or "dist" in opt.metrics:
        metric_dict["lv_endo_dist"] = SurfaceDist(label_val=1)
    if use_all or "curvature" in opt.metrics:
        metric_dict["curvature"] = Curvature()
        for segment_name in ['basal', 'mid', 'apical']:
            for side in [1, 2]:
                metric_dict[f"curve_{segment_name}{str(side)}"] = CurvatureIndividual(segment_name, side)
    if use_all or "hausdorff" in opt.metrics:
        metric_dict["hausdorff"] = Hausdorff()
    if use_all or "simplicity" in opt.metrics:
        metric_dict["lv_simplicity"] = Simplicity(label_val=1)
        if opt.output_nc == 4:
            metric_dict["lv_epi_simplicity"] = Simplicity(label_val=1, modifier=SliceMerger((1, 2)))
            metric_dict["la_simplicity"] = Simplicity(label_val=3)
    if use_all or "convexity" in opt.metrics:
        metric_dict["lv_convexity"] = Convexity(label_val=1)
        if opt.output_nc == 4:
            metric_dict["lv_epi_convexity"] = Convexity(label_val=1, modifier=SliceMerger((1, 2)))
            metric_dict["la_convexity"] = Convexity(label_val=3)
    if use_all or "cosine" in opt.metrics:
        # metric_dict["cosine"] = CosineSim()
        metric_dict["lv_cosine_6"] = CosineSim(label_val=1, nsegs=6)
        if opt.output_nc == 4:
            metric_dict["lv_epi_cosine_6"] = CosineSim(label_val=1, nsegs=6, modifier=SliceMerger((1, 2)))
            metric_dict["la_cosine_6"] = CosineSim(label_val=3, nsegs=6)
    if losses is not None and opt.isTrain:
        metric_dict.update(add_losses_to_metrics(losses))
    return Metrics(metric_dict, opt.phase, confidence)
