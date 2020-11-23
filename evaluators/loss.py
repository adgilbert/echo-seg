import logging
from abc import abstractmethod

import torch
from kornia.utils import one_hot as kornia_one_hot
# Below are imported for access from other files to keep everything in the same place
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.functional import softmax
from .metrics import MetricBase
from .weighted_cross_entropy import WeightedCrossEntropyLoss


class LossBase(MetricBase):
    """
    Base class for all loss functions.

    This is implemented as a subclass of MetricBase so loss metrics can be tracked like other metrics.
    However, note that we overwrite the call method of MetricBase so that results are processed as the entire batch
    rather than one-by-one as is done for other metrics.
    """

    def __init__(self, loss_weight, out_type="segs"):
        # Note: As currently implemented there is no need to subclass nn.Module right now.
        # If so, then the __call__ function would have to be bodified to return the __call__ from nn.Module
        # This would automatically call the forward methods then so that part would not need to change.
        self.weight = loss_weight
        assert out_type in ["segs", "bboxes", "curve"], f"out_type {out_type} must be one of 'segs', 'bboxes', 'curve'"
        super(LossBase, self).__init__(out_type, lower_is_better=True)
        self.scheduler = None

    def process_single(self, output: torch.Tensor, target: torch.Tensor = None):
        pass  # For LossBase, process_single is not called. Implemented here to meet abstractmethod reqs.

    def __call__(self, outputs: dict, targets: dict = None, confidence=None):
        """ In this case we overwrite the __call__ method of MetricBase """
        if self.phase in ["train", "adv_train"] and not outputs[self._type].requires_grad:
            # Avoid calculating results twice. The self.phase should allow it to be called from metrics during
            # evaluation/inference if desired.
            # Seems a little hacky, but should be safe since loss only really matters if output requires grad
            return
        res = self.weight * self.forward(outputs[self._type], targets[self.target_name])
        self.results.append(res.item())
        return res

    @property
    def type(self):
        return self._type

    @abstractmethod
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        # subclasses should implement this

    def state_dict(self):
        return None  # subclasses may override with a custom function to store values to restore

    def load_state_dict(self, state_dict):
        pass  # subclasses may override with a custom function to load values

    def scheduler_step(self):
        # useful in the case that a scheduler is implemented.
        if self.scheduler is not None:
            self.scheduler.step()
        if hasattr(self, "optimizer_D"):
            optD = self.optimizer_D[0]
            lr = optD.param_groups[0]["lr"]
            logging.info(f"adversarial lr = {lr}")


def dice_loss(output: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.

    See :class:`~kornia.losses.DiceLoss` for details.
    """
    if not torch.is_tensor(output):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(output)))

    if not len(output.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                         .format(output.shape))

    if not output.shape[-2:] == target.shape[-2:]:
        raise ValueError("input and target shapes must be the same. Got: {}"
                         .format(output.shape, output.shape))

    if not output.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {}".format(
                output.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = softmax(output, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = kornia_one_hot(
        target, num_classes=output.shape[1],
        device=output.device, dtype=output.dtype)

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(input_soft + target_one_hot, dims)

    dice_score = 2. * intersection / (cardinality + eps)
    return torch.mean(-dice_score + 1.)


class Dice(LossBase):
    """ Implementation of DiceLoss based on kornia dice loss """

    def __init__(self, loss_weight, out_type="segs"):
        super(Dice, self).__init__(loss_weight, out_type)
        self.eps: float = 1e-6

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(target.shape) > 3:
            target = target.squeeze()
            if len(target.shape) == 2:
                target = target.unsqueeze(0)  # in case of batch size of 1
        target = target.type(torch.int64)
        return dice_loss(output, target, self.eps)


class BCEWithLogits(LossBase):
    """ Wrapper around BCEWithLogitsBase"""

    def __init__(self, loss_weight, out_type="segs"):
        super(BCEWithLogits, self).__init__(loss_weight, out_type)
        self.loss = BCEWithLogitsLoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.type(torch.float32)
        return self.loss(output, target)


class CrossEntropy(LossBase):
    """Wrapper around Cross Entropy """

    def __init__(self, loss_weight, out_type="segs", **kwargs):
        super(CrossEntropy, self).__init__(loss_weight, out_type)
        self.loss = CrossEntropyLoss(**kwargs)
        self.has_weight = "weight" in kwargs

    def forward(self, output: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        target_B, target_C, target_H, target_W = target.shape
        # if weight is provided it will be a GPU tensor so have to match that
        if self.has_weight and torch.cuda.is_available():
            return self.loss(output.cuda(), target.view(target_B, target_H, target_W).cuda())
        else:
            return self.loss(output, target.view(target_B, target_H, target_W))


class MSE(LossBase):
    """Wrapper around MSELoss """

    def __init__(self, loss_weight, out_type="bboxes"):
        super(MSE, self).__init__(loss_weight, out_type)
        self.loss = MSELoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(output, target)


class WCE(LossBase):
    """ Wrapper around Weighted Cross Entropy"""

    def __init__(self, loss_weight, out_type="segs"):
        super(WCE, self).__init__(loss_weight, out_type)
        self.loss = WeightedCrossEntropyLoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(output, target)


class Losses:
    """
    Class to handle combine all losses.

    Generally just call the class with inputs and ouputs

    enable(loss_name)/disable(loss_name) can be used to enable/disable loss function components for selective use
    during trianing
    """

    def __init__(self):
        self.loss_dict = dict()
        self.current_vals = dict()
        self.current_losses = list()

    def add_loss(self, name: str, loss_fcn: LossBase):
        self.loss_dict[name] = loss_fcn
        self.current_losses.append(name)

    def enable(self, loss_name: str):
        assert loss_name in self.loss_dict, f"{loss_name} not in loss_dict ({self.loss_dict.keys()} use add_loss()"
        if loss_name not in self.current_losses:
            self.current_losses.append(loss_name)

    def disable(self, loss_name: str):
        assert loss_name in self.loss_dict, f"{loss_name} not in loss_dict ({self.loss_dict.keys()} use add_loss()"
        if loss_name in self.current_losses:
            self.current_losses.remove(loss_name)

    def disable_all(self):
        self.current_losses = list()

    def enable_only(self, loss_name: str):
        self.disable_all()
        self.enable(loss_name)

    def enable_all(self):
        self.current_losses = list(self.loss_dict.keys())

    def set_loss_target(self, loss_name: str, target_name: str):
        """ used to set the key in the labels dict for the loss function to use as a target """
        assert loss_name in self.loss_dict, f"{loss_name} not found in Losses ({list(self.loss_dict.keys())}"
        self.loss_dict[loss_name].set_target(target_name)

    def reset_targets(self):
        for loss in self.loss_dict.values():
            loss.reset_target()

    def _sum(self):
        return sum([cv for cv in self.current_vals.values()])

    def __call__(self, outputs, targets):
        self.current_vals = dict()
        for name in self.current_losses:
            self.current_vals[name] = self.loss_dict[name](outputs, targets)
        return self._sum()

    def __repr__(self):
        string = "Loss: "
        for k, v in self.current_vals.values():
            string += f"{k}:{v} + "
        string = string[:-2]  # strip last
        return string

    def to_dict(self):
        d = self.current_vals.copy()
        for loss in self.loss_dict:
            if loss not in d:
                d[loss] = torch.tensor(-.1)  # May fix visdom plotting error
        return d

    def state_dict(self):
        """ state dictionary that can be saved for later restoring """
        state_dict = {n: l.state_dict() for n, l in self.loss_dict.items()}
        return state_dict

    def load_state_dict(self, state_dict):
        for n, sd in state_dict.items():
            if n in self.loss_dict:
                self.loss_dict[n].load_state_dict(sd)
            else:
                logging.warning(f"not loading state_dict for {n} because not found in current losses")

    def scheduler_step(self):
        for loss in self.loss_dict.values():
            loss.scheduler_step()  # update any parameters if necessary
