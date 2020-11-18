import logging
from abc import abstractmethod

import torch
from kornia.utils import one_hot as kornia_one_hot
from torch import nn
# Below are imported for access from other files to keep everything in the same place
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.functional import softmax, one_hot

from segmentation.UltrasoundSegmentation.models.discriminator import NLayerDiscriminator
from segmentation.UltrasoundSegmentation.models.net import init_net, get_norm_layer
from transformation.CycleGAN_and_pix2pix.models.networks import GANLoss
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


class Pix2PixNoInputOtherLabels(LossBase):
    """
    A class just like Pix2Pix loss except for it does not use an input image (so discriminator only
    learns from labels) and also the real labels can be derived from a different source.
    """

    def __init__(self, loss_weight, opt, out_type="segs"):
        super(Pix2PixNoInputOtherLabels, self).__init__(loss_weight, out_type)
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.separate_discs = opt.separate_discs
        if self.separate_discs:
            self.netD = [self.define_D(1, 64, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)
                         for _ in range(opt.output_nc)]

        else:
            self.netD = [self.define_D(opt.output_nc, 64, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)]
        if opt.isTrain:
            self.optimizer_D = [torch.optim.Adam(nD.parameters(), lr=opt.adv_lr, betas=(opt.beta1, 0.999))
                                for nD in self.netD]
        self.criterionGAN = GANLoss("lsgan").to(self.device)
        self.weight = loss_weight
        self.loss_D = torch.tensor(-.1)  # watch discriminator loss as well
        self.isTrain = opt.isTrain
        self.sigmoid = nn.Sigmoid()
        self.num_classes = opt.output_nc
        if opt.adv_lr_policy == "step_higher":
            if opt.num_epochs > 20:
                raise NotImplementedError("scheduler currently not configured")
                # logging.info("configuring Step Higher for discriminator optimizer. lr=lrx2 at epoch 10 and 20")
                # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer_D, milestones=[10, 20], gamma=2.)
            else:
                logging.warning("adv_lr_policy step only configured for >20 epochs. Not using.")

    @staticmethod
    def define_D(input_nc, ndf, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=()):
        norm_layer = get_norm_layer(norm_type=norm)
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
        return init_net(net, init_type, init_gain, gpu_ids)

    def backward_D(self, outputs, real_AB, netD):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = self.sigmoid(outputs)  # only outputs
        pred_fake = netD(fake_AB.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = real_AB.type(torch.float)  # cast to float to match output from network
        pred_real = netD(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (loss_D_fake + loss_D_real) * 0.5
        self.loss_D *= self.weight  # need to multiple loss_D times weight here since it won't be outside.
        # if outputs.requires_grad and self.isTrain:  # else just calculating for metrics
        self.loss_D.backward()

    def backward_G(self, outputs, netD):
        """Calculate GAN and L1 loss for the generator"""
        # G(A) should fake the discriminator
        fake_AB = self.sigmoid(outputs)  # outputs  # torch.cat((inputs, outputs), 1)
        pred_fake = netD(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        return loss_G_GAN  # actual G update is performed in train.py

    def _forward(self, output: torch.Tensor, target: torch.Tensor, netD, optimizer_D) -> torch.Tensor:
        """ separate out forward call for case of multiple discriminators"""
        # update D
        # when this function is called from metrics requires_grad will be false, and self.loss_D will be set already
        # so no need to redo the update D step
        if output.requires_grad and self.isTrain:
            self.set_requires_grad(netD, True)  # enable backprop for D
            optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D(output, target, netD)  # calculate gradients for D
            # if output.requires_grad and self.isTrain:
            optimizer_D.step()  # update D's weights

        # update G
        self.set_requires_grad(netD, False)  # D requires no gradients when optimizing G
        return self.backward_G(output, netD)  # no direct targets, just trying to full discriminator

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # need one hot here to match the fake outputs - permute gets dimensions right from one_hot output
        if self.num_classes > 1:
            target = one_hot(target, num_classes=self.num_classes).squeeze(1).permute((0, 3, 1, 2))
        if self.separate_discs:
            loss = 0
            for c in range(self.num_classes):
                o = output[:, c, :, :].unsqueeze(1)
                t = target[:, c, :, :].unsqueeze(1)
                loss += self._forward(o, t, self.netD[c], self.optimizer_D[c])
            return loss / self.num_classes
        else:
            return self._forward(output, target, self.netD[0], self.optimizer_D[0])

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def state_dict(self):
        """ Get params to save """
        return dict(
            model_dict=[nD.state_dict() for nD in self.netD],
            optim_dict=[oD.state_dict() for oD in self.optimizer_D],
        )

    def load_state_dict(self, state_dict):
        """ Reload params """
        for nD, oD, model_params, optim_params in zip(self.netD, self.optimizer_D, state_dict["model_dict"],
                                                      state_dict["optim_dict"]):
            nD.load_state_dict(model_params)
            oD.load_state_dict(optim_params)


class P2PDiscLossWrapper(MetricBase):
    """
    A helper class to provide access to the discriminative loss of Pix2PixNoInputOtherLabels
    in metrics.
    """

    def __init__(self, p2p_loss: Pix2PixNoInputOtherLabels):
        super(P2PDiscLossWrapper, self).__init__(lower_is_better=True)
        self.p2p_loss = p2p_loss

    def process_single(self, output: torch.Tensor, target: torch.Tensor):
        pass  # not necessary here since we overwrite call function

    def __call__(self, outputs: dict, targets: dict, confidence=None):
        # Relies on p2p loss being already called and updating it's own loss
        # If this function is called before p2p_loss is called we will get the loss from
        # the previous call. However, that case shouldn't matter much for the purpose of metrics.
        self.results.append(self.p2p_loss.loss_D.item())


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
