# absolute import fails from JupyterNotebooks
import logging
import torch
import functools
import torch.nn as nn
import torch.optim as optim

from torch.nn import init
from torch.optim import lr_scheduler
from .resnet import ResnetGenerator
from .unet import UnetGenerator


def get_network(input_nc, output_nc, filters, model, norm='batch', use_dropout=False, init_type='normal',
                init_gain=0.02, gpu_ids=[], crop_size=256, include_bbox=False, include_pseudo=False, coord_conv=False,
                *args, **kwargs):
    """Create a network

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        filters (int) -- the number of filters in the last conv layer
        model (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        crop_size (int) -- crop_size for the network (necessary for UNet skip connections)
        include_bbox (bool) -- whether to also output a bounding box for the prediction
        include_pseudo (bool) -- whether the pseudo image will be appended as a channel at the input
        coord_conv (bool) -- whether the network should use coord_conv at the input
        *args, **kwargs -- these must be included so that get_network can be called with get_network(vars(**opt)) from
        the train/test options.


    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).

    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    if include_bbox and "resnet" not in model:
        raise NotImplementedError("include bbox not implemented for networks other than resnet")
    norm_layer = get_norm_layer(norm_type=norm)
    input_nc = input_nc + include_pseudo  # add a channel for pseudo

    if model == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, filters, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                              crop_size=crop_size, include_bbox=include_bbox)
    elif model == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, filters, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
                              crop_size=crop_size, include_bbox=include_bbox)
    elif model == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, filters, norm_layer=norm_layer, use_dropout=use_dropout,
                            include_coordconv=coord_conv)
    elif model == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, filters, norm_layer=norm_layer, use_dropout=use_dropout,
                            include_coordconv=coord_conv)
    else:
        raise NotImplementedError(f'Generator model name {model} is not recognized')
    return init_net(net, init_type, init_gain, gpu_ids)


def collect_scheduler(opt, optimizer):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy is None or opt.lr_policy == "":
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda x: 1.0)
    # if opt.lr_policy == 'linear':
    #     def lambda_rule(epoch):
    #         lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
    #         return lr_l
    #     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        logging.info("scheduler is a step scheduler with step_size = 10, gamma=.2")
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    elif opt.lr_policy == "single_step_epoch10":
        logging.info("scheduler has a single step at epoch 10 with gamma = 0.1")
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    # elif opt.lr_policy == 'plateau':
    #     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    # elif opt.lr_policy == 'cosine':
    #     scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        raise NotImplementedError(f'learning rate policy {opt.lr_policy} is not implemented')
    return scheduler


def collect_optimizer(opt, model):
    if "weight_decay" in opt.loss:
        if opt.weights is None:
            decay_weight = 0.01  # https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab
        else:
            decay_weight = opt.weights[opt.loss.index("weight_decay")]
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=decay_weight)
    else:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    return optimizer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "glorot":
                init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=()):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
