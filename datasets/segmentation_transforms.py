import logging
import random

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms


def show_img(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


class SegResize:
    """ special resize for segmentation dataset """

    def __init__(self, new_size):
        self.new_size = (new_size, new_size)
        self.image_transform = transforms.Resize(self.new_size, Image.BICUBIC)
        self.label_transform = transforms.Resize(self.new_size, Image.NEAREST)  # nearest for label

    def __call__(self, **kwargs):
        for k, im in kwargs.items():
            if k == "labels":
                t = self.label_transform
            else:
                t = self.image_transform
            kwargs[k] = t(im)
        return kwargs

    def __repr__(self):
        return f"SegResize({self.new_size})"


class SegRandomAffine(transforms.RandomAffine):
    """ special version of random affine for segmentation dataset
    Need all the same parameters as regular random affine:
    degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0
    https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomAffine
    """

    def __init__(self, *args, **kwargs):
        transforms.RandomAffine.__init__(self, *args, **kwargs)

    def __call__(self, **kwargs):
        im = kwargs["images"] if "images" in kwargs else kwargs["labels"]
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, im.size)
        for k, im in kwargs.items():
            if k == "labels":
                resample = Image.NEAREST
            else:
                resample = Image.BILINEAR
            kwargs[k] = F.affine(im, *ret, resample=resample, fillcolor=self.fillcolor)
        return kwargs

    def __repr__(self):
        return f"SegRandomAffine(deg={self.degrees}), trans={self.translate}, scale={self.scale})"


class SegRandomCrop(transforms.RandomCrop):
    """ Special version of random crop for segmentation dataset
    same parameters as regular random crop:
    size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'
    https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomCrop
    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        transforms.RandomCrop.__init__(self, size, padding, pad_if_needed, fill, padding_mode)

    def handle_padding(self, img):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

            # pad the width if needed note that img.size and self.size use opposite conventions for h, w
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
        return img

    def __call__(self, **kwargs):

        crop_params = None
        for k, im in kwargs.items():
            im = self.handle_padding(im)
            if crop_params is None:  # only define once for all images
                crop_params = self.get_params(im, self.size)
            kwargs[k] = F.crop(im, *crop_params)
        return kwargs

    def __repr__(self):
        return f"SegRandomCrop(size={self.size})"


class SegCenterCrop:
    """ Special version of center crop for segmentation dataset
    same parameters as regular center crop: size
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, **kwargs):
        for k, im in kwargs.items():
            kwargs[k] = F.center_crop(im, self.size)
        return kwargs

    def __repr__(self):
        return f"SegCenterCrop(size={self.size})"


class SegNormalize:
    """ copy of normalize transform for segmentation dataset. Will only normalize the image not the label
    same params: mean, std, inplace=False
    https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#Normalize
    SHOULD BE CALLED AFTER TO_TENSOR  because expects input of C, H, W
    """

    def __init__(self, mean_and_std: dict):
        """
        :param mean_and_std: should two level dict with first level being image name (i.e. images, pseudos) and second
        level "mean" and "std"
        """
        self.mean_and_std = mean_and_std

    def __call__(self, **kwargs):
        for k, im in kwargs.items():
            if k in self.mean_and_std and k != "labels":
                mean = (self.mean_and_std[k]["mean"],)
                std = (self.mean_and_std[k]["std"],)
                kwargs[k] = F.normalize(kwargs[k], mean, std, False)
        return kwargs

    def __repr__(self):
        mean_std_str = " - ".join(
            [f"{k}: mean={self.mean_and_std[k]['mean']} std={self.mean_and_std[k]['std']}" for k in self.mean_and_std if
             k != "labels"])
        return f"SegNormalize({mean_std_str})"


class SegToTensor(transforms.ToTensor):
    def __init__(self, num_classes):
        """
        :param num_classes: number of classes expected in each image
        """
        self.num_classes = num_classes

    def __call__(self, **kwargs):
        for k, im in kwargs.items():
            kwargs[k] = F.to_tensor(kwargs[k])
            if k == "labels":
                if self.num_classes > 0:
                    kwargs[k] *= self.num_classes  # need to put in range [0, C - 1] before casting to int
                kwargs[k] = kwargs[k].type(torch.LongTensor)
        return kwargs

    def __repr__(self):
        return "SegToTensor"


class SegGrayscale:
    """ copy of grayscale transform. Will only be applied to image """

    def __call__(self, **kwargs):
        # assert "images" in kwargs, f"{self} only works on images"
        if "images" in kwargs:
            kwargs["images"] = F.to_grayscale(kwargs["images"], 1)
        return kwargs

    def __repr__(self):
        return "SegGrayscale"


class SegGamma:
    """ randomly adjust gamma of input. Only applied to image"""

    def __init__(self, gamma_range: tuple):
        assert len(gamma_range) == 2, f"gamma should be of format (gamm_min, gamma_max) but was {gamma_range}"
        self.gamma_range = gamma_range

    def __call__(self, **kwargs):
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        for k, im in kwargs.items():
            if k != "labels":
                kwargs[k] = F.adjust_gamma(kwargs[k], gamma)
        return kwargs

    def __repr__(self):
        return f"SegGamma({self.gamma_range})"


class SegCompose(transforms.Compose):
    """ copy of compose for segmentation dataset"""

    def __init__(self, input_transforms):
        transforms.Compose.__init__(self, input_transforms)

    def __call__(self, **kwargs):
        for t in self.transforms:
            try:
                kwargs = t(**kwargs)
            except TypeError as e:
                raise TypeError(f"err for transform {str(t)}: {e}")
        return kwargs

    def __repr__(self):
        string = "sequence of:\n\t"
        string += "\n\t".join([str(t) for t in self.transforms])
        return string


def get_transform(opt, grayscale=True, convert=True, mean_and_std=None, is_train=True):
    transform_list = []
    if grayscale:
        transform_list.append(SegGrayscale())
    if 'resize' in opt.preprocess:
        transform_list.append(SegResize(opt.load_size))

    if 'crop' in opt.preprocess:
        if is_train:
            transform_list.append(SegRandomCrop(opt.crop_size))
        else:
            transform_list.append(SegCenterCrop(opt.crop_size))

    if 'affine' in opt.preprocess and is_train:
        # TODO: parametrize these somewhere
        transform_list.append(SegRandomAffine(10, translate=(.1, .1), scale=(.9, 1.1), shear=None))

    if "gamma" in opt.preprocess and is_train:
        transform_list.append(SegGamma((.4, 1.2)))

    if convert:
        transform_list += [SegToTensor(opt.output_nc - 1)]

    if "norm" in opt.preprocess:
        assert grayscale, "implement non greyscale for SegNormalize"
        transform_list += [SegNormalize(mean_and_std)]
    final_t = SegCompose(transform_list)
    logging.info(f"Transform is {str(final_t)}")
    return final_t
