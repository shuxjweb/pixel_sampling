# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from .transforms import RandomErasing, RandomSwap
import collections
import sys
if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


def build_transforms(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.RandomHorizontalFlip(),
            T.Pad(10),
            T.RandomCrop([cfg.height, cfg.width]),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.5, mean=PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform
        ])

    return transform



def build_transforms_head(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.Pad(10),
            T.RandomCrop([cfg.height, cfg.width]),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.5, mean=PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform
        ])

    return transform


################################################################


def build_transforms_base(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.Pad(10),
            T.RandomCrop([cfg.height, cfg.width]),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToTensor(),
            normalize_transform,
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform
        ])

    return transform



def build_transforms_hist(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    transform = T.Compose([
        T.Resize([cfg.height, cfg.width]),
        T.ToTensor(),
    ])

    return transform


def build_transforms_eraser(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.Pad(10),
            T.RandomCrop([cfg.height, cfg.width]),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToTensor(),
            normalize_transform,
            # RandomErasing(probability=0.5, mean=PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform
        ])

    return transform

def build_transforms_visual(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    transform = T.Compose([
        T.Resize([cfg.height, cfg.width]),
        T.ToTensor(),
    ])

    return transform

# def build_transforms_swap(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
#     normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
#
#     if is_train:
#         transform = [T.Compose([
#             T.Resize([cfg.height, cfg.width]),
#             T.RandomHorizontalFlip(p=0.5),
#             T.Pad(10),
#             T.RandomCrop([cfg.height, cfg.width]),
#             T.ToTensor(),
#             normalize_transform
#         ]), RandomSwap(probability=0.5, mean=PIXEL_MEAN)]
#     else:
#         transform = [T.Compose([
#             T.Resize([cfg.height, cfg.width]),
#             T.ToTensor(),
#             normalize_transform
#         ])]
#
#     return transform

def build_transforms_swap(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform,
            RandomSwap(probability=0.5, mean=PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform
        ])

    return transform


def build_transforms_open(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.5, mean=PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform
        ])

    return transform




def build_transforms_usize(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Scale(cfg.height),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.5, mean=PIXEL_MEAN)
        ])
    else:
        if cfg.train == 'train':
            transform = T.Compose([
                T.Scale(cfg.height),
                T.CenterCrop([cfg.height, cfg.width]),
                T.ToTensor(),
                normalize_transform
            ])
        else:
            transform = T.Compose([
                T.ToTensor(),
                normalize_transform
            ])

    return transform





class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)



def build_transforms_no_erase(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform
        ])

    return transform


def build_transforms_visual(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225], use_eraser=False):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    if use_eraser:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            RandomErasing(probability=1, mean=PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            # normalize_transform
        ])

    return transform





def build_transforms_bap(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225], rate=0.875):
    if is_train:
        transform = T.Compose([
            T.Resize([int(cfg.height//rate), int(cfg.width//rate)]),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=32. / 255., saturation=0.5),
            T.Pad(10),
            T.RandomCrop([cfg.height, cfg.width]),
            T.ToTensor(),
            T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
            # RandomErasing(probability=0.5, mean=PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
        ])

    return transform



