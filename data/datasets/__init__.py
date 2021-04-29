# encoding: utf-8
from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .market_triplet import MarketTriplet
from .msmt17 import MSMT17
from .prcc import PRCC
from .prcc_gcn import PRCC_GCN
from .celeba import CELEBA
from .celeba_msk import CELEBA_MSK
from .dataset_loader import ImageDataset, ImageDatasetMask, ImageDatasetPath, ImageDatasetGcnMask
from .dataset_loader import ImageDatasetVisualMask

__factory = {
    'market1501': Market1501,
    'market_triplet': MarketTriplet,
    'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'prcc': PRCC,
    'prcc_gcn': PRCC_GCN,
    'celeba': CELEBA,
    'celeba_msk': CELEBA_MSK,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
