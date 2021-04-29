# encoding: utf-8

from .pcb import pcb_p6
from .pcb_seg import pcb_global
from .hpm import HPM
from .mgn import MGN


def build_model(num_classes=None, model_type='base'):
    if model_type == 'global':
        model = pcb_global(num_classes)
    if model_type == 'pcb':
        model = pcb_p6(num_classes)
    elif model_type == 'hpm':
        model = HPM(num_classes)
    elif model_type == 'mgn':
        model = MGN(num_classes)
    else:
        pass
    return model


