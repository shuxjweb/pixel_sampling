import numpy as np
import torch

from data.datasets.eval_reid import eval_func


def get_map(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):




    eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50)






















