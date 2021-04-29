# encoding: utf-8

import os
import torch
import numpy as np
from utils.reid_metric import R1_mAP
import shutil
import torch.nn.functional as F
import json
import errno
import cv2


def fliplr(img):
    # flip horizontal
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.cpu().index_select(3, inv_idx)
    return img_flip.cuda()


def norm(f):
    # f = f.squeeze()
    fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
    f = f.div(fnorm.expand_as(f))
    return f


def mkdir_if_missing(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(os.path.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def inference_prcc_global(model, test_loader, num_query, use_flip=True):
    print('Test')
    model.eval()
    metric = R1_mAP(num_query, 500)
    with torch.no_grad():
        for ii, batch in enumerate(test_loader):
            data, pid, cmp, fnames, mask = batch                          # [128, 3, 256, 128]
            data = data.to("cuda") if torch.cuda.device_count() >= 1 else data
            b, c, h, w = data.shape

            feat = model(data)

            if use_flip:
                data_f = fliplr(data.cpu()).cuda()              # [128, 3, 256, 128]
                feat_f = model(data_f)
                feat += feat_f

            feat = F.normalize(feat, p=2, dim=-1)

            metric.update([feat, pid, cmp])
    cmc, mAP = metric.compute()
    return mAP, cmc[0]



def inference_prcc_visual_rank(model, test_loader, num_query, home, show_rank=20, use_flip=True, max_rank=50):
    print('Test')

    if not os.path.exists(home):
        os.makedirs(home)

    model.eval()
    feats = []
    pids = []
    camids = []
    fnames_all = []
    num_total = len(test_loader)
    with torch.no_grad():
        for ii, batch in enumerate(test_loader):
            data, pid, cmp, fnames, mask = batch                          # [128, 3, 256, 128]
            data = data.to("cuda") if torch.cuda.device_count() >= 1 else data
            b, c, h, w = data.shape

            feat = model(data)                              # [64, 4*2048]

            if use_flip:
                data_f = fliplr(data.cpu()).cuda()              # [128, 3, 256, 128]
                feat_f = model(data_f)
                feat += feat_f
            feat = F.normalize(feat, p=2, dim=-1)

            feats.append(feat)
            pids.extend(pid)
            camids.extend(cmp)
            fnames_all.extend(fnames)

            if ii % 5 == 0:
                print(num_total, ii+1)



    feats = torch.cat(feats, dim=0)  # [6927, 2048]
    feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # [6927, 2048]
    pids = np.array(pids)
    camids = np.array(camids)
    fnames_all = np.array(fnames_all)

    # query
    qf = feats[:num_query]  # [3368, 2048]
    q_pids = np.asarray(pids[:num_query])  # [3368,]
    q_camids = np.asarray(camids[:num_query])  # [3368,]
    q_fnames = fnames_all[:num_query]
    # gallery
    gf = feats[num_query:]  # [15913, 2048]
    g_pids = np.asarray(pids[num_query:])  # [15913,]
    g_camids = np.asarray(camids[num_query:])  # [15913,]
    g_fnames = fnames_all[num_query:]

    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat = distmat - 2 * torch.matmul(qf, gf.t())
    distmat = distmat.cpu().numpy()

    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)  # [3368, 15913]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)  # [3368, 15913]

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):  # 3368
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_fname = q_fnames[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)  # [15913,]
        keep = np.invert(remove)  # [15913,]

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]  # [15908,]

        if not np.any(orig_cmc):
            continue

        g_fnames_s = g_fnames[order][keep][0: show_rank]
        g_pids_s = g_pids[order][keep][0: show_rank]

        # copy query
        src = q_fname
        q_fname_list = q_fname.split('/')
        name_dir = q_fname_list[-3] + '_' + q_fname_list[-2] + '_' + q_fname_list[-1]
        path_d = os.path.join(home, name_dir)
        if not os.path.exists(path_d):
            os.makedirs(path_d)
        dst = os.path.join(path_d, name_dir)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

        # copy gallery
        path_g = os.path.join(path_d, 'gallery')
        if not os.path.exists(path_g):
            os.makedirs(path_g)

        for kk, g_fname in enumerate(g_fnames_s):
            src = g_fname
            g_fname_list = g_fname.split('/')
            if q_pid == g_pids_s[kk]:
                name_dir = str(kk + 1).zfill(3) + '_' + 'right' + '_' + g_fname_list[-3] + '_' + g_fname_list[-2] + '_' + g_fname_list[-1]
            else:
                name_dir = str(kk + 1).zfill(3) + '_' + 'wrong' + '_' + g_fname_list[-3] + '_' + g_fname_list[-2] + '_' + g_fname_list[-1]

            dst = os.path.join(path_g, name_dir)
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)



def inference_prcc_pcb(model, test_loader, num_query, use_flip=True):
    print('Test')
    model.eval()
    metric = R1_mAP(num_query, 500)
    with torch.no_grad():
        for ii, batch in enumerate(test_loader):
            data, pid, cmp, fnames, masks = batch                          # [128, 3, 256, 128]
            data = data.to("cuda") if torch.cuda.device_count() >= 1 else data
            b, c, h, w = data.shape

            feat = model(data)                              # [64, 4*2048]

            if use_flip:
                data_f = fliplr(data.cpu()).cuda()              # [128, 3, 256, 128]
                feat_f = model(data_f)
                feat += feat_f

                # fnorm = torch.norm(feat, p=2, dim=1, keepdim=True) * np.sqrt(6)  # [128, 1, 4]
                # feat = feat.div(fnorm.expand_as(feat))  # [64, 2048, 4]

                feat = F.normalize(feat, p=2, dim=-1)

            metric.update([feat, pid, cmp])
    cmc, mAP = metric.compute()
    return mAP, cmc[0]




def norm_mask(img):
    v_min = img.min()
    v_max = img.max()
    img = (img - v_min) / (abs(v_max - v_min) + 1e-5)
    return img


def show_cam_on_image(img, mask, ratio=0.5):           # [256, 128, 3], [256, 128], img->min=0, max=1, mask->min=0, max=0.99
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)     # [256, 128, 3]
    heatmap = np.float32(heatmap) / 255     # [256, 128, 3]
    # cam = 0.5*heatmap + 0.5*np.float32(img)         # [256, 128, 3]
    cam = ratio * heatmap + (1 - ratio) * np.float32(img)  # [256, 128, 3]
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)               # [256, 128, 3]
    return cam




