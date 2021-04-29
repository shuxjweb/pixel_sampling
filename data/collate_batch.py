# encoding: utf-8

import torch


def train_collate_fn(batch):
    imgs, pids, _, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids



def train_collate_gcn_mask(batch):
    imgs, masks, pids, _, pathes = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs), pids, pathes, torch.cat(masks)


def val_collate_gcn_mask(batch):
    imgs, masks, pids, camids, paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, paths, torch.cat(masks, dim=0)


######################################################




def train_collate_fn_path(batch):
    imgs, pids, _, pathes = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, pathes


def train_collate_fn_mem(batch):
    imgs, pids, camids, pathes = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, pathes

def train_collate_fn_idx(batch):
    imgs, pids, _, _, idx = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, idx

def train_collate_fn_adv(batch):
    imgs, pids, _, paths, labels = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    labels = torch.tensor(labels, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, paths, labels

def train_collate_fn_dom(batch):
    imgs, pids, _, pathes, ll = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    ll = torch.tensor(ll, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, pathes, ll

def train_collate_fn_vl(batch):
    imgs, pids, _, pathes, textes = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, pathes, torch.cat(textes, dim=0)

def val_collate_fn_path(batch):
    imgs, pids, camids, paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, paths

def val_collate_fn_idx(batch):
    imgs, pids, camids, paths, idx = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, paths

def val_collate_fn_vl(batch):
    imgs, pids, camids, paths, textes = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, paths, torch.cat(textes, dim=0)

def train_collate_bg(batch):
    imgs, pids, camids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids



def train_collate_mask(batch):
    imgs, pids, _, _, mask, vis = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, torch.stack(mask, dim=0), torch.stack(vis, dim=0)


def val_collate_mask(batch):
    imgs, pids, camids, _, mask, vis = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, torch.stack(mask, dim=0), torch.stack(vis, dim=0)



def train_collate_build(batch):
    imgs, pids, _, _, mask = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, torch.stack(mask, dim=0)


def val_collate_build(batch):
    imgs, pids, camids, _, mask = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, torch.stack(mask, dim=0)


def train_collate_fn_visual(batch):
    imgs, pids, _, _, mask, path = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, torch.stack(mask, dim=0), path


def val_collate_fn_visual(batch):
    imgs, pids, camids, _, mask, path = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, torch.stack(mask, dim=0), path







def train_collate_gcn_mask_head(batch):
    imgs, masks, pids, _, pathes, img_heads, img_legs = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, pathes, torch.cat(masks, dim=0), torch.stack(img_heads, dim=0), torch.stack(img_legs, dim=0)


def val_collate_gcn_mask_head(batch):
    imgs, masks, pids, camids, paths, img_heads, img_legs = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, paths, torch.cat(masks, dim=0), torch.stack(img_heads, dim=0), torch.stack(img_legs, dim=0)



def train_collate_gcn_mask_seg(batch):
    imgs, masks, pids, _, pathes, box_list, msk_list = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, pathes, torch.cat(masks, dim=0), torch.stack(box_list, dim=0), torch.stack(msk_list, dim=0)


def val_collate_gcn_mask_seg(batch):
    imgs, masks, pids, camids, paths, box_list, msk_list = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, paths, torch.cat(masks, dim=0), torch.stack(box_list, dim=0), torch.stack(msk_list, dim=0)




