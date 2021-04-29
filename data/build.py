# encoding: utf-8

import torch
from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn, train_collate_mask, val_collate_mask, train_collate_fn_path, train_collate_gcn_mask
from .collate_batch import val_collate_fn_path, val_collate_gcn_mask
from .datasets import init_dataset, ImageDataset, ImageDatasetMask, ImageDatasetGcnMask, ImageDatasetVisualMask
from .samplers import RandomIdentitySampler, RandomIdentitySamplerGcn
from .transforms import build_transforms, build_transforms_visual, build_transforms_head
from .transforms import build_transforms_eraser,  build_transforms_base
import numpy as np
import random





def make_data_loader_prcc_hrnet(cfg, h=64, w=32, use_eraser=False):
    if use_eraser:
        train_transforms = build_transforms_head(cfg, is_train=True)
        val_transforms = build_transforms_head(cfg, is_train=False)
    else:
        train_transforms = build_transforms_base(cfg, is_train=True)
        val_transforms = build_transforms_base(cfg, is_train=False)

    dataset = init_dataset(cfg.dataset, root=cfg.data_dir)
    num_classes = dataset.num_train_pids    # 150
    num_workers = cfg.num_works             # 8


    train_loader = DataLoader(
        ImageDatasetGcnMask(dataset.train, h, w, train_transforms), batch_size=cfg.batch_size,
        sampler=RandomIdentitySamplerGcn(dataset.train, cfg.batch_size, cfg.img_per_id),  # 64, 4
        num_workers=num_workers, collate_fn=train_collate_gcn_mask
    )

    train_loader_ca = DataLoader(
        ImageDatasetGcnMask(dataset.train_ca, h, w, train_transforms), batch_size=cfg.batch_size,
        sampler=RandomIdentitySamplerGcn(dataset.train, cfg.batch_size, cfg.img_per_id),  # 64, 4
        num_workers=num_workers, collate_fn=train_collate_gcn_mask
    )

    train_loader_cb = DataLoader(
        ImageDatasetGcnMask(dataset.train_cb, h, w, train_transforms), batch_size=cfg.batch_size,
        sampler=RandomIdentitySamplerGcn(dataset.train, cfg.batch_size, cfg.img_per_id),  # 64, 4
        num_workers=num_workers, collate_fn=train_collate_gcn_mask
    )

    val_set_c = ImageDatasetGcnMask(dataset.query_c + dataset.gallery, h, w, val_transforms)
    val_loader_c = DataLoader(
        val_set_c, batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_gcn_mask
    )

    val_set_b = ImageDatasetGcnMask(dataset.query_b + dataset.gallery, h, w, val_transforms)
    val_loader_b = DataLoader(
        val_set_b, batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_gcn_mask
    )


    return train_loader, train_loader_ca, train_loader_cb, val_loader_c, val_loader_b, len(dataset.query_c), len(dataset.query_b), num_classes




def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.num_works

    dataset = init_dataset(cfg.dataset, root=cfg.data_dir)

    num_classes = dataset.num_train_pids        # market:751,  msmst: 1041
    train_set = ImageDataset(dataset.train, train_transforms)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        sampler=RandomIdentitySampler(dataset.train, cfg.batch_size, cfg.img_per_id),           # 64, 4
        num_workers=num_workers, collate_fn=train_collate_fn
    )

    test_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, test_loader, len(dataset.query), num_classes



def make_data_loader_path(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.num_works

    dataset = init_dataset(cfg.dataset, root=cfg.data_dir)

    num_classes = dataset.num_train_pids        # 5000
    train_set = ImageDataset(dataset.train, train_transforms)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        sampler=RandomIdentitySampler(dataset.train, cfg.batch_size, cfg.img_per_id),           # 64, 4
        num_workers=num_workers, collate_fn=train_collate_fn_path
    )

    test_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn_path
    )


    return dataset, train_loader, test_loader, len(dataset.query), num_classes



def make_data_loader_path_visual(cfg):
    transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.num_works

    dataset = init_dataset(cfg.dataset, root=cfg.data_dir)

    num_classes = dataset.num_train_pids        # 5000
    train_set = ImageDataset(dataset.train, transforms)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=train_collate_fn_path
    )

    test_set = ImageDataset(dataset.query + dataset.gallery, transforms)
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn_path
    )


    return dataset, train_loader, test_loader, len(dataset.query), num_classes




def make_data_loader_path_data(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.num_works

    dataset = init_dataset(cfg.dataset, root=cfg.data_dir)
    data2set = init_dataset(cfg.data2set, root=cfg.data2_dir)

    num_classes = dataset.num_train_pids + data2set.num_train_pids        # 5000

    for ii, (img_path, pid, camid) in enumerate(data2set.train):
        pid_new = pid + dataset.num_train_pids
        camid_new = camid + dataset.num_train_cams
        dataset.train.append((img_path, pid_new, camid_new))

    train_set = ImageDataset(dataset.train, train_transforms)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        sampler=RandomIdentitySampler(dataset.train, cfg.batch_size, cfg.img_per_id),           # 64, 4
        num_workers=num_workers, collate_fn=train_collate_fn_path
    )

    test_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn_path
    )


    return dataset, train_loader, test_loader, len(dataset.query), num_classes




def make_data_loader_data(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.num_works

    dataset = init_dataset(cfg.dataset, root=cfg.data_dir)
    data2set = init_dataset(cfg.data2set, root=cfg.data2_dir)

    num_classes = dataset.num_train_pids + data2set.num_train_pids        # 5000

    for ii, (img_path, pid, camid) in enumerate(data2set.train):
        pid_new = pid + dataset.num_train_pids
        camid_new = camid + dataset.num_train_cams
        dataset.train.append((img_path, pid_new, camid_new))

    train_set = ImageDataset(dataset.train, train_transforms)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        sampler=RandomIdentitySampler(dataset.train, cfg.batch_size, cfg.img_per_id),           # 64, 4
        num_workers=num_workers, collate_fn=train_collate_fn
    )

    test_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )


    return dataset, train_loader, test_loader, len(dataset.query), num_classes



def make_data_loader_visual(cfg):
    train_transforms = build_transforms_visual(cfg, is_train=False)
    val_transforms = build_transforms_visual(cfg, is_train=False)
    num_workers = cfg.num_works

    dataset = init_dataset(cfg.dataset, root=cfg.data_dir)

    num_classes = dataset.num_train_pids        # market:751,  msmst: 1041
    train_set = ImageDataset(dataset.train, train_transforms)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        sampler=RandomIdentitySampler(dataset.train, cfg.batch_size, cfg.img_per_id),           # 64, 4
        num_workers=num_workers, collate_fn=train_collate_fn
    )

    test_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, test_loader, len(dataset.query), num_classes



def make_data_loader_visual_mask(cfg):
    train_transforms = build_transforms_visual(cfg, is_train=False)
    val_transforms = build_transforms_visual(cfg, is_train=False)
    num_workers = cfg.num_works

    dataset = init_dataset(cfg.dataset, root=cfg.data_dir)
    num_classes = dataset.num_train_pids        # market:751,  msmst: 1041

    train_loader = DataLoader(
        ImageDatasetVisualMask(dataset.train, train_transforms),
        batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers, collate_fn=train_collate_fn_path
    )

    test_loader = DataLoader(
        ImageDatasetVisualMask(dataset.query_c + dataset.gallery, val_transforms),
        batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn_path
    )

    return train_loader, test_loader, len(dataset.query_c), num_classes




def make_data_loader_mask(cfg):
    import torchvision.transforms as T
    from data.transforms.build import RandomErasing
    PIXEL_MEAN = [0.485, 0.456, 0.406]
    PIXEL_STD = [0.229, 0.224, 0.225]

    transform_train = T.Compose([
        T.Resize([cfg.height, cfg.width]),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop([cfg.height, cfg.width]),
        T.ToTensor(),
        T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
        RandomErasing(probability=0.5, mean=PIXEL_MEAN)
    ])

    transform_test = T.Compose([
        T.Resize([cfg.height, cfg.width]),
        T.ToTensor(),
        T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
    ])

    train_transforms_list = ['resize', 'hflip']
    val_transforms_list = ['resize']
    num_workers = cfg.num_works

    dataset = init_dataset(cfg.dataset, root=cfg.data_dir)

    num_classes = dataset.num_train_pids        # market:751,  msmst: 1041
    train_set = ImageDatasetMask(dataset.train, cfg, transform_train, train_transforms_list)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        sampler=RandomIdentitySampler(dataset.train, cfg.batch_size, cfg.img_per_id),           # 64, 4
        num_workers=num_workers, collate_fn=train_collate_mask
    )

    test_set = ImageDatasetMask(dataset.query + dataset.gallery, cfg, transform_test, val_transforms_list)
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_mask
    )

    return train_loader, test_loader, len(dataset.query), num_classes






def make_data_loader_prcc_gcn(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)

    dataset = init_dataset(cfg.dataset, root=cfg.data_dir)
    num_classes = dataset.num_train_pids    # 150
    num_workers = cfg.num_works             # 8

    train_set = ImageDatasetGcnMask(dataset.train, train_transforms)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        sampler=RandomIdentitySamplerGcn(dataset.train, cfg.batch_size, cfg.img_per_id),  # 64, 4
        num_workers=num_workers, collate_fn=train_collate_gcn_mask
    )

    train_loader_ca = DataLoader(
        ImageDataset(dataset.train_ca, train_transforms), batch_size=cfg.batch_size,
        sampler=RandomIdentitySamplerGcn(dataset.train, cfg.batch_size, cfg.img_per_id),  # 64, 4
        num_workers=num_workers, collate_fn=train_collate_gcn_mask
    )

    train_loader_cb = DataLoader(
        ImageDataset(dataset.train_cb, train_transforms), batch_size=cfg.batch_size,
        sampler=RandomIdentitySamplerGcn(dataset.train, cfg.batch_size, cfg.img_per_id),  # 64, 4
        num_workers=num_workers, collate_fn=train_collate_gcn_mask
    )

    val_set_c = ImageDatasetGcnMask(dataset.query_c + dataset.gallery, val_transforms)
    val_loader_c = DataLoader(
        val_set_c, batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_gcn_mask
    )

    val_set_b = ImageDatasetGcnMask(dataset.query_b + dataset.gallery, val_transforms)
    val_loader_b = DataLoader(
        val_set_b, batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_gcn_mask
    )


    return train_loader, train_loader_ca, train_loader_cb, val_loader_c, val_loader_b, len(dataset.query_c), len(dataset.query_b), num_classes




def make_data_loader_prcc(cfg, use_eraser=False):
    if use_eraser:
        train_transforms = build_transforms_head(cfg, is_train=True)
        val_transforms = build_transforms_head(cfg, is_train=False)
    else:
        train_transforms = build_transforms_base(cfg, is_train=True)
        val_transforms = build_transforms_base(cfg, is_train=False)

    dataset = init_dataset(cfg.dataset, root=cfg.data_dir)
    num_classes = dataset.num_train_pids    # 150
    num_workers = cfg.num_works             # 8

    train_loader = DataLoader(
        ImageDataset(dataset.train, train_transforms), batch_size=cfg.batch_size,
        sampler=RandomIdentitySampler(dataset.train, cfg.batch_size, cfg.img_per_id),  # 64, 4
        num_workers=num_workers, collate_fn=train_collate_fn_path
    )

    train_loader_ca = DataLoader(
        ImageDataset(dataset.train_ca, train_transforms), batch_size=cfg.batch_size,
        sampler=RandomIdentitySampler(dataset.train, cfg.batch_size, cfg.img_per_id),  # 64, 4
        num_workers=num_workers, collate_fn=train_collate_fn_path
    )

    train_loader_cb = DataLoader(
        ImageDataset(dataset.train_cb, train_transforms), batch_size=cfg.batch_size,
        sampler=RandomIdentitySampler(dataset.train, cfg.batch_size, cfg.img_per_id),  # 64, 4
        num_workers=num_workers, collate_fn=train_collate_fn_path
    )

    val_loader_c = DataLoader(
        ImageDataset(dataset.query_c + dataset.gallery, val_transforms), batch_size=cfg.batch_size_test,
        shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn_path
    )

    val_loader_b = DataLoader(
        ImageDataset(dataset.query_b + dataset.gallery, val_transforms), batch_size=cfg.batch_size_test,
        shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn_path
    )


    return train_loader, train_loader_ca, train_loader_cb, val_loader_c, val_loader_b, len(dataset.query_c), len(dataset.query_b), num_classes



def make_data_loader_base(cfg, h=64, w=32):
    train_transforms = build_transforms_base(cfg, is_train=True)
    val_transforms = build_transforms_base(cfg, is_train=False)

    dataset = init_dataset(cfg.dataset, root=cfg.data_dir)
    num_classes = dataset.num_train_pids    # 150
    num_workers = cfg.num_works             # 8


    train_loader = DataLoader(
        ImageDatasetGcnMask(dataset.train, h, w, train_transforms), batch_size=cfg.batch_size,
        sampler=RandomIdentitySamplerGcn(dataset.train, cfg.batch_size, cfg.img_per_id),  # 64, 4
        num_workers=num_workers, collate_fn=train_collate_gcn_mask
    )

    val_set = ImageDatasetGcnMask(dataset.query + dataset.gallery, h, w, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_gcn_mask
    )

    return train_loader, val_loader, len(dataset.query), num_classes



def make_data_loader_prcc_eraser(cfg, h=64, w=32):
    train_transforms = build_transforms_eraser(cfg, is_train=True)
    val_transforms = build_transforms_eraser(cfg, is_train=False)

    dataset = init_dataset(cfg.dataset, root=cfg.data_dir)
    num_classes = dataset.num_train_pids    # 150
    num_workers = cfg.num_works             # 8


    train_loader = DataLoader(
        ImageDatasetGcnMask(dataset.train, h, w, train_transforms), batch_size=cfg.batch_size,
        sampler=RandomIdentitySamplerGcn(dataset.train, cfg.batch_size, cfg.img_per_id),  # 64, 4
        num_workers=num_workers, collate_fn=train_collate_gcn_mask
    )

    train_loader_ca = DataLoader(
        ImageDatasetGcnMask(dataset.train_ca, h, w, train_transforms), batch_size=cfg.batch_size,
        sampler=RandomIdentitySamplerGcn(dataset.train, cfg.batch_size, cfg.img_per_id),  # 64, 4
        num_workers=num_workers, collate_fn=train_collate_gcn_mask
    )

    train_loader_cb = DataLoader(
        ImageDatasetGcnMask(dataset.train_cb, h, w, train_transforms), batch_size=cfg.batch_size,
        sampler=RandomIdentitySamplerGcn(dataset.train, cfg.batch_size, cfg.img_per_id),  # 64, 4
        num_workers=num_workers, collate_fn=train_collate_gcn_mask
    )

    val_set_c = ImageDatasetGcnMask(dataset.query_c + dataset.gallery, h, w, val_transforms)
    val_loader_c = DataLoader(
        val_set_c, batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_gcn_mask
    )

    val_set_b = ImageDatasetGcnMask(dataset.query_b + dataset.gallery, h, w, val_transforms)
    val_loader_b = DataLoader(
        val_set_b, batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_gcn_mask
    )


    return train_loader, train_loader_ca, train_loader_cb, val_loader_c, val_loader_b, len(dataset.query_c), len(dataset.query_b), num_classes



def make_data_loader_prcc_visual(cfg, h=64, w=32, use_eraser=False):
    val_transforms = build_transforms_visual(cfg, is_train=False, use_eraser=use_eraser)

    dataset = init_dataset(cfg.dataset, root=cfg.data_dir)
    num_classes = dataset.num_train_pids    # 150
    num_workers = cfg.num_works             # 8


    train_loader = DataLoader(
        ImageDatasetGcnMask(dataset.train, h, w, val_transforms), batch_size=cfg.batch_size,
        sampler=RandomIdentitySamplerGcn(dataset.train, cfg.batch_size, cfg.img_per_id),  # 64, 4
        num_workers=num_workers, collate_fn=train_collate_gcn_mask
    )

    train_loader_ca = DataLoader(
        ImageDatasetGcnMask(dataset.train_ca, h, w, val_transforms), batch_size=cfg.batch_size,
        sampler=RandomIdentitySamplerGcn(dataset.train, cfg.batch_size, cfg.img_per_id),  # 64, 4
        num_workers=num_workers, collate_fn=train_collate_gcn_mask
    )

    train_loader_cb = DataLoader(
        ImageDatasetGcnMask(dataset.train_cb, h, w, val_transforms), batch_size=cfg.batch_size,
        sampler=RandomIdentitySamplerGcn(dataset.train, cfg.batch_size, cfg.img_per_id),  # 64, 4
        num_workers=num_workers, collate_fn=train_collate_gcn_mask
    )

    val_set_c = ImageDatasetGcnMask(dataset.query_c + dataset.gallery, h, w, val_transforms)
    val_loader_c = DataLoader(
        val_set_c, batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_gcn_mask
    )

    val_set_b = ImageDatasetGcnMask(dataset.query_b + dataset.gallery, h, w, val_transforms)
    val_loader_b = DataLoader(
        val_set_b, batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_gcn_mask
    )


    return train_loader, train_loader_ca, train_loader_cb, val_loader_c, val_loader_b, len(dataset.query_c), len(dataset.query_b), num_classes



def update_label(labels):
    classes = sorted(list(set(labels)))
    classes = [item for item in classes if item != -1]
    start = len(classes)
    pid2label = {pid: ii for ii, pid in enumerate(classes)}
    y = []
    for lab in labels:
        if lab == -1:
            y.append(start)
            start += 1
        else:
            ll = pid2label[lab]
            y.append(ll)
    return y


def get_split(imgs, pids, paths, labels):
    imgs1 = imgs[0::2]
    pids1 = pids[0::2]
    paths1 = paths[0::2]
    labels1 = labels[0::2]
    imgs2 = imgs[1::2]
    pids2 = pids[1::2]
    paths2 = paths[1::2]
    labels2 = labels[1::2]
    return imgs1, pids1, paths1, labels1, imgs2, pids2, paths2, labels2


def get_split_list(data_list):
    out1 = []
    out2 = []
    for item in data_list:
        out1.append(item[0::2])
        out2.append(item[1::2])
    out = out1 + out2
    return out



def get_sub_dataset(pid2name, num_id=2000):
    pids_all = sorted((pid2name.keys()))
    pids_sub = random.sample(pids_all, num_id)
    data_sub = []
    for pid in pids_sub:
        lines = pid2name[pid]
        data_sub += lines
    return data_sub





