# encoding: utf-8

import os
import torch
import datetime
import shutil
import random
import cv2
import numpy as np
import copy
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from engine.inference import inference_prcc_pcb, inference_prcc_global

import copy
import errno
import shutil

def fliplr(img):
    # flip horizontal
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.cpu().index_select(3, inv_idx)
    return img_flip.cuda()


def norm(f):
    f = f.squeeze()
    fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
    f = f.div(fnorm.expand_as(f))
    return f


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def erase_attention(feature, attention_map, thr_val=0.7, PIXEL_MEAN=[0.485, 0.456, 0.406]):      # [64, 3, 256, 128], [64, 1, 256, 128], 0.7
    b, _, h, w = attention_map.size()
    thr_val = random.uniform(0.5, 0.9)
    pos = torch.ge(attention_map, thr_val)            # [64, 1, 256, 128]
    pos = pos.expand_as(feature)                      # [64, 3, 256, 128]
    feature[range(b), 0][pos.data[range(b), 0]] = PIXEL_MEAN[0]                # [64, 3, 256, 128]
    feature[range(b), 1][pos.data[range(b), 1]] = PIXEL_MEAN[1]
    feature[range(b), 2][pos.data[range(b), 2]] = PIXEL_MEAN[2]
    return feature



def get_cam_mask(cam_ori, target, h, w):
    target = target.long()
    b = cam_ori.size(0)

    attention = cam_ori.detach().clone().requires_grad_(True)[range(b), target.data, :, :]  # [64, 16, 8]
    attention = attention.unsqueeze(1)                # [64, 1, 28, 28]
    attention = F.interpolate(attention.cpu(), (h, w), mode='bilinear', align_corners=False)      # [64, 1, 256, 128]
    attention = normalize_tensor(attention)           # [64, 1, 256, 128]

    return attention


def normalize_tensor(x):      # [64, 1, 28, 28]
    map_size = x.size()
    aggregated = x.view(map_size[0], map_size[1], -1)      # [64, 1, 784]
    minimum, _ = torch.min(aggregated, dim=-1, keepdim=True)             # [64, 1, 1]
    maximum, _ = torch.max(aggregated, dim=-1, keepdim=True)             # [64, 1, 1]
    normalized = torch.div(aggregated - minimum, maximum - minimum)      # [64, 1, 784]
    normalized = normalized.view(map_size)      # [64, 1, 28, 28]

    return normalized

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        old_lr = float(param_group['lr'])
        return old_lr


def do_train_prcc_base(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_fn, num_query, start_epoch, acc_best, lr_type='step'):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    print_num = int(len(train_loader) / 5)

    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        lr = float(get_lr(optimizer))

        for ii, (img, target, path, mask) in enumerate(train_loader):       # [64, 3, 256, 128],  [64,],  [64,], [64, 6, 256, 128]
            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]
            b, c, h, w = img.shape

            score, feat = model(img)                            # [64, 150], [64, 2018]

            loss = loss_fn(score, feat, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (score.max(1)[1] == target).float().mean()
            loss = float(loss)
            acc = float(acc)

            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii+1, len(train_loader), loss, acc, lr))

        mAP, cmc1 = inference_prcc_global(model, test_loader, num_query)
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - cmc1: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        if is_best:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_acc': acc_test,
            }, is_best, fpath=cfg.logs_dir)
            last_acc_val = acc_test


        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)
        writer.add_scalar('lr', lr, epoch + 1)

        if lr_type == 'plateau':
            scheduler.step(acc_test)

        if lr_type == 'step':
            scheduler.step()


    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1 = inference_prcc_global(model, test_loader, num_query)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%}, mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()



def do_train_prcc_hpm(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_fn, num_query, start_epoch, acc_best, lr_type='step'):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    loss = 0.0
    print_num = int(len(train_loader) / 5)

    for epoch in range(start_epoch, cfg.max_epochs):            # 120
        model.train()

        for ii, (img, target, path, mask) in enumerate(train_loader):       # [64, 3, 256, 128],  [64,]
            optimizer.zero_grad()

            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]

            scores, feats = model(img)                          # [64, 751], [64, 256]

            loss = []
            for jj, (score, feat) in enumerate(zip(scores, feats)):       # [64, 751], [64, 256]
                loss.append(loss_fn(score, feat, target))                 # 11.1710
            # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
            loss = torch.stack(loss).sum()
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (score.max(1)[1] == target).float().mean()
            loss = float(loss)
            acc = float(acc)
            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii+1, len(train_loader), loss, acc, scheduler.get_lr()[0]))

        mAP, cmc1 = inference_prcc_global(model, test_loader, num_query)

        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - Final: cmc1: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_acc': acc_test,
        }, is_best, fpath=cfg.logs_dir)
        if is_best:
            last_acc_val = acc_test

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)

        if lr_type == 'plateau':
            scheduler.step(acc_test)

        if lr_type == 'step':
            scheduler.step()

    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1 = inference_prcc_global(model, test_loader, num_query)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()



def do_train_prcc_hpm_pix(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_fn, num_query, start_epoch, acc_best, lr_type='step'):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    loss = 0.0
    print_num = int(len(train_loader) / 5)

    for epoch in range(start_epoch, cfg.max_epochs):            # 120
        model.train()


        for ii, (img, target, path, mask) in enumerate(train_loader):       # [64, 3, 256, 128],  [64,]
            optimizer.zero_grad()

            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]
            mask = mask.cuda() if use_cuda else mask  # [64, 6, 256, 128]

            b, c, h, w = img.shape

            mask_i = mask.argmax(dim=1).unsqueeze(dim=1)  # [64, 1, 256, 128]
            mask_i = mask_i.expand_as(img)
            img_a = copy.deepcopy(img)

            # upper clothes sampling
            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 2] = img_r[msk_r == 2]

            # pant sampling
            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 3] = img_r[msk_r == 3]

            img_c = torch.cat([img, img_a], dim=0)
            target_c = torch.cat([target, target], dim=0)

            scores, feats = model(img_c)                          # [64, 751], [64, 256]

            loss = []
            for jj, (score, feat) in enumerate(zip(scores, feats)):       # [64, 751], [64, 256]
                loss.append(loss_fn(score, feat, target_c))                 # 11.1710
            # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
            loss = torch.stack(loss).sum()
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (score.max(1)[1] == target_c).float().mean()
            loss = float(loss)
            acc = float(acc)
            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii+1, len(train_loader), loss, acc, scheduler.get_lr()[0]))

        mAP, cmc1 = inference_prcc_global(model, test_loader, num_query)

        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - Final: cmc1: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_acc': acc_test,
        }, is_best, fpath=cfg.logs_dir)
        if is_best:
            last_acc_val = acc_test

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)

        if lr_type == 'plateau':
            scheduler.step(acc_test)

        if lr_type == 'step':
            scheduler.step()

    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1 = inference_prcc_global(model, test_loader, num_query)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()



def do_train_prcc_mgn(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_fn, num_query, start_epoch, acc_best, lr_type='step'):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    loss = 0.0
    print_num = int(len(train_loader) / 5)

    for epoch in range(start_epoch, cfg.max_epochs):            # 120
        model.train()

        for ii, (img, target, path, mask) in enumerate(train_loader):       # [64, 3, 256, 128],  [64,]
            optimizer.zero_grad()

            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]

            feats, feat_sub, scores = model(img)  # predict, [fg_p1, fg_p2, fg_p3], [l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3]

            loss = loss_fn(feat_sub, scores, target)
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (scores[0].max(1)[1] == target).float().mean()
            loss = float(loss)
            acc = float(acc)
            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii + 1, len(train_loader), loss, acc, scheduler.get_lr()[0]))

        mAP, cmc1 = inference_prcc_global(model, test_loader, num_query)
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - cmc1: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_acc': acc_test,
        }, is_best, fpath=cfg.logs_dir)
        if is_best:
            last_acc_val = acc_test

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)

        if lr_type == 'plateau':
            scheduler.step(acc_test)

        if lr_type == 'step':
            scheduler.step()

    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1 = inference_prcc_global(model, test_loader, num_query)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()




def do_train_prcc_mgn_pix(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_fn, num_query, start_epoch, acc_best, lr_type='step'):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    loss = 0.0
    print_num = int(len(train_loader) / 5)

    for epoch in range(start_epoch, cfg.max_epochs):            # 120
        model.train()

        for ii, (img, target, path, mask) in enumerate(train_loader):       # [64, 3, 256, 128],  [64,]
            optimizer.zero_grad()

            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]
            mask = mask.cuda() if use_cuda else mask  # [64, 6, 256, 128]

            b, c, h, w = img.shape

            mask_i = mask.argmax(dim=1).unsqueeze(dim=1)  # [64, 1, 256, 128]
            mask_i = mask_i.expand_as(img)
            img_a = copy.deepcopy(img)

            # upper clothes sampling
            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 2] = img_r[msk_r == 2]

            # pant sampling
            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 3] = img_r[msk_r == 3]

            img_c = torch.cat([img, img_a], dim=0)
            target_c = torch.cat([target, target], dim=0)

            feats, feat_sub, scores = model(img_c)  # predict, [fg_p1, fg_p2, fg_p3], [l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3]

            loss = loss_fn(feat_sub, scores, target_c)
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (scores[0].max(1)[1] == target_c).float().mean()
            loss = float(loss)
            acc = float(acc)
            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii + 1, len(train_loader), loss, acc, scheduler.get_lr()[0]))

        mAP, cmc1 = inference_prcc_global(model, test_loader, num_query)
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - cmc1: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_acc': acc_test,
        }, is_best, fpath=cfg.logs_dir)
        if is_best:
            last_acc_val = acc_test

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)

        if lr_type == 'plateau':
            scheduler.step(acc_test)

        if lr_type == 'step':
            scheduler.step()

    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1 = inference_prcc_global(model, test_loader, num_query)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()



def do_train_prcc_pcb(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_fn, num_query, start_epoch, acc_best, lr_type='step'):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    print_num = int(len(train_loader) / 5)

    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        lr = float(get_lr(optimizer))

        for ii, (img, target, path, mask) in enumerate(train_loader):       # [64, 3, 256, 128],  [64,]
            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]
            b, c, h, w = img.shape

            score, feat = model(img)    # [64, 4, 150], [64, 2048*4]

            loss = loss_fn(score, feat, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (score.sum(dim=1).max(1)[1] == target).float().mean()
            loss = float(loss)
            acc = float(acc)

            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii+1, len(train_loader), loss, acc, lr))

        mAP, cmc1 = inference_prcc_pcb(model, test_loader, num_query)

        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - cmc1: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_acc': acc_test,
        }, is_best, fpath=cfg.logs_dir)
        if is_best:
            last_acc_val = acc_test

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)
        writer.add_scalar('lr', lr, epoch + 1)

        if lr_type == 'plateau':
            scheduler.step(acc_test)

        if lr_type == 'step':
            scheduler.step()


    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1 = inference_prcc_pcb(model, test_loader, num_query)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%}, mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()



def do_train_prcc_pcb_pix(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_fn, num_query, start_epoch, acc_best, lr_type='step'):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    print_num = int(len(train_loader) / 5)

    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        lr = float(get_lr(optimizer))

        for ii, (img, target, path, mask) in enumerate(train_loader):       # [64, 3, 256, 128],  [64,]
            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]
            b, c, h, w = img.shape

            mask = mask.cuda() if use_cuda else mask  # [64, 6, 256, 128]
            mask_i = mask.argmax(dim=1).unsqueeze(dim=1)  # [64, 1, 256, 128]
            mask_i = mask_i.expand_as(img)
            img_a = copy.deepcopy(img)

            # upper clothes sampling
            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 2] = img_r[msk_r == 2]

            # pant sampling
            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 3] = img_r[msk_r == 3]

            img_c = torch.cat([img, img_a], dim=0)
            target_c = torch.cat([target, target], dim=0)
            score, feat = model(img_c)  # [64, 150], [64, 2018]

            loss = loss_fn(score, feat, target_c)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (score.sum(dim=1).max(1)[1] == target_c).float().mean()
            loss = float(loss)
            acc = float(acc)

            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii+1, len(train_loader), loss, acc, lr))

        mAP, cmc1 = inference_prcc_pcb(model, test_loader, num_query)

        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - cmc1: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_acc': acc_test,
        }, is_best, fpath=cfg.logs_dir)
        if is_best:
            last_acc_val = acc_test

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)
        writer.add_scalar('lr', lr, epoch + 1)

        if lr_type == 'plateau':
            scheduler.step(acc_test)

        if lr_type == 'step':
            scheduler.step()


    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1 = inference_prcc_pcb(model, test_loader, num_query)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%}, mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()




def do_train_prcc_pix_mse(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_fn, num_query, start_epoch, acc_best, lr_type='step'):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    print_num = int(len(train_loader) / 5)

    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        lr = float(get_lr(optimizer))

        for ii, (img, target, path, mask) in enumerate(train_loader):       # [64, 3, 256, 128],  [64,],  [64,], [64, 6, 256, 128]
            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]
            b, c, h, w = img.shape

            mask = mask.cuda() if use_cuda else mask            # [64, 6, 256, 128]
            mask_i = mask.argmax(dim=1).unsqueeze(dim=1)        # [64, 1, 256, 128]
            mask_i = mask_i.expand_as(img)
            img_a = copy.deepcopy(img)

            # upper clothes sampling
            index = np.random.permutation(b)
            img_r = img[index]                                  # [64, 3, 256, 128]
            msk_r = mask_i[index]                               # [64, 6, 256, 128]
            img_a[mask_i == 2] = img_r[msk_r == 2]

            # pant sampling
            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 3] = img_r[msk_r == 3]

            img_c = torch.cat([img, img_a], dim=0)
            target_c = torch.cat([target, target], dim=0)
            score, feat = model(img_c)                            # [64, 150], [64, 2018]

            loss = loss_fn(score, feat, target_c)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (score.max(1)[1] == target_c).float().mean()
            loss = float(loss)
            acc = float(acc)

            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii+1, len(train_loader), loss, acc, lr))

        mAP, cmc1 = inference_prcc_global(model, test_loader, num_query)
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - cmc1: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_acc': acc_test,
        }, is_best, fpath=cfg.logs_dir)
        if is_best:
            last_acc_val = acc_test

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)
        writer.add_scalar('lr', lr, epoch + 1)

        if lr_type == 'plateau':
            scheduler.step(acc_test)

        if lr_type == 'step':
            scheduler.step()


    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1 = inference_prcc_global(model, test_loader, num_query)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%}, mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()




def save_checkpoint(state, is_best, fpath):
    if len(fpath) != 0:
        mkdir_if_missing(fpath)

    fpath = os.path.join(fpath, 'checkpoint.pth')
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, os.path.join(os.path.dirname(fpath), 'checkpoint_best.pth'))

