import sys
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import make_grid

import mixed_precision
from utils import test_model
from stats import AverageMeterSet, update_train_accuracies
from datasets import Dataset
from costs import loss_xent


def _train(model, optim_inf, scheduler_inf, checkpoint, epochs,
           train_loader, test_loader, stat_tracker, log_dir, device):
    '''
    Training loop for optimizing encoder
    '''
    # If mixed precision is on, will add the necessary hooks into the model
    # and optimizer for half() conversions
    model, optim_inf = mixed_precision.initialize(model, optim_inf)
    optim_raw = mixed_precision.get_optimizer(optim_inf)
    # get target LR for LR warmup -- assume same LR for all param groups
    for pg in optim_raw.param_groups:
        lr_real = pg['lr']

    # prepare checkpoint and stats accumulator
    next_epoch, total_updates = checkpoint.get_current_position(fine_tuning=False)
    fast_stats = AverageMeterSet()
    # run main training loop
    for epoch in range(next_epoch, epochs):
        epoch_stats = AverageMeterSet()
        epoch_updates = 0
        time_start = time.time()

        for _, ((images1, images2), labels) in enumerate(train_loader):
            # get data and info about this minibatch
            labels = torch.cat([labels, labels]).to(device)
            images1 = images1.to(device)
            images2 = images2.to(device)
            # run forward pass through model to get global and local features
            res_dict = model(x1=images1, x2=images2, fine_tuning=False, get_bop_lgt=False)
            lgt_glb_mlp, lgt_bop_mlp, lgt_glb_lin, lgt_bop_lin = res_dict['class']
            # compute costs for all self-supervised tasks
            loss_g2l = (res_dict['g2l_1t5'] +
                        res_dict['g2l_1t7'] +
                        res_dict['g2l_5t5'])
            loss_inf = loss_g2l + res_dict['lgt_reg']

            # compute loss for online evaluation classifiers
            loss_cls = (loss_xent(lgt_glb_mlp, labels) +
                        loss_xent(lgt_bop_mlp, labels) +
                        loss_xent(lgt_glb_lin, labels) +
                        loss_xent(lgt_bop_lin, labels))

            # do hacky learning rate warmup -- we stop when LR hits lr_real
            if (total_updates < 500):
                lr_scale = min(1., float(total_updates + 1) / 500.)
                for pg in optim_raw.param_groups:
                    pg['lr'] = lr_scale * lr_real

            # reset gradient accumlators and do backprop
            loss_opt = loss_inf + loss_cls
            optim_inf.zero_grad()
            mixed_precision.backward(loss_opt, optim_inf)  # backwards with fp32/fp16 awareness
            optim_inf.step()

            # record loss and accuracy on minibatch
            epoch_stats.update_dict({
                'loss_inf': loss_inf.item(),
                'loss_cls': loss_cls.item(),
                'loss_g2l': loss_g2l.item(),
                'lgt_reg': res_dict['lgt_reg'].item(),
                'loss_g2l_1t5': res_dict['g2l_1t5'].item(),
                'loss_g2l_1t7': res_dict['g2l_1t7'].item(),
                'loss_g2l_5t5': res_dict['g2l_5t5'].item()
            }, n=1)
            update_train_accuracies(epoch_stats, labels, lgt_glb_mlp, lgt_bop_mlp,
                                    lgt_glb_lin, lgt_bop_lin)

            # shortcut diagnostics to deal with long epochs
            total_updates += 1
            epoch_updates += 1
            if (total_updates % 100) == 0:
                time_stop = time.time()
                spu = (time_stop - time_start) / 100.
                print('Epoch {0:d}, {1:d} updates -- {2:.4f} sec/update'
                      .format(epoch, epoch_updates, spu))
                time_start = time.time()
            if (total_updates % 1000) == 0:
                # record diagnostics
                eval_start = time.time()
                fast_stats = AverageMeterSet()
                test_model(model, test_loader, device,
                           fast_stats, max_evals=50000, get_bop_lgt=False)
                stat_tracker.record_stats(
                    fast_stats.averages(total_updates, prefix='fast/'))
                eval_time = time.time() - eval_start
                stat_str = fast_stats.pretty_string(ignore=model.tasks)
                stat_str = '-- {0:d} updates, eval_time {1:.2f}: {2:s}'.format(
                    total_updates, eval_time, stat_str)
                print(stat_str)

        # update learning rate
        scheduler_inf.step(epoch)
        test_model(model, test_loader, device,
                   fast_stats, max_evals=200000, get_bop_lgt=False)
        epoch_str = epoch_stats.pretty_string(ignore=model.tasks)
        diag_str = '{0:d}: {1:s}'.format(epoch, epoch_str)
        print(diag_str)
        sys.stdout.flush()
        stat_tracker.record_stats(epoch_stats.averages(epoch, prefix='costs/'))
        # checkpoint the model
        checkpoint.update(epoch + 1, total_updates, fine_tuning=False)


def train_self_supervised(model, learning_rate, dataset, train_loader,
                          test_loader, stat_tracker, checkpoint, log_dir, device):
    # configure optimizer
    mods_inf = [m for m in model.info_modules]
    mods_cls = [m for m in model.class_modules]
    mods_to_opt = mods_inf + mods_cls
    optimizer = optim.Adam(
        [{'params': mod.parameters(), 'lr': learning_rate} for mod in mods_to_opt],
        betas=(0.8, 0.999), weight_decay=1e-5, eps=1e-8)
    # configure learning rate schedulers for the optimizers
    if dataset in [Dataset.C10, Dataset.C100, Dataset.STL10]:
        scheduler = MultiStepLR(optimizer, milestones=[250, 280], gamma=0.2)
        epochs = 300
    else:
        # best imagenet results use longer schedules...
        # -- e.g., milestones=[60, 90], epochs=110
        scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=0.2)
        epochs = 45
    # train the model
    _train(model, optimizer, scheduler, checkpoint, epochs,
           train_loader, test_loader, stat_tracker, log_dir, device)
