import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch import optim
import numpy as np
from tqdm import tqdm

from utils import AverageMeter, accuracy
from loss import LossComputer
import sloss
#from pytorch_transformers import AdamW, WarmupLinearSchedule

def run_epoch(epoch, model, student, soptimizer,scriterion,
              loader,args,is_training, show_progress=False, log_every=50, scheduler=None):
    """
    scheduler is only used inside this function if model is bert.
    """
    model.eval()
    if is_training:
        student.train()
        if args.model == 'bert':
            model.zero_grad()
    else:
        student.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):

            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            if args.model == 'bert':
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y
                )[1] # [1] returns logits
            else:
                outputs = student(x)
            with torch.no_grad():
                out_teacher = model(x)
            loss_main = scriterion(outputs, out_teacher, y)

            if is_training:
                if args.model == 'bert':
                    loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scheduler.step()
                    optimizer.step()
                    model.zero_grad()
                else:
                    soptimizer.zero_grad()
                    loss_main.backward()
                    soptimizer.step()

            if False and is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()
                scsv_logger.log(epoch, batch_idx, sloss_computer.get_stats(student, args))
                scsv_logger.flush()
                sloss_computer.log_stats(logger, is_training)
                sloss_computer.reset_stats()

        if False and (not is_training) and loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()
            scsv_logger.log(epoch, batch_idx, sloss_computer.get_stats(student, args))
            scsv_logger.flush()
            sloss_computer.log_stats(logger, is_training)
            if is_training:
                sloss_computer.reset_stats()


def train(model,student, criterion,scriterion, dataset,
          logger, train_csv_logger, val_csv_logger,
          test_csv_logger,strain_csv_logger, sval_csv_logger, stest_csv_logger, args, epoch_offset):
    model = model.cuda()
    student = student.cuda()
    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)



    # BERT uses its own scheduler and optimizer
    if args.model == 'bert':
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            eps=args.adam_epsilon)
        t_total = len(dataset['train_loader']) * args.n_epochs
        print(f'\nt_total is {t_total}\n')
        scheduler = WarmupLinearSchedule(
            optimizer,
            warmup_steps=args.warmup_steps,
            t_total=t_total)
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay)
        soptimizer = optim.Adam(student.parameters())
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                'min',
                factor=0.1,
                patience=5,
                threshold=0.0001,
                min_lr=0,
                eps=1e-08)
        else:
            scheduler = None

    best_val_acc = 0
    for epoch in range(epoch_offset, epoch_offset+args.n_epochs):
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        run_epoch(epoch, model,student, soptimizer,scriterion,
            dataset['train_loader'], args,
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler)



        if epoch % 1 == 0:
            torch.save(student, os.path.join(args.log_dir, 's%d_model.pth' % epoch))
        logger.write('\n')
    #torch.save(student, os.path.join(args.log_dir, 'slast_model.pth'))
        
     
