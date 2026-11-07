# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yue Hu <18671129361@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics

import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools.loss_utils import get_loss_mode, compute_training_loss

from icecream import ic


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Resume training path. When set, model_dir/config.yaml is loaded.')
    parser.add_argument('--pretrained_path', default='',
                        help='Checkpoint file or log dir used only for model weight initialization.')
    parser.add_argument('--run_test', action='store_true',
                        help='Run inference after training finishes.')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    opt = parser.parse_args()
    return opt


def print_parameter_audit(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen_summary = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            root = name.split('.')[0]
            frozen_summary[root] = frozen_summary.get(root, 0) + p.numel()
    print('Total parameters: {}'.format(total_params))
    print('Trainable parameters: {}'.format(trainable_params))
    print('Trainable parameter names:')
    for name in trainable_names:
        print('  {}'.format(name))
    print('Frozen parameter summary:')
    for name, count in sorted(frozen_summary.items()):
        print('  {}: {}'.format(name, count))


def main():
    opt = train_parser()
    if opt.model_dir and opt.pretrained_path:
        raise ValueError('--model_dir resumes an experiment; --pretrained_path initializes weights for a new experiment. Use only one.')
    config_source = os.path.join(opt.model_dir, 'config.yaml') if opt.model_dir else opt.hypes_yaml
    print('Config source: {}'.format(config_source))
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_params = hypes['train_params']
    # Preserve legacy single-process loading unless a dataset opts in.
    num_workers = int(train_params.get('num_workers', 0))
    persistent_workers = bool(train_params.get('persistent_workers', False))
    prefetch_factor = int(train_params.get('prefetch_factor', 2))
    empty_cache_each_iter = bool(
        train_params.get('empty_cache_each_iter', True)
    )
    if num_workers < 0:
        raise ValueError("train_params.num_workers must be >= 0")
    if prefetch_factor < 1:
        raise ValueError("train_params.prefetch_factor must be >= 1")
    print(
        "[DataLoader] num_workers={} persistent_workers={} "
        "prefetch_factor={} empty_cache_each_iter={}".format(
            num_workers, persistent_workers and num_workers > 0,
            prefetch_factor if num_workers > 0 else None,
            empty_cache_each_iter
        )
    )

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=num_workers,
                              persistent_workers=(persistent_workers and num_workers > 0),
                              prefetch_factor=(prefetch_factor if num_workers > 0 else None),
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=num_workers,
                            persistent_workers=(persistent_workers and num_workers > 0),
                            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
                            collate_fn=opencood_validate_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)

    # define the loss
    criterion = train_utils.create_loss(hypes)
    loss_mode = get_loss_mode(hypes)
    print('Supervision mode: {}'.format(loss_mode))

    # Resume keeps the old experiment config/epoch. Pretrained initializes only
    # model weights, then starts a new experiment with fresh optimizer/scheduler.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
    else:
        init_epoch = 0
        saved_path = train_utils.setup_train(hypes)
        if opt.pretrained_path:
            model = train_utils.load_pretrained_model(
                opt.pretrained_path,
                model,
                allowed_missing_prefixes=(
                    'fusion_net.dfa.',
                    'backbone.multi_scale_neck.',
                ),
                allow_unexpected=False,
            )

    print_parameter_audit(model)
    optimizer = train_utils.setup_optimizer(hypes, model)
    scheduler = train_utils.setup_lr_schedular(
        hypes, optimizer, init_epoch=init_epoch if opt.model_dir else None
    )

    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        for i, batch_data in enumerate(train_loader):
            if batch_data is None:
                continue
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            # case1 : late fusion train --> only ego needed,
            # and ego is (random) selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            batch_data['ego']['epoch'] = epoch
            output_dict = model(batch_data['ego'])
            # first argument is always your output dictionary,
            # second argument is always your label dictionary.
            final_loss = compute_training_loss(
                output_dict, batch_data, hypes, criterion, loss_mode
            )

            criterion.logging(epoch, i, len(train_loader), writer)

            # back-propagation
            final_loss.backward()
            optimizer.step()

            if empty_cache_each_iter:
                torch.cuda.empty_cache()

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    output_dict = model(batch_data['ego'])

                    final_loss = compute_training_loss(
                        output_dict, batch_data, hypes, criterion, loss_mode
                    )
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        scheduler.step(epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)
    torch.cuda.empty_cache()
    run_test = opt.run_test
    if run_test:
        fusion_method = opt.fusion_method
        cmd = f"python /GPFS/data/yhu/code/OpenCOOD/opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()
