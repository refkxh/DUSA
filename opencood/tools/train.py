# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils
from opencood.tools import train_utils, inference_utils


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--adv_training', action='store_true',
                        help='whether to perform adversarial training domain adaptation')
    parser.add_argument("--target_domain_suffix", type=str, default='dair',
                        help='adversarial training target domain yaml file suffix')
    parser.add_argument('--use_pseudo_label', action='store_true',
                        help='whether to use pseudo label for adversarial training')
    parser.add_argument('--pseudo_label_id', type=str, default='0',
                        help='the id of pseudo label')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    multi_gpu_utils.init_distributed_mode(opt)

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    if opt.adv_training:
        target_domain_hypes_path = opt.hypes_yaml.replace('.yaml', f'_{opt.target_domain_suffix}.yaml')
        target_domain_hypes = yaml_utils.load_yaml(target_domain_hypes_path)
        opencood_validate_dataset = build_dataset(target_domain_hypes, visualize=False, train=False)

        if opt.use_pseudo_label:
            target_domain_hypes['pseudo_label'] = {'id': opt.pseudo_label_id, 'pos_thresh': 0.5, 'neg_thresh': 0.25}
        target_domain_train_dataset = build_dataset(target_domain_hypes, visualize=False, train=True)
        if opt.use_pseudo_label:
            del target_domain_hypes['pseudo_label']

    else:
        opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
        val_batch_size = target_domain_hypes['train_params']['batch_size'] if opt.adv_training else \
            hypes['train_params']['batch_size']
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=val_batch_size,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)

    if opt.adv_training:
        if opt.distributed:
            target_domain_sample_train = DistributedSampler(target_domain_train_dataset)
            target_domain_batch_sampler_train = torch.utils.data.BatchSampler(
                target_domain_sample_train, target_domain_hypes['train_params']['batch_size'], drop_last=True)

            target_domain_train_loader = DataLoader(target_domain_train_dataset,
                                                    batch_sampler=target_domain_batch_sampler_train,
                                                    num_workers=8,
                                                    collate_fn=target_domain_train_dataset.collate_batch_train)
        else:
            target_domain_train_loader = DataLoader(target_domain_train_dataset,
                                                    batch_size=target_domain_hypes['train_params']['batch_size'],
                                                    num_workers=8,
                                                    collate_fn=target_domain_train_dataset.collate_batch_train,
                                                    shuffle=True,
                                                    pin_memory=False,
                                                    drop_last=True)

        target_domain_train_loader_iter = iter(target_domain_train_loader)

    print('---------------Creating Model------------------')
    if opt.adv_training:
        hypes['model']['args']['domain_cls'] = True
        # hypes['model']['args']['domain_cls_img'] = True
        # hypes['model']['args']['agent_cls'] = True
        hypes['model']['args']['agent_cls_img'] = True
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        else:
            scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'],
                                       domain='source' if opt.adv_training else None)
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'],
                                           domain='source' if opt.adv_training else None)

            if opt.adv_training:
                if not opt.half:
                    final_loss.backward()
                else:
                    scaler.scale(final_loss).backward()

                try:
                    target_domain_batch_data = next(target_domain_train_loader_iter)
                except StopIteration:
                    target_domain_train_loader_iter = iter(target_domain_train_loader)
                    target_domain_batch_data = next(target_domain_train_loader_iter)
                target_domain_batch_data = train_utils.to_device(target_domain_batch_data, device)
                if not opt.half:
                    ouput_dict_target = model(target_domain_batch_data['ego'])
                    final_loss = criterion(ouput_dict_target,
                                            target_domain_batch_data['ego']['label_dict'],
                                            domain='target',
                                            da_agent_loss=True,
                                            use_pseudo_label=opt.use_pseudo_label)
                else:
                    with torch.cuda.amp.autocast():
                        ouput_dict_target = model(target_domain_batch_data['ego'])
                        final_loss = criterion(ouput_dict_target,
                                                target_domain_batch_data['ego']['label_dict'],
                                                domain='target',
                                                da_agent_loss=True,
                                                use_pseudo_label=opt.use_pseudo_label)

            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

        if opt.use_pseudo_label:
            print('Saving Pseudo Label')
            target_domain_hypes['pseudo_label_generation'] = True
            target_domain_hypes['postprocess']['target_args']['score_threshold'] = 0.25
            print('Dataset Building')
            opencood_dataset = build_dataset(target_domain_hypes, visualize=False, train=False)
            print(f"{len(opencood_dataset)} samples found.")
            data_loader = DataLoader(opencood_dataset,
                                     batch_size=1,
                                     num_workers=0,
                                     collate_fn=opencood_dataset.collate_batch_test,
                                     shuffle=False,
                                     pin_memory=False,
                                     drop_last=False)

            model.eval()
            for i, batch_data in tqdm(enumerate(data_loader)):
                with torch.no_grad():
                    batch_data = train_utils.to_device(batch_data, device)
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        inference_utils.inference_intermediate_fusion(batch_data,
                                                                      model,
                                                                      opencood_dataset)

                    pseudo_label_save_dir = \
                        os.path.join(target_domain_hypes['dair_data_dir'], 'pseudo-label', opt.pseudo_label_id)
                    os.makedirs(pseudo_label_save_dir, exist_ok=True)
                    inference_utils.save_pseudo_label_dair(pred_box_tensor,
                                                           pred_score,
                                                           batch_data['ego']['lidar_pose'][0],
                                                           batch_data['ego']['frame_id'],
                                                           pseudo_label_save_dir)

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
