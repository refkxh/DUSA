# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor,
                target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class PointPillarLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarLoss, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0

        self.cls_weight = args['cls_weight']
        self.reg_coe = args['reg']
        self.loss_dict = {}

    def forward(self, output_dict, target_dict, domain=None, da_agent_loss=False, use_pseudo_label=False):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        domain : Optional [str]
            Domain of the loss, if None, the discriminator loss is not calculated.
        da_agent_loss : bool
            Whether to calculate the loss for the domain adaptation agent discriminator.
        """
        total_loss = 0

        # Detection loss
        if domain is None or domain == 'source' or use_pseudo_label:
            rm = output_dict['rm']
            psm = output_dict['psm']
            targets = target_dict['targets']

            cls_preds = psm.permute(0, 2, 3, 1).contiguous()

            box_cls_labels = target_dict['pos_equal_one']
            box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

            positives = box_cls_labels > 0
            negatives = box_cls_labels == 0

            if use_pseudo_label:
                box_cls_labels_neg = target_dict['neg_equal_one']
                box_cls_labels_neg = box_cls_labels_neg.view(psm.shape[0], -1).contiguous()
                negatives = box_cls_labels_neg > 0

            negative_cls_weights = negatives * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            reg_weights = positives.float()

            pos_normalizer = positives.sum(1, keepdim=True).float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_targets = box_cls_labels
            cls_targets = cls_targets.unsqueeze(dim=-1)

            cls_targets = cls_targets.squeeze(dim=-1)
            one_hot_targets = torch.zeros(
                *list(cls_targets.shape), 2,
                dtype=cls_preds.dtype, device=cls_targets.device
            )
            one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
            cls_preds = cls_preds.view(psm.shape[0], -1, 1)
            one_hot_targets = one_hot_targets[..., 1:]

            cls_loss_src = self.cls_loss_func(cls_preds,
                                              one_hot_targets,
                                              weights=cls_weights)  # [N, M]
            cls_loss = cls_loss_src.sum() / psm.shape[0]
            conf_loss = cls_loss * self.cls_weight

            # regression
            rm = rm.permute(0, 2, 3, 1).contiguous()
            rm = rm.view(rm.size(0), -1, 7)
            targets = targets.view(targets.size(0), -1, 7)
            box_preds_sin, reg_targets_sin = self.add_sin_difference(rm,
                                                                     targets)
            loc_loss_src =\
                self.reg_loss_func(box_preds_sin,
                                   reg_targets_sin,
                                   weights=reg_weights)
            reg_loss = loc_loss_src.sum() / rm.shape[0]
            reg_loss *= self.reg_coe

            total_loss += reg_loss + conf_loss

            if not use_pseudo_label:
                self.loss_dict.update({'reg_loss': reg_loss,
                                       'conf_loss': conf_loss})
            else:
                self.loss_dict.update({'reg_loss_pseudo': reg_loss,
                                       'conf_loss_pseudo': conf_loss})

        # Domain classification loss
        if domain is not None:
            if 'domain_cls' in output_dict:
                domain_cls_pred = output_dict['domain_cls']  # N, 1
                if domain == 'source':
                    domain_cls_gt = torch.zeros_like(domain_cls_pred, dtype=torch.float32, device=domain_cls_pred.device)
                else:
                    # domain_cls_pred = domain_cls_pred[::2]
                    domain_cls_gt = torch.ones_like(domain_cls_pred, dtype=torch.float32, device=domain_cls_pred.device)

                domain_cls_loss = F.binary_cross_entropy_with_logits(domain_cls_pred, domain_cls_gt)
                domain_cls_loss = domain_cls_loss * 0.5

                total_loss += domain_cls_loss
                self.loss_dict[f'{domain}_domain_cls_loss'] = domain_cls_loss

            # if 'domain_feat_cls' in output_dict:
            #     domain_feat_cls_pred = output_dict['domain_feat_cls']  # N, C, H, W
            #     N = domain_feat_cls_pred.shape[0]
            #     domain_feat_cls_pred = domain_feat_cls_pred.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
            #     domain_feat_cls_pred = domain_feat_cls_pred.reshape(N, -1)  # N, H, W, C -> N, H*W*C
            #
            #     if domain == 'source':
            #         domain_feat_cls_gt = torch.zeros_like(domain_feat_cls_pred, dtype=torch.float32, device=domain_feat_cls_pred.device)
            #     else:
            #         # domain_feat_cls_pred = domain_feat_cls_pred[::2]
            #         domain_feat_cls_gt = torch.ones_like(domain_feat_cls_pred, dtype=torch.float32, device=domain_feat_cls_pred.device)
            #
            #     domain_feat_cls_loss = F.binary_cross_entropy_with_logits(domain_feat_cls_pred, domain_feat_cls_gt)
            #     domain_feat_cls_loss = domain_feat_cls_loss * 0.5
            #
            #     total_loss += domain_feat_cls_loss
            #     self.loss_dict[f'{domain}_domain_feat_cls_loss'] = domain_feat_cls_loss
            #
            # if 'domain_ins_cls' in output_dict:
            #     domain_cls_pred = output_dict['domain_ins_cls']  # N, H
            #     # domain_ins_cls_pred = domain_ins_cls_pred.view(-1, 1)  # N, H -> N*H, 1
            #     if domain == 'source':
            #         domain_cls_gt = torch.zeros_like(domain_cls_pred, dtype=torch.float32, device=domain_cls_pred.device)
            #     else:
            #         # domain_ins_cls_pred = domain_ins_cls_pred[::2]
            #         domain_cls_gt = torch.ones_like(domain_cls_pred, dtype=torch.float32, device=domain_cls_pred.device)
            #
            #     domain_cls_loss = F.binary_cross_entropy_with_logits(domain_cls_pred, domain_cls_gt)
            #     domain_cls_loss = domain_cls_loss * 0.5
            #
            #     total_loss += domain_cls_loss
            #     self.loss_dict[f'{domain}_domain_ins_cls_loss'] = domain_cls_loss

        # Agent classification loss
        if da_agent_loss and domain == 'target':
            if 'agent_cls_img' in output_dict:
                agent_cls_img_pred = output_dict['agent_cls_img']  # N, C, H, W
                agent_cls_img_weight = output_dict['agent_cls_img_weight']  # N, C, H, W
                N = agent_cls_img_pred.shape[0]
                agent_cls_img_pred = agent_cls_img_pred.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
                agent_cls_img_pred = agent_cls_img_pred.reshape(N, -1)  # N, H, W, C -> N, H*W*C

                agent_cls_img_weight = agent_cls_img_weight.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
                agent_cls_img_weight = agent_cls_img_weight.reshape(N, -1)  # N, H, W, C -> N, H*W*C

                agent_cls_img_gt = torch.zeros((2, agent_cls_img_pred.shape[1]), dtype=torch.float32,
                                                device=agent_cls_img_pred.device)
                agent_cls_img_gt[1] = 1
                agent_cls_img_gt = agent_cls_img_gt.repeat(N // 2, 1)

                agent_cls_img_loss = F.binary_cross_entropy_with_logits(agent_cls_img_pred, agent_cls_img_gt,
                                                                        weight=agent_cls_img_weight)
                # agent_cls_img_loss = agent_cls_img_loss * 1.0

                total_loss += agent_cls_img_loss
                self.loss_dict['agent_cls_img_loss'] = agent_cls_img_loss

            # if 'agent_feat_cls' in output_dict:
            #     agent_feat_cls_pred = output_dict['agent_feat_cls']  # N, C, H, W
            #     # agent_cls_img_weight = output_dict['agent_cls_img_weight']
            #     N = agent_feat_cls_pred.shape[0]
            #     agent_feat_cls_pred = agent_feat_cls_pred.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
            #     agent_feat_cls_pred = agent_feat_cls_pred.reshape(N, -1)  # N, H, W, C -> N, H*W*C
            #
            #     # agent_cls_img_weight = agent_cls_img_weight.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
            #     # agent_cls_img_weight = agent_cls_img_weight.reshape(N, -1)  # N, H, W, C -> N, H*W*C
            #
            #     agent_feat_cls_gt = torch.zeros((2, agent_feat_cls_pred.shape[1]), dtype=torch.float32,
            #                                    device=agent_feat_cls_pred.device)
            #     agent_feat_cls_gt[1] = 1
            #     agent_feat_cls_gt = agent_feat_cls_gt.repeat(N // 2, 1)
            #
            #     agent_feat_cls_loss = F.binary_cross_entropy_with_logits(agent_feat_cls_pred, agent_feat_cls_gt)
            #     agent_feat_cls_loss = agent_feat_cls_loss * 1.0
            #
            #     total_loss += agent_feat_cls_loss
            #     self.loss_dict['agent_feat_cls_loss'] = agent_feat_cls_loss
            #
            # if 'agent_ins_cls' in output_dict:
            #     agent_ins_cls_pred = output_dict['agent_ins_cls']  # N, H
            #     N = agent_ins_cls_pred.shape[0]
            #     agent_ins_cls_gt = torch.zeros((2, agent_ins_cls_pred.shape[1]), dtype=torch.float32,
            #                                    device=agent_ins_cls_pred.device)
            #     agent_ins_cls_gt[1] = 1
            #     agent_ins_cls_gt = agent_ins_cls_gt.repeat(N // 2, 1)
            #
            #     agent_ins_cls_loss = F.binary_cross_entropy_with_logits(agent_ins_cls_pred, agent_ins_cls_gt)
            #     agent_ins_cls_loss = agent_ins_cls_loss * 1.0
            #
            #     total_loss += agent_ins_cls_loss
            #     self.loss_dict['agent_ins_cls_loss'] = agent_ins_cls_loss

        if domain == 'target':
            self.loss_dict['total_loss'] += total_loss
        else:
            self.loss_dict['total_loss'] = total_loss

        return total_loss

    def cls_loss_func(self, input: torch.Tensor,
                      target: torch.Tensor,
                      weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2


    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        descriptions = [f'[Epoch {epoch}][{batch_id + 1}/{batch_len}]',
                        f'Loss: {self.loss_dict["total_loss"].item():.2f}',
                        f'Conf: {self.loss_dict["conf_loss"].item():.2f}',
                        f'Loc: {self.loss_dict["reg_loss"].item():.2f}']

        if 'conf_loss_pseudo' in self.loss_dict:
            descriptions.append(f'ConfP: {self.loss_dict["conf_loss_pseudo"].item():.2f}')
        if 'reg_loss_pseudo' in self.loss_dict:
            descriptions.append(f'LocP: {self.loss_dict["reg_loss_pseudo"].item():.2f}')

        if 'source_domain_cls_loss' in self.loss_dict:
            domain_cls_loss = self.loss_dict['source_domain_cls_loss'].item() + \
                                   self.loss_dict['target_domain_cls_loss'].item()
            descriptions.append(f'Dom: {domain_cls_loss:.2f}')
        # if 'source_domain_feat_cls_loss' in self.loss_dict:
        #     domain_feat_cls_loss = self.loss_dict['source_domain_feat_cls_loss'].item() + \
        #                         self.loss_dict['target_domain_feat_cls_loss'].item()
        #     descriptions.append(f'DomF: {domain_feat_cls_loss:.2f}')
        # if 'source_domain_ins_cls_loss' in self.loss_dict:
        #     domain_ins_cls_loss = self.loss_dict['source_domain_ins_cls_loss'].item() + \
        #                             self.loss_dict['target_domain_ins_cls_loss'].item()
        #     descriptions.append(f'DomI: {domain_ins_cls_loss:.2f}')
        if 'agent_cls_img_loss' in self.loss_dict:
            descriptions.append(f'Agt: {self.loss_dict["agent_cls_img_loss"].item():.2f}')
        # if 'agent_feat_cls_loss' in self.loss_dict:
        #     descriptions.append(f'AgtF: {self.loss_dict["agent_feat_cls_loss"].item():.2f}')
        # if 'agent_ins_cls_loss' in self.loss_dict:
        #     descriptions.append(f'AgtI: {self.loss_dict["agent_ins_cls_loss"].item():.2f}')

        description_str = ' | '.join(descriptions)
        if pbar is None:
            print(description_str)
        else:
            pbar.set_description(description_str)

        writer.add_scalar('Regression_loss', self.loss_dict['reg_loss'].item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('Confidence_loss', self.loss_dict['conf_loss'].item(),
                          epoch*batch_len + batch_id)

        if 'conf_loss_pseudo' in self.loss_dict:
            writer.add_scalar('Confidence_loss_pseudo', self.loss_dict['conf_loss_pseudo'].item(),
                              epoch*batch_len + batch_id)
        if 'reg_loss_pseudo' in self.loss_dict:
            writer.add_scalar('Regression_loss_pseudo', self.loss_dict['reg_loss_pseudo'].item(),
                              epoch*batch_len + batch_id)

        if 'source_domain_cls_loss' in self.loss_dict:
            writer.add_scalar('Domain_cls_loss', domain_cls_loss,
                              epoch*batch_len + batch_id)
        # if 'source_domain_feat_cls_loss' in self.loss_dict:
        #     writer.add_scalar('Domain_feat_cls_loss', domain_feat_cls_loss,
        #                       epoch*batch_len + batch_id)
        # if 'source_domain_ins_cls_loss' in self.loss_dict:
        #     writer.add_scalar('Domain_ins_cls_loss', domain_ins_cls_loss,
        #                       epoch*batch_len + batch_id)
        if 'agent_cls_img_loss' in self.loss_dict:
            writer.add_scalar('Agent_cls_img_loss', self.loss_dict['agent_cls_img_loss'].item(),
                              epoch*batch_len + batch_id)
        # if 'agent_feat_cls_loss' in self.loss_dict:
        #     writer.add_scalar('Agent_feat_cls_loss', self.loss_dict['agent_feat_cls_loss'].item(),
        #                       epoch*batch_len + batch_id)
        # if 'agent_ins_cls_loss' in self.loss_dict:
        #     writer.add_scalar('Agent_ins_cls_loss', self.loss_dict['agent_ins_cls_loss'].item(),
        #                       epoch*batch_len + batch_id)
