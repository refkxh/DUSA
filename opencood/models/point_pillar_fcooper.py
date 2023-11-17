# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
from opencood.models.fuse_modules.fuse_utils import extract_ego
from opencood.models.da_modules.classifier import DAClsHead, DAImgHead


class PointPillarFCooper(nn.Module):
    """
    F-Cooper implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(PointPillarFCooper, self).__init__()

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        # Spatial Map
        spatial_map_xs = torch.linspace(args['lidar_range'][0], args['lidar_range'][3], steps=256)
        spatial_map_ys = torch.linspace(args['lidar_range'][1], args['lidar_range'][4], steps=96)
        x, y = torch.meshgrid(spatial_map_xs, spatial_map_ys, indexing='xy')  # (96, 256), (96, 256)
        spatial_map_tmp = [y, x]
        spatial_map_tmp = torch.stack(spatial_map_tmp, dim=0)  # (2, 96, 256)
        spatial_map_tmp = spatial_map_tmp.unsqueeze(0)  # (1, 2, 96, 256)
        spatial_map_tmp = spatial_map_tmp.abs()
        self.register_buffer('spatial_map', spatial_map_tmp)

        # DA Classifiers
        self.domain_cls_flag = False
        if 'domain_cls' in args:
            self.domain_cls_flag = True
            self.domain_cls_weight = nn.Parameter(torch.ones((96, 256)))
            self.domain_classifier = DAClsHead(256 + 2, -0.05)

        self.agent_cls_img_flag = False
        if 'agent_cls_img' in args:
            self.agent_cls_img_flag = True
            self.agent_classifier_img = DAImgHead(256 + 2, -0.1)

        self.fusion_net = SpatialFusion()

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm_single = self.cls_head(spatial_features_2d)

        N = spatial_features_2d.shape[0]
        spatial_map = self.spatial_map.repeat(N, 1, 1, 1)
        spatial_features_2d_da = torch.cat([spatial_features_2d, spatial_map], dim=1)
        spatial_features_2d_da_ego = extract_ego(spatial_features_2d_da, record_len)

        if self.domain_cls_flag:
            domain_cls = self.domain_classifier(spatial_features_2d_da_ego, self.domain_cls_weight)
        agent_cls_weight = psm_single.detach().sigmoid().mean(dim=1, keepdim=True)
        if self.agent_cls_img_flag and N % 2 == 0:
            agent_cls_img = self.agent_classifier_img(spatial_features_2d_da)
            agent_cls_img_weight = torch.minimum(agent_cls_weight[::2], agent_cls_weight[1::2])
            agent_cls_img_weight = agent_cls_img_weight / (agent_cls_img_weight.max() + 1e-6)
            agent_cls_img_weight = agent_cls_img_weight.repeat_interleave(2, dim=0)

        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        fused_feature = self.fusion_net(spatial_features_2d, record_len)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        if self.domain_cls_flag:
            output_dict['domain_cls'] = domain_cls
        if self.agent_cls_img_flag and N % 2 == 0:
            output_dict['agent_cls_img'] = agent_cls_img
            output_dict['agent_cls_img_weight'] = agent_cls_img_weight

        return output_dict
