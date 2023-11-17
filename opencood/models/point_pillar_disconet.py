# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.da_modules.classifier import DAClsHead, DAImgHead
from opencood.models.fuse_modules.fuse_utils import extract_ego, regroup
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter


class PixelWeightLayer(nn.Module):
    def __init__(self, channel):
        super(PixelWeightLayer, self).__init__()

        self.conv1_1 = nn.Conv2d(channel * 2, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        # self.bn1_4 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))

        return x_1


class PointPillarDiscoNet(nn.Module):
    def __init__(self, args):
        super(PointPillarDiscoNet, self).__init__()
        self.max_cav = args['max_cav']
        self.discrete_ratio = args['voxel_size'][0]

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        self.pixel_weight_layer = PixelWeightLayer(128 * 3)

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

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
            self.domain_classifier = DAClsHead(384 + 2, -0.05)

        self.agent_cls_img_flag = False
        if 'agent_cls_img' in args:
            self.agent_cls_img_flag = True
            self.agent_classifier_img = DAImgHead(384 + 2, -0.1)

        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_number'],
                                  kernel_size=1)

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
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

        ########## FUSION START ##########
        # we concat ego's feature with other agent
        # N, C, H, W -> B, L, C, H, W
        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)

        B, _, C, H, W = regroup_feature.shape

        out = []

        for b in range(B):
            # number of valid agent
            num_cav = record_len[b]

            # (N, C, H, W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            neighbor_feature = regroup_feature[b][:num_cav]
            # (N, C, H, W)
            ego_feature = regroup_feature[b][0].view(1, C, H, W).expand(num_cav, -1, -1, -1)
            # (N, 2C, H, W)
            neighbor_feature_cat = torch.cat((neighbor_feature, ego_feature), dim=1)
            # (N, 1, H, W)
            agent_weight = self.pixel_weight_layer(neighbor_feature_cat)
            # (N, 1, H, W)
            agent_weight = F.softmax(agent_weight, dim=0)

            agent_weight = agent_weight.expand(-1, C, -1, -1)
            # (N, C, H, W)
            feature_fused = torch.sum(agent_weight * neighbor_feature, dim=0)
            out.append(feature_fused)

        spatial_features_2d = torch.stack(out)

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {'psm': psm,
                       'rm': rm}

        if self.domain_cls_flag:
            output_dict['domain_cls'] = domain_cls
        if self.agent_cls_img_flag and N % 2 == 0:
            output_dict['agent_cls_img'] = agent_cls_img
            output_dict['agent_cls_img_weight'] = agent_cls_img_weight

        return output_dict
