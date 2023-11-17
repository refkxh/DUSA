# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone
# from opencood.models.da_modules.classifier import DAFeatureHead, DAInstanceHead


class PointPillarIntermediate(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediate, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = AttBEVBackbone(args['base_bev_backbone'], 64)

        self.da_avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        # DA Classifiers
        self.domain_cls_flag = False
        if 'domain_cls' in args:
            self.domain_cls_flag = True
            self.domain_feature_classifier = DAFeatureHead(384, -0.1)
            self.domain_instance_classifier = DAInstanceHead(128, -0.1)

        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
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

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        if self.domain_cls_flag:
            domain_feat_cls = self.domain_feature_classifier(spatial_features_2d)

            domain_psm = psm.permute(0, 2, 3, 1).contiguous()  # (B, H, W, A) (B, 96, 256, 2)
            domain_psm = self.da_avgpool(domain_psm)  # (B, H, W/2, A/2) (B, 96, 128, 1)
            domain_psm = domain_psm.view(domain_psm.shape[0] * domain_psm.shape[1], -1) # (B*H, W/2*A/2) (B*96, 128)
            domain_ins_cls = self.domain_instance_classifier(domain_psm)  # (B*H, 1) (B*96, 1)
            domain_ins_cls = domain_ins_cls.view(psm.shape[0], -1)  # (B, H) (B, 96)

        output_dict = {'psm': psm,
                       'rm': rm}

        if self.domain_cls_flag:
            output_dict['domain_feat_cls'] = domain_feat_cls
            output_dict['domain_ins_cls'] = domain_ins_cls

        return output_dict