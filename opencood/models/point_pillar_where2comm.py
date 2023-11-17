# import torch
import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.where2comm_fuse import Where2comm
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
# from opencood.models.da_modules.classifier import DAFeatureHead, DAInstanceHead
# from opencood.models.fuse_modules.fuse_utils import extract_ego


class PointPillarWhere2comm(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2comm, self).__init__()
        self.max_cav = args['max_cav']
        # Pillar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # Used to down-sample the feature map for efficient computation
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False

        if args['compression']:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        else:
            self.compression = False

        # Spatial Map
        # spatial_map_xs = torch.linspace(args['lidar_range'][0], args['lidar_range'][3], steps=128)
        # spatial_map_ys = torch.linspace(args['lidar_range'][1], args['lidar_range'][4], steps=48)
        # x, y = torch.meshgrid(spatial_map_xs, spatial_map_ys, indexing='xy')  # (48, 128), (48, 128)
        # spatial_map_tmp = [y, x]
        # spatial_map_tmp = torch.stack(spatial_map_tmp, dim=0)  # (2, 48, 128)
        # spatial_map_tmp = spatial_map_tmp.unsqueeze(0)  # (1, 2, 48, 128)
        # spatial_map_tmp = spatial_map_tmp.abs()
        # self.register_buffer('spatial_map', spatial_map_tmp)

        # DA Classifiers
        self.domain_cls_flag = False
        if 'domain_cls' in args:
            self.domain_cls_flag = True
            self.domain_feature_classifier = DAFeatureHead(256, -0.1)
            self.domain_instance_classifier = DAInstanceHead(64, -0.1)

        self.agent_cls_flag = False
        if 'agent_cls' in args:
            self.agent_cls_flag = True
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.agent_feature_classifier = DAFeatureHead(256, -0.5)
            self.agent_instance_classifier = DAInstanceHead(64, -0.5)

        # self.domain_cls_flag = False
        # if 'domain_cls' in args:
        #     self.domain_cls_flag = True
        #     self.domain_cls_weight = nn.Parameter(torch.ones((48, 128)))
        #     self.domain_classifier = DAClsHead(256 + 2, -1)
        #
        # self.domain_cls_img_flag = False
        # if 'domain_cls_img' in args:
        #     self.domain_cls_img_flag = True
        #     self.domain_classifier_img = DAImgHead(256 + 2, -1)
        #
        # self.agent_cls_flag = False
        # if 'agent_cls' in args:
        #     self.agent_cls_flag = True
        #     self.agent_classifier = DAClsHead(256 + 2, -1)
        #
        # self.agent_cls_img_flag = False
        # if 'agent_cls_img' in args:
        #     self.agent_cls_img_flag = True
        #     self.agent_classifier_img = DAImgHead(256 + 2, -1)

        self.fusion_net = Where2comm(args['where2comm_fusion'])
        self.multi_scale = args['where2comm_fusion']['multi_scale']

        self.cls_head = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'], kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay.
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
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        # N, C, H', W': [N, 256, 48, 176]
        spatial_features_2d = batch_dict['spatial_features_2d']
        # Down-sample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm_single = self.cls_head(spatial_features_2d)

        if self.agent_cls_flag:
            agent_feat_cls = self.agent_feature_classifier(spatial_features_2d)

            agent_psm_single = psm_single.permute(0, 2, 3, 1).contiguous()  # (B, H, W, A) (B, 48, 128, 2)
            agent_psm_single = self.avgpool(agent_psm_single)  # (B, H, W/2, A/2) (B, 48, 64, 1)
            agent_psm_single = agent_psm_single.view(agent_psm_single.shape[0] * agent_psm_single.shape[1], -1) # (B*H, W/2*A/2) (B*48, 64)
            agent_ins_cls = self.agent_instance_classifier(agent_psm_single)  # (B*H, 1) (B*48, 1)
            agent_ins_cls = agent_ins_cls.view(psm_single.shape[0], -1)  # (B, H) (B, 48)

        # batch_size = spatial_features_2d.shape[0]
        # spatial_map = self.spatial_map.repeat(batch_size, 1, 1, 1)
        # spatial_features_2d_da = torch.cat([spatial_features_2d, spatial_map], dim=1)
        # spatial_features_2d_da_ego = extract_ego(spatial_features_2d_da, record_len)
        #
        # if self.domain_cls_flag:
        #     domain_cls = self.domain_classifier(spatial_features_2d_da_ego, self.domain_cls_weight)
        # if self.domain_cls_img_flag:
        #     domain_cls_img = self.domain_classifier_img(spatial_features_2d_da_ego)
        # agent_cls_weight = psm_single.detach().sigmoid().mean(dim=1, keepdim=True)
        # if self.agent_cls_flag:
        #     agent_cls = self.agent_classifier(spatial_features_2d_da, agent_cls_weight)
        # if self.agent_cls_img_flag:
        #     agent_cls_img = self.agent_classifier_img(spatial_features_2d_da)
        #     agent_cls_img_weight, _ = agent_cls_weight.min(dim=0, keepdim=True)
        #     agent_cls_img_weight = agent_cls_img_weight / (agent_cls_img_weight.max() + 1e-6)
        #     agent_cls_img_weight = agent_cls_img_weight.repeat(batch_size, 1, 1, 1)

        # Compressor
        if self.compression:
            # The ego feature is also compressed
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        if self.multi_scale:
            # Bypass communication cost, communicate at high resolution, neither shrink nor compress
            fused_feature, communication_rates = self.fusion_net(batch_dict['spatial_features'],
                                                                 psm_single,
                                                                 record_len,
                                                                 pairwise_t_matrix,
                                                                 self.backbone)
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates = self.fusion_net(spatial_features_2d,
                                                                 psm_single,
                                                                 record_len,
                                                                 pairwise_t_matrix)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        if self.domain_cls_flag:
            domain_feat_cls = self.domain_feature_classifier(fused_feature)

            domain_psm = psm.permute(0, 2, 3, 1).contiguous()  # (B, H, W, A) (B, 48, 128, 2)
            domain_psm = self.avgpool(domain_psm)  # (B, H, W/2, A/2) (B, 48, 64, 1)
            domain_psm = domain_psm.view(domain_psm.shape[0] * domain_psm.shape[1], -1) # (B*H, W/2*A/2) (B*48, 64)
            domain_ins_cls = self.domain_instance_classifier(domain_psm)  # (B*H, 1) (B*48, 1)
            domain_ins_cls = domain_ins_cls.view(psm.shape[0], -1)  # (B, H) (B, 48)

        output_dict = {'psm': psm, 'rm': rm, 'com': communication_rates}

        if self.domain_cls_flag:
            output_dict['domain_feat_cls'] = domain_feat_cls
            output_dict['domain_ins_cls'] = domain_ins_cls

        if self.agent_cls_flag:
            output_dict['agent_feat_cls'] = agent_feat_cls
            output_dict['agent_ins_cls'] = agent_ins_cls

        # if self.domain_cls_flag:
        #     output_dict['domain_cls'] = domain_cls
        # if self.domain_cls_img_flag:
        #     output_dict['domain_cls_img'] = domain_cls_img
        # if self.agent_cls_flag:
        #     output_dict['agent_cls'] = agent_cls
        # if self.agent_cls_img_flag:
        #     output_dict['agent_cls_img'] = agent_cls_img
        #     output_dict['agent_cls_img_weight'] = agent_cls_img_weight

        return output_dict
