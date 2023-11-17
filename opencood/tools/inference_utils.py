# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from collections import OrderedDict

import json
import numpy as np
import torch

from opencood.utils.box_utils import corner_to_center, project_box3d
from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.utils.transformation_utils import x_to_world


def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']

    output_dict['ego'] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return inference_early_fusion(batch_data, model, dataset)


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy_test' % timestamp), gt_np)


def save_pseudo_label_dair(pred_tensor, pred_score, ego_lidar_pose, frame_id, save_dir):
    """
    Save prediction tensor to yaml file.
    For dair-v2x dataset.
    """
    pose_np = torch_tensor_to_numpy(ego_lidar_pose)
    transformation_matrix = x_to_world(pose_np)

    if pred_tensor is not None:
        pred_np = torch_tensor_to_numpy(pred_tensor)
        world_8_points = project_box3d(pred_np, transformation_matrix)
        world_center_boxes = corner_to_center(world_8_points, order='hwl')
    else:
        world_center_boxes = np.zeros((0, 7))
        world_8_points = np.zeros((0, 8, 3))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_list = []
    for i in range(len(world_center_boxes)):
        center_box = world_center_boxes[i].tolist()
        eight_points = world_8_points[i].tolist()
        save_list.append({"type": "car",
                          "3d_dimensions": {"h": center_box[3], "w": center_box[4], "l": center_box[5]},
                          "3d_location": {"x": center_box[0], "y": center_box[1], "z": center_box[2]},
                          "rotation": center_box[6],
                          "world_8_points": eight_points,
                          "score": pred_score[i].item()})

    with open(os.path.join(save_dir, f'{frame_id}.json'), 'w') as f:
        json.dump(save_list, f)
